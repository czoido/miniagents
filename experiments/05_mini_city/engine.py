"""Simulation engine — generation, interaction, actions, overnight, chronicles."""

import random
import re
import time

from mlx_lm.sample_utils import make_sampler

from .prompts import (
    ACTION_SYSTEM,
    CITIZEN_SYSTEM,
    EVENT_SYSTEM,
    FAREWELL_HINT,
    FAREWELL_WORDS,
    GOAL_UPDATE_SYSTEM,
    INTERACTION_RULES,
    NARRATOR_SYSTEM,
    OVERNIGHT_GOALS_SYSTEM,
    OVERNIGHT_MEMORY_SYSTEM,
    PREMISE_SYSTEM,
    SUMMARY_SYSTEM,
)
from .world import Action, Citizen, DEFAULT_PREMISE, Event, Premise, TimeSlot, EVENT_SEED_POOL

_STYLES = ["red", "green", "magenta", "yellow", "bright_black", "cyan",
           "blue", "bright_red", "bright_green", "bright_magenta"]


# ---------------------------------------------------------------------------
# Low-level generation
# ---------------------------------------------------------------------------


def _msg(role: str, text: str) -> dict:
    return {"role": role, "content": [{"type": "text", "text": text}]}


def generate(
    model, system: str, user_msg: str, temp: float, max_tokens: int,
) -> tuple[str, float]:
    sampler = make_sampler(temp=temp, top_p=0.95, top_k=30)
    t0 = time.perf_counter()
    response = model.generate(
        [_msg("system", system), _msg("user", user_msg)],
        max_tokens=max_tokens,
        sampler=sampler,
    )
    elapsed = time.perf_counter() - t0
    return (response.content or "").strip(), elapsed


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def format_goals(citizen: Citizen) -> str:
    if not citizen.goals:
        return "No specific goals right now."
    return "\n".join(f"- {g}" for g in citizen.goals)


def format_transcript(transcript: list[tuple[str, str]]) -> str:
    return "\n".join(f"{name}: {line}" for name, line in transcript)


def _recent_events(citizen: Citizen, max_events: int = 5) -> str:
    if not citizen.memory:
        return "Nothing notable has happened yet today."
    recent = citizen.memory[-max_events:]
    return "\n".join(f"- {e}" for e in recent)


# ---------------------------------------------------------------------------
# Dialogue guards
# ---------------------------------------------------------------------------


def _is_farewell(text: str) -> bool:
    low = text.lower()
    return any(fw in low for fw in FAREWELL_WORDS)


def _is_repetitive(line: str, transcript: list[tuple[str, str]]) -> bool:
    if len(transcript) < 2:
        return False
    low = line.lower().strip()
    for _, prev in transcript[-4:]:
        prev_low = prev.lower().strip()
        if low == prev_low:
            return True
        shorter, longer = sorted([low, prev_low], key=len)
        if shorter and len(shorter) > 20 and shorter in longer:
            return True
    return False


# ---------------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------------


def select_pairs(
    citizens: list[Citizen], count: int, rng: random.Random,
) -> list[tuple[Citizen, Citizen]]:
    """Select random unique pairs for this hour."""
    pool = list(citizens)
    rng.shuffle(pool)
    pairs: list[tuple[Citizen, Citizen]] = []
    used: set[str] = set()
    for i in range(0, len(pool) - 1, 2):
        if len(pairs) >= count:
            break
        a, b = pool[i], pool[i + 1]
        if a.name not in used and b.name not in used:
            pairs.append((a, b))
            used.update({a.name, b.name})
    return pairs


# ---------------------------------------------------------------------------
# Goal management
# ---------------------------------------------------------------------------


def update_goals(model, citizen: Citizen, summary: str) -> None:
    """Update a citizen's goals based on what just happened."""
    current = format_goals(citizen)
    prompt = (
        f"Character: {citizen.name} the {citizen.role}, has ${citizen.money}\n"
        f"Personality: {citizen.personality.split('.')[0]}.\n"
        f"Current goals:\n{current}\n\n"
        f"What just happened: {summary}\n\n"
        f"Write ONLY {citizen.name}'s updated goals (not anyone else's)."
    )
    raw, _ = generate(model, GOAL_UPDATE_SYSTEM, prompt, 0.2, 120)
    new_goals = [
        line.lstrip("-•* 0123456789.)")
        for line in raw.strip().splitlines()
        if line.strip()
    ]
    if new_goals:
        citizen.goals = new_goals[:4]


# ---------------------------------------------------------------------------
# Interaction
# ---------------------------------------------------------------------------


def run_interaction(
    model,
    a: Citizen,
    b: Citizen,
    hour: int,
    slot: TimeSlot,
    max_tokens: int,
    day_event: str = "",
    discussed_topics: list[str] | None = None,
    premise: Premise | None = None,
) -> Event:
    """Run a multi-turn conversation between two citizens."""
    max_turns = 6
    dialogue_tokens = min(max_tokens, 80)
    t0 = time.perf_counter()

    village_ctx = f" in {premise.village}" if premise else ""
    time_context = f"It's {slot.time}{village_ctx}. You're at {slot.location}."
    news_hint = f"\nToday's news: {day_event}" if day_event else ""
    avoid_hint = ""
    if discussed_topics:
        avoid_hint = (
            "\nTopics already discussed today (bring up something DIFFERENT): "
            + "; ".join(discussed_topics[-6:])
        )

    speakers = [a, b]
    transcript: list[tuple[str, str]] = []

    for turn in range(max_turns):
        speaker = speakers[turn % 2]
        listener = speakers[(turn + 1) % 2]

        system = (
            f"{speaker.personality}\n\n"
            f"You have ${speaker.money} in your pocket.\n"
            f"Your goals for today:\n{format_goals(speaker)}\n\n"
            f"{INTERACTION_RULES}"
        )

        if turn == 0:
            heard = _recent_events(speaker)
            user = (
                f"{time_context} You see {listener.name} the {listener.role}."
                f"{news_hint}{avoid_hint}\n"
                f"Things you've heard today:\n{heard}\n\n"
                f"What do you say? (1-3 sentences, no narration)"
            )
        else:
            history = format_transcript(transcript)
            closing = FAREWELL_HINT if turn >= 3 else ""
            user = (
                f"{time_context}\n"
                f"Conversation so far:\n{history}\n\n"
                f"What do you say next? (1-3 sentences, no narration){closing}"
            )

        line, _ = generate(model, system, user, speaker.temp, dialogue_tokens)
        line = line.strip('"\'').strip()

        if _is_repetitive(line, transcript):
            break

        transcript.append((speaker.name, line))

        if turn >= 2 and _is_farewell(line):
            break

    conv_text = format_transcript(transcript)
    conversation = f"At {slot.time} near {slot.location}:\n{conv_text}"
    summary, _ = generate(model, SUMMARY_SYSTEM, conversation, 0.3, 80)

    a.memory.append(summary)
    b.memory.append(summary)

    update_goals(model, a, summary)
    update_goals(model, b, summary)

    elapsed = time.perf_counter() - t0

    return Event(
        hour=hour,
        time_label=slot.time,
        location=slot.location,
        participants=[a.name, b.name],
        transcript=transcript,
        summary=summary,
        elapsed_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Daily actions
# ---------------------------------------------------------------------------


def resolve_daily_actions(
    model, citizens: list[Citizen], rng: random.Random,
) -> list[Action]:
    """Each citizen either works or buys something at the store."""
    actions: list[Action] = []
    shuffled = list(citizens)
    rng.shuffle(shuffled)

    for c in shuffled:
        # Daily income always applies first
        if c.income > 0:
            c.money += c.income

        goals_text = format_goals(c)
        mem_text = "\n".join(c.memory[-5:]) if c.memory else "Nothing happened."
        prompt = (
            f"Character: {c.name} the {c.role}. Has ${c.money}.\n"
            f"Daily income from job: ${c.income} (already earned today)\n"
            f"Goals:\n{goals_text}\n"
            f"What happened today:\n{mem_text}\n\n"
            f"What does {c.name} do? BUY something or NOTHING."
        )
        raw, _ = generate(model, ACTION_SYSTEM, prompt, 0.4, 60)
        line = raw.strip().splitlines()[0] if raw.strip() else "NOTHING"
        line = re.split(r"\bor\b", line, flags=re.IGNORECASE)[0].strip()
        line = line.strip('"\'')

        action = Action(citizen=c.name, description=line)
        if c.income > 0:
            action.earned = c.income

        dollars = re.findall(r"\$(\d+)", line)
        amount = int(dollars[0]) if dollars else 0

        upper = line.upper()
        if upper.startswith("BUY") and amount > 0:
            cost = min(amount, c.money)
            c.money -= cost
            action.cost = cost
            c.memory.append(
                f"Earned ${c.income} from work. Bought: {line} "
                f"(spent ${cost}, have ${c.money} left)"
            )
        else:
            c.memory.append(
                f"Earned ${c.income} from work (now have ${c.money})"
            )

        update_goals(model, c, action.description)
        actions.append(action)

    return actions


# ---------------------------------------------------------------------------
# Overnight phase
# ---------------------------------------------------------------------------


def overnight(model, citizens: list[Citizen], day_events: list[Event]) -> None:
    """Compress memories and refresh goals between days."""
    for c in citizens:
        if not c.memory:
            continue
        mem_text = "\n".join(c.memory)
        prompt = (
            f"Character: {c.name} ({c.role})\n"
            f"Everything they experienced today:\n{mem_text}\n\n"
            f"Write the key facts they'll remember tomorrow."
        )
        raw, _ = generate(model, OVERNIGHT_MEMORY_SYSTEM, prompt, 0.2, 150)
        compressed = [
            ln.lstrip("-•* ") for ln in raw.strip().splitlines() if ln.strip()
        ]
        c.memory = compressed[:5] if compressed else c.memory[-3:]

    for c in citizens:
        current = format_goals(c)
        mem_text = "\n".join(c.memory)
        prompt = (
            f"Character: {c.name} the {c.role}, has ${c.money}\n"
            f"Personality: {c.personality.split('.')[0]}.\n"
            f"What they remember from today:\n{mem_text}\n"
            f"Current goals:\n{current}\n\n"
            f"Write ONLY {c.name}'s goals for tomorrow (not anyone else's)."
        )
        raw, _ = generate(model, OVERNIGHT_GOALS_SYSTEM, prompt, 0.3, 120)
        new_goals = [
            ln.lstrip("-•* 0123456789.)") for ln in raw.strip().splitlines()
            if ln.strip()
        ]
        if new_goals:
            c.goals = new_goals[:4]


# ---------------------------------------------------------------------------
# Chronicles
# ---------------------------------------------------------------------------


def day_chronicle(
    model,
    citizens: list[Citizen],
    day_events: list[Event],
    day_num: int,
    actions: list[Action] | None = None,
) -> tuple[str, float]:
    """Generate a factual chronicle for one day."""
    event_blocks = []
    for e in day_events:
        conv = format_transcript(e.transcript)
        event_blocks.append(
            f"[{e.time_label} at {e.location}]\n{conv}\nSummary: {e.summary}"
        )
    event_log = "\n\n".join(event_blocks)

    citizen_roster = "\n".join(
        f"- {c.name} ({c.role}, ${c.money}): {c.personality.split('.')[0]}."
        for c in citizens
    )

    actions_text = ""
    if actions:
        lines = [f"- {a.citizen}: {a.description}" for a in actions]
        actions_text = f"\n\nEnd-of-day actions:\n" + "\n".join(lines)

    prompt = (
        f"Village citizens:\n{citizen_roster}\n\n"
        f"Conversations of day {day_num}:\n{event_log}"
        f"{actions_text}\n\n"
        f"Summarize what happened today."
    )

    chronicle, elapsed = generate(model, NARRATOR_SYSTEM, prompt, 0.3, 500)
    return chronicle.strip(), elapsed


# ---------------------------------------------------------------------------
# Random event generation
# ---------------------------------------------------------------------------


def generate_event(
    model, citizens: list[Citizen], previous_events: list[str],
    rng: random.Random, premise: Premise | None = None,
) -> str:
    """Generate a contextual random event based on village history."""
    setting = f"Setting: {premise.summary()}\n" if premise else ""

    if not previous_events:
        prompt = (
            f"{setting}Villagers: "
            + ", ".join(f"{c.name} ({c.role})" for c in citizens)
            + "\n\nGenerate ONE surprising event for the first morning."
        )
        raw, _ = generate(model, EVENT_SYSTEM, prompt, 0.9, 50)
        line = raw.strip().splitlines()[0] if raw.strip() else rng.choice(EVENT_SEED_POOL)
        return line

    citizen_names = ", ".join(f"{c.name} ({c.role}, ${c.money})" for c in citizens)
    prev = "\n".join(f"- {e}" for e in previous_events)
    recent_memories = []
    for c in citizens:
        if c.memory:
            recent_memories.append(f"{c.name}: {c.memory[-1]}")
    context = "\n".join(recent_memories[-4:]) if recent_memories else "Nothing yet."

    prompt = (
        f"{setting}Villagers: {citizen_names}\n\n"
        f"Previous events (do NOT repeat):\n{prev}\n\n"
        f"Recent village happenings:\n{context}\n\n"
        f"Generate ONE new surprising event for this morning."
    )
    raw, _ = generate(model, EVENT_SYSTEM, prompt, 0.9, 50)
    line = raw.strip().splitlines()[0] if raw.strip() else rng.choice(EVENT_SEED_POOL)
    return line


# ---------------------------------------------------------------------------
# Dynamic premise generation
# ---------------------------------------------------------------------------


def generate_premise(model) -> Premise:
    """Generate a unique village setting."""
    raw, _ = generate(model, PREMISE_SYSTEM, "Create a village setting.", 0.95, 150)
    fields: dict[str, str] = {}
    for line in raw.strip().splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            fields[key.strip().upper()] = val.strip().strip('"\'')

    village = fields.get("VILLAGE", "").strip()
    region = fields.get("REGION", "").strip()
    era = fields.get("ERA", "").strip()
    mood = fields.get("MOOD", "").strip()
    store = fields.get("STORE", "").strip()

    if village and region:
        return Premise(
            village=village,
            region=region,
            era=era or "present-day",
            mood=mood or "Things are changing.",
            store=store or f"{village} General Store",
        )
    return DEFAULT_PREMISE


# ---------------------------------------------------------------------------
# Dynamic citizen generation
# ---------------------------------------------------------------------------


def _parse_citizen(raw: str, index: int) -> Citizen | None:
    """Parse a model-generated citizen block into a Citizen object."""
    fields: dict[str, str] = {}
    for line in raw.strip().splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            fields[key.strip().upper()] = val.strip()

    name = fields.get("NAME", "").strip().strip('"\'')
    role = fields.get("ROLE", "").strip().strip('"\'')
    personality_raw = fields.get("PERSONALITY", "").strip()
    age = fields.get("AGE", "30").strip()

    if not name or not role or not personality_raw:
        return None

    # Strip role from name: "Kaelen the Cartographer" → "Kaelen"
    for pattern in [f" the {role}", f" The {role}", f" the {role.lower()}"]:
        if pattern in name:
            name = name.replace(pattern, "").strip()
    name = name.split(" the ")[0].strip()
    name = name.split(",")[0].strip()

    personality = (
        f"You are {name}, {age}, the village {role.lower()}. {personality_raw} "
        f"You always speak in first person as yourself."
    )

    goals = []
    for k in ("GOAL1", "GOAL2", "GOAL3"):
        g = fields.get(k, "").strip()
        if g:
            goals.append(g)
    if not goals:
        return None

    try:
        money = int(re.sub(r"[^\d]", "", fields.get("MONEY", "100")) or "100")
    except ValueError:
        money = 100
    try:
        income = int(re.sub(r"[^\d]", "", fields.get("INCOME", "0")) or "0")
    except ValueError:
        income = 0

    money = max(10, min(500, money))
    income = max(0, min(80, income))

    style = _STYLES[index % len(_STYLES)]
    temp_map = {"Baker": 0.85, "Doctor": 0.5, "Artist": 0.9, "Mayor": 0.7}
    temp = temp_map.get(role, 0.7 + (index % 3) * 0.1)

    return Citizen(
        name=name,
        role=role,
        personality=personality,
        goals=goals,
        style=style,
        temp=temp,
        money=money,
        income=income,
    )


def generate_citizens(model, count: int, premise: Premise | None = None) -> list[Citizen]:
    """Generate N unique citizens using the model."""
    citizens: list[Citizen] = []
    used_names: set[str] = set()
    used_roles: set[str] = set()
    setting = f"Setting: {premise.summary()}\n\n" if premise else ""

    for i in range(count):
        existing = ""
        if citizens:
            lines = [f"- {c.name} the {c.role}" for c in citizens]
            existing = f"Existing villagers (do NOT repeat):\n" + "\n".join(lines) + "\n\n"

        prompt = f"{setting}{existing}Create villager #{i + 1} of {count}."
        raw, _ = generate(model, CITIZEN_SYSTEM, prompt, 0.9, 200)
        citizen = _parse_citizen(raw, i)

        if citizen and citizen.name not in used_names and citizen.role not in used_roles:
            used_names.add(citizen.name)
            used_roles.add(citizen.role)
            citizens.append(citizen)
        elif i < count + 3:
            prompt = f"{setting}{existing}Create a COMPLETELY DIFFERENT villager #{i + 1}."
            raw, _ = generate(model, CITIZEN_SYSTEM, prompt, 0.95, 200)
            citizen = _parse_citizen(raw, i)
            if citizen and citizen.name not in used_names:
                used_names.add(citizen.name)
                used_roles.add(citizen.role)
                citizens.append(citizen)

    return citizens
