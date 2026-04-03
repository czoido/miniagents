"""Simulation engine — generation, interaction, actions, overnight, chronicles."""

import random
import re
import time

from mlx_lm.sample_utils import make_sampler

from .prompts import (
    ACTION_SYSTEM,
    CHOOSE_TARGET_SYSTEM,
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
    SCHEDULE_SYSTEM,
    SUMMARY_SYSTEM,
)
from .world import (
    Action, Citizen, DAY_SCHEDULE, DEFAULT_PREMISE,
    Event, Premise, TimeSlot, EVENT_SEED_POOL,
)

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
# Group formation — citizens choose who they want to talk to
# ---------------------------------------------------------------------------


def _choose_target(model, citizen: Citizen, others: list[Citizen]) -> str:
    """Ask a citizen who they want to talk to."""
    others_list = "\n".join(
        f"- {c.name} ({c.role})" for c in others
    )
    rels = ""
    for c in others:
        r = citizen.relationships.get(c.name)
        if r:
            rels += f"- {c.name}: {r}\n"

    goals_text = format_goals(citizen)
    recent = _recent_events(citizen)

    prompt = (
        f"You are {citizen.name} the {citizen.role}.\n"
        f"Your goals:\n{goals_text}\n"
        f"Recent events:\n{recent}\n"
    )
    if rels:
        prompt += f"Your relationships:\n{rels}"
    prompt += (
        f"\nWho do you want to talk to?\n{others_list}\n\n"
        f"Write ONLY one name."
    )
    raw, _ = generate(model, CHOOSE_TARGET_SYSTEM, prompt, 0.3, 10)
    chosen = raw.strip().splitlines()[0].strip() if raw.strip() else ""
    # Match to an actual name
    for c in others:
        if c.name.lower() in chosen.lower():
            return c.name
    return others[0].name if others else ""


def form_groups(
    model, citizens: list[Citizen], rng: random.Random,
) -> list[list[Citizen]]:
    """Each citizen chooses who to talk to; form groups from the choices."""
    cmap = {c.name: c for c in citizens}
    choices: dict[str, str] = {}

    for c in citizens:
        others = [o for o in citizens if o.name != c.name]
        if others:
            target = _choose_target(model, c, others)
            choices[c.name] = target

    # Build groups: if A→B and C→B, they all meet at B
    assigned: set[str] = set()
    groups: list[list[Citizen]] = []

    targets_of: dict[str, list[str]] = {}
    for seeker, target in choices.items():
        targets_of.setdefault(target, []).append(seeker)

    # Process most-sought people first (they attract groups)
    popular = sorted(targets_of.items(), key=lambda x: -len(x[1]))
    for target_name, seekers in popular:
        group_names = {target_name} | {s for s in seekers}
        group_names -= assigned
        if len(group_names) >= 2:
            group = [cmap[n] for n in group_names if n in cmap]
            rng.shuffle(group)
            groups.append(group)
            assigned.update(group_names)

    # Remaining unassigned: pair them up or let them join existing groups
    leftover = [c for c in citizens if c.name not in assigned]
    if len(leftover) >= 2:
        rng.shuffle(leftover)
        for i in range(0, len(leftover) - 1, 2):
            groups.append([leftover[i], leftover[i + 1]])
    elif len(leftover) == 1 and groups:
        groups[-1].append(leftover[0])

    return groups


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
    group: list[Citizen],
    hour: int,
    slot: TimeSlot,
    max_tokens: int,
    day_event: str = "",
    discussed_topics: list[str] | None = None,
    premise: Premise | None = None,
) -> Event:
    """Run a multi-turn conversation among N citizens (round-robin)."""
    n = len(group)
    max_turns = max(6, n * 2)
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

    others_names = lambda s: ", ".join(c.name for c in group if c.name != s.name)
    transcript: list[tuple[str, str]] = []

    for turn in range(max_turns):
        speaker = group[turn % n]
        present = others_names(speaker)

        secret_ctx = (
            f"\nYour SECRET (protect this at all costs): {speaker.secret}\n"
            if speaker.secret else ""
        )
        rel_lines = []
        for c in group:
            if c.name == speaker.name:
                continue
            r = speaker.relationships.get(c.name)
            if r:
                rel_lines.append(f"- {c.name}: {r}")
        rel_ctx = f"\nYour history:\n" + "\n".join(rel_lines) + "\n" if rel_lines else ""

        system = (
            f"{speaker.personality}\n\n"
            f"REMEMBER: You ARE {speaker.name}. Never refer to yourself "
            f"in the third person. The others present are: {present}.\n\n"
            f"You have ${speaker.money} in your pocket.\n"
            f"Your goals for today:\n{format_goals(speaker)}\n"
            f"{secret_ctx}{rel_ctx}\n"
            f"{INTERACTION_RULES}"
        )

        if turn == 0:
            heard = _recent_events(speaker)
            who = " and ".join(c.name + " the " + c.role for c in group if c.name != speaker.name)
            user = (
                f"{time_context} You ({speaker.name}) see {who}."
                f"{news_hint}{avoid_hint}\n"
                f"Things you've heard today:\n{heard}\n\n"
                f"Respond as {speaker.name}. (1-3 sentences, no narration)"
            )
        else:
            history = format_transcript(transcript)
            closing = FAREWELL_HINT if turn >= n * 2 else ""
            user = (
                f"{time_context}\n"
                f"Conversation so far:\n{history}\n\n"
                f"Respond as {speaker.name}. (1-3 sentences, no narration){closing}"
            )

        line, _ = generate(model, system, user, speaker.temp, dialogue_tokens)
        line = line.strip('"\'').strip()

        if _is_repetitive(line, transcript):
            break

        transcript.append((speaker.name, line))

        if turn >= n and _is_farewell(line):
            break

    conv_text = format_transcript(transcript)
    conversation = f"At {slot.time} near {slot.location}:\n{conv_text}"
    summary, _ = generate(model, SUMMARY_SYSTEM, conversation, 0.3, 80)

    for c in group:
        c.memory.append(summary)
        update_goals(model, c, summary)

    elapsed = time.perf_counter() - t0

    return Event(
        hour=hour,
        time_label=slot.time,
        location=slot.location,
        participants=[c.name for c in group],
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


def generate_premise(model, user_setup: str = "") -> Premise:
    """Generate a unique village setting, optionally guided by user input."""
    if user_setup:
        prompt = (
            f"The user described this scenario:\n\"{user_setup}\"\n\n"
            f"Create a village setting that fits this description."
        )
    else:
        prompt = "Create a village setting."
    raw, _ = generate(model, PREMISE_SYSTEM, prompt, 0.95, 150)
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


def generate_schedule(model, premise: Premise) -> list[TimeSlot]:
    """Generate locations that fit the premise instead of using hardcoded ones."""
    prompt = f"Setting: {premise.summary()}\n\nGenerate 10 time slots for this setting."
    raw, _ = generate(model, SCHEDULE_SYSTEM, prompt, 0.8, 400)

    slots: list[TimeSlot] = []
    for line in raw.strip().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 4:
            slots.append(TimeSlot(
                time=parts[0], period=parts[1],
                location=parts[2], atmosphere=parts[3],
            ))

    if len(slots) >= 3:
        return slots
    return DAY_SCHEDULE


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

    secret = fields.get("SECRET", "").strip()

    personality = (
        f"You are {name}, {age}, the village {role.lower()}. {personality_raw}"
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
        secret=secret,
        style=style,
        temp=temp,
        money=money,
        income=income,
    )


def generate_citizens(
    model, count: int, premise: Premise | None = None, user_setup: str = "",
) -> list[Citizen]:
    """Generate N unique citizens using the model."""
    citizens: list[Citizen] = []
    used_names: set[str] = set()
    used_roles: set[str] = set()
    setting = f"Setting: {premise.summary()}\n\n" if premise else ""
    constraint = f"User request: \"{user_setup}\"\nFollow this description closely.\n\n" if user_setup else ""

    for i in range(count):
        existing = ""
        if citizens:
            lines = [f"- {c.name} the {c.role}" for c in citizens]
            existing = f"Existing villagers (do NOT repeat):\n" + "\n".join(lines) + "\n\n"

        prompt = f"{setting}{constraint}{existing}Create villager #{i + 1} of {count}."
        raw, _ = generate(model, CITIZEN_SYSTEM, prompt, 0.9, 200)
        citizen = _parse_citizen(raw, i)

        if citizen and citizen.name not in used_names and citizen.role not in used_roles:
            used_names.add(citizen.name)
            used_roles.add(citizen.role)
            citizens.append(citizen)
        elif i < count + 3:
            prompt = f"{setting}{constraint}{existing}Create a COMPLETELY DIFFERENT villager #{i + 1}."
            raw, _ = generate(model, CITIZEN_SYSTEM, prompt, 0.95, 200)
            citizen = _parse_citizen(raw, i)
            if citizen and citizen.name not in used_names:
                used_names.add(citizen.name)
                used_roles.add(citizen.role)
                citizens.append(citizen)

    if len(citizens) >= 2:
        _generate_relationships(model, citizens, premise)

    return citizens


_RELATIONSHIP_SYSTEM = (
    "You generate relationships between villagers for a drama simulation.\n"
    "For EACH pair, write exactly one line:\n"
    "NAME1 -> NAME2: [relationship — e.g. 'owes $50', 'suspects of theft', "
    "'secret lovers', 'bitter rivals since childhood', 'blackmailing over a secret']\n\n"
    "RULES:\n"
    "- At least HALF the relationships must be NEGATIVE (rivalry, debt, suspicion, grudge).\n"
    "- Be specific and dramatic. No bland friendships.\n"
    "- Write ONLY the relationship lines. Nothing else."
)


def _generate_relationships(
    model, citizens: list[Citizen], premise: Premise | None = None,
) -> None:
    """Generate relationships between citizens and inject into their knowledge."""
    names_roles = "\n".join(f"- {c.name}: {c.role}" for c in citizens)
    setting = f"Setting: {premise.summary()}\n\n" if premise else ""
    prompt = f"{setting}Villagers:\n{names_roles}\n\nGenerate one relationship per pair."
    raw, _ = generate(model, _RELATIONSHIP_SYSTEM, prompt, 0.9, 300)

    for line in raw.strip().splitlines():
        if "->" not in line:
            continue
        left, _, right = line.partition("->")
        name_a = left.strip().split()[0] if left.strip() else ""
        if ":" in right:
            name_b_part, _, rel = right.partition(":")
            name_b = name_b_part.strip().split()[0] if name_b_part.strip() else ""
            rel = rel.strip()
        else:
            continue

        for c in citizens:
            if c.name == name_a:
                c.relationships[name_b] = rel
            elif c.name == name_b:
                c.relationships[name_a] = rel
