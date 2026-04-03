"""System prompts and text constants for the Mini City simulation."""

INTERACTION_RULES = (
    "OUTPUT FORMAT: Write ONLY your character's spoken words. Nothing else.\n\n"
    "CORRECT:\n"
    'I saw you near the well last night. You better have a good explanation.\n\n'
    "WRONG:\n"
    'You walk up to Marco and say "Hey Marco...". He responds...\n'
    "ALSO WRONG:\n"
    'You\'re at the bakery, and Viktor shows up. He asks if you\'ve seen...\n\n'
    "RULES:\n"
    "- Write ONLY 1-3 short sentences. MAX 40 words.\n"
    "- Be direct, blunt, emotional. Characters have strong opinions.\n"
    "- Protect your SECRET — deflect, lie, or change the subject if it comes up.\n"
    "- Gossip, accuse, confront, demand, threaten, bargain, manipulate.\n"
    "- Try to advance your goals — even at others' expense.\n"
    "- If you suspect someone is hiding something, PRESS them.\n"
    "- Do NOT repeat what you already said. Bring up a DIFFERENT topic or goal.\n"
    "- NEVER narrate actions or describe scenes.\n"
    "- NEVER use 'You', 'He', 'She', or 'They' to describe what's happening.\n"
    "- NEVER write the other person's dialogue. You are ONE person.\n"
    "- NEVER use poetry, metaphors, or flowery language.\n"
    "- NEVER use quotation marks around your own words."
)

FAREWELL_HINT = (
    " If the conversation is winding down or you have nothing more to say,"
    " wrap up naturally (e.g. 'Alright, I gotta go', 'See you later')."
)

FAREWELL_WORDS = [
    "bye", "goodbye", "see you", "later", "gotta go", "have to go",
    "got to go", "heading out", "take care", "catch you later",
    "i should go", "i gotta", "i need to go", "anyway, i",
    "well, i better", "i'll let you", "off i go",
]

SUMMARY_SYSTEM = (
    "Summarize this conversation between two villagers in one sentence. "
    "Focus on what happened: agreements, conflicts, news shared, requests made, "
    "or decisions taken. Be factual and specific. Write ONLY the summary. "
    "Write in English only."
)

GOAL_UPDATE_SYSTEM = (
    "You manage ONE character's goal list. Update ONLY their personal goals.\n"
    "RULES:\n"
    "- REMOVE goals that were clearly accomplished, paid for, or resolved.\n"
    "- Keep goals that are still pending.\n"
    "- Add a NEW goal ONLY if this character personally committed to something.\n"
    "- NEVER add goals that belong to the OTHER person in the conversation.\n"
    "- Include costs if a goal requires money (e.g. 'Buy paint at the store ($15)').\n"
    "- Goals must match the character's ROLE — a doctor doesn't sell paintings,\n"
    "  a baker doesn't buy medical supplies, a mayor doesn't paint murals.\n"
    "- Keep the list to 2-4 items max.\n"
    "- Write ONLY the goals, one per line. No numbers, no bullets, no explanation."
)

NARRATOR_SYSTEM = (
    "You are a factual reporter summarizing what happened in a village today. "
    "You receive a log of conversations and actions. Write a plain, factual summary:\n"
    "- State WHO talked to WHOM, WHEN, WHERE, and WHAT was discussed or decided.\n"
    "- Note purchases, money spent, goals completed, and work done.\n"
    "- Note any agreements, conflicts, unresolved issues, or changes in plans.\n"
    "- Be direct and concise. No literary flourishes, no metaphors, no atmosphere.\n"
    "- Use simple past tense. 8-12 sentences max.\n"
    "- Write in English only."
)

OVERNIGHT_MEMORY_SYSTEM = (
    "Compress a character's memories from today into 3-5 key facts they would "
    "remember tomorrow. Drop trivial greetings. Keep important information: "
    "agreements, promises, conflicts, news, things they learned.\n"
    "Write ONLY the facts, one per line. No bullets, no explanation."
)

OVERNIGHT_GOALS_SYSTEM = (
    "A character is going to sleep. Write ONLY this character's personal goals "
    "for TOMORROW.\n"
    "RULES:\n"
    "- REMOVE goals that were accomplished or paid for today.\n"
    "- Keep unfinished goals that match this character's role.\n"
    "- Add 1-2 NEW goals based on what happened (promises, problems, opportunities).\n"
    "- NEVER include goals that belong to other villagers.\n"
    "- A baker's goals involve baking/selling/suppliers. A doctor's involve health/clinic.\n"
    "  An artist's involve paintings/commissions. A mayor's involve politics/projects.\n"
    "  A drifter's involve finding work/shelter. A teacher's involve school/kids.\n"
    "- If a goal needs money, include the estimated cost.\n"
    "- Keep the list to 2-4 items.\n"
    "- Write ONLY the goals, one per line."
)

FINAL_SUMMARY_SYSTEM = (
    "You are a factual reporter. You receive daily summaries of events in a village "
    "over several days. Write a final recap (8-15 sentences) covering:\n"
    "- What changed over the period.\n"
    "- Which conflicts were resolved and which remain.\n"
    "- How relationships between villagers evolved.\n"
    "- What the village looks like at the end.\n"
    "Be direct and factual. No literary flourishes.\n"
    "Write in English only."
)

ACTION_SYSTEM = (
    "A villager decides ONE optional purchase at the end of the day.\n"
    "They already earned their daily income — now they choose to BUY or do NOTHING.\n"
    "The village has a general store that sells ANYTHING.\n\n"
    "RULES:\n"
    "- If a goal needs an item or service, BUY it.\n"
    "- They CANNOT spend more money than they have.\n"
    "- Prices: basic items $5-20, specialized $20-60, major $60-150.\n"
    "- The action must match the character's ROLE and goals.\n"
    "- If nothing is needed or they can't afford it, say NOTHING.\n\n"
    "Write EXACTLY one line:\n"
    "BUY [item] for $[price]\n"
    "NOTHING\n\n"
    "WRONG: BUY paint for $20 or NOTHING\n"
    "CORRECT: BUY paint for $20\n\n"
    "Pick ONE. No 'or'. No explanation. One line only."
)

PREMISE_SYSTEM = (
    "You create a setting for a dramatic village simulation full of tension.\n"
    "Write EXACTLY in this format, one field per line:\n\n"
    "VILLAGE: [a unique village name]\n"
    "REGION: [geography — e.g. coastal fishing town, mountain valley, desert oasis, "
    "river delta, forest clearing, volcanic island, arctic outpost]\n"
    "ERA: [time period flavor — e.g. 1920s rural, medieval, near-future, "
    "1800s frontier, present-day remote]\n"
    "MOOD: [one line describing an URGENT crisis or conflict that divides the village — "
    "e.g. 'someone was found dead near the well', 'the mine collapse killed three men', "
    "'a stranger claims to own half the land']\n"
    "STORE: [name of the general store, run by a local character]\n\n"
    "RULES:\n"
    "- The MOOD must be a crisis that forces villagers to take sides.\n"
    "- Be specific and vivid. No generic fantasy.\n"
    "- The village must feel like a real, grounded place under pressure.\n"
    "- Write ONLY the fields above. Nothing else."
)

CITIZEN_SYSTEM = (
    "You create ONE villager for a village simulation full of drama and tension.\n"
    "Write EXACTLY in this format, one field per line:\n\n"
    "NAME: [first name]\n"
    "ROLE: [job — e.g. Baker, Doctor, Farmer, Blacksmith, Librarian]\n"
    "AGE: [20-70]\n"
    "PERSONALITY: [2-3 sentences: who they are, how they talk, their flaws]\n"
    "SECRET: [a hidden fact that would cause trouble if revealed — a past crime, "
    "a forbidden love, a hidden debt, a stolen identity, a betrayal]\n"
    "GOAL1: [concrete goal with cost if applicable]\n"
    "GOAL2: [concrete goal with cost if applicable]\n"
    "GOAL3: [concrete goal with cost if applicable]\n"
    "MONEY: [starting money 20-400, based on role]\n"
    "INCOME: [daily income 0-60, based on role]\n\n"
    "RULES:\n"
    "- The character must be DIFFERENT from all existing characters listed.\n"
    "- Different name, different role, different personality.\n"
    "- Personality should include a flaw or quirk that creates CONFLICT.\n"
    "- The SECRET must be specific and dangerous if revealed.\n"
    "- Goals must be concrete and actionable (not vague wishes).\n"
    "- At least one goal should conflict with another villager's interests.\n"
    "- Characters and goals should FIT the village setting described.\n"
    "- Write ONLY the fields above. Nothing else."
)

SCHEDULE_SYSTEM = (
    "You generate a list of locations for a village simulation.\n"
    "Each location must FIT the setting described.\n\n"
    "Write EXACTLY 10 lines, one per time slot, in this format:\n"
    "TIME | PERIOD | LOCATION | ATMOSPHERE\n\n"
    "Example for a desert island:\n"
    "7:00 AM | Dawn | the beach | Waves lap against the shore as the sun rises.\n"
    "8:30 AM | Morning | the camp | Smoke rises from last night's fire.\n\n"
    "Example for a medieval village:\n"
    "7:00 AM | Dawn | the blacksmith | The forge glows red in the early light.\n"
    "8:30 AM | Morning | the town square | Merchants set up their carts.\n\n"
    "RULES:\n"
    "- Locations must be SPECIFIC to the setting (no bakeries on islands, no beaches in mountains).\n"
    "- Each location must be DIFFERENT.\n"
    "- Atmosphere is one vivid sentence.\n"
    "- Write ONLY the 10 lines. Nothing else."
)

EVENT_SYSTEM = (
    "You generate ONE dramatic event for a village simulation.\n"
    "The event must be DISRUPTIVE — it forces villagers to react, take sides, or panic.\n\n"
    "GOOD EVENTS (high drama):\n"
    "- A fire destroys the blacksmith's workshop overnight.\n"
    "- Someone found blood on the general store floor this morning.\n"
    "- A stranger arrived claiming the mayor stole his inheritance.\n"
    "- The well water turned brown — someone may have poisoned it.\n"
    "- A villager's private letters were found posted on the church door.\n\n"
    "RULES:\n"
    "- Write ONE sentence, max 20 words.\n"
    "- The event MUST create conflict, suspicion, fear, or urgency.\n"
    "- Be specific — name places, objects, dollar amounts.\n"
    "- The event must be NEW — different from previous events listed below.\n"
    "- NEVER repeat a previous event. NEVER be vague or abstract.\n"
    "- Write ONLY the event sentence. Nothing else."
)

CHOOSE_TARGET_SYSTEM = (
    "A villager decides who they want to talk to RIGHT NOW.\n"
    "Based on their goals, secrets, relationships, and recent events, "
    "they pick the person they MOST need to see.\n\n"
    "Write ONLY one name from the list. Nothing else.\n"
    "If they want to be alone, write NOBODY."
)
