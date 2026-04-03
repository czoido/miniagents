"""System prompts and text constants for the Mini City simulation."""

INTERACTION_RULES = (
    "OUTPUT FORMAT: Write ONLY your character's spoken words. Nothing else.\n\n"
    "CORRECT:\n"
    'Hey Marco, the flour prices are killing me. Did you talk to the supplier?\n\n'
    "WRONG:\n"
    'You walk up to Marco and say "Hey Marco...". He responds...\n'
    "ALSO WRONG:\n"
    'You\'re at the bakery, and Viktor shows up. He asks if you\'ve seen...\n\n'
    "RULES:\n"
    "- Write ONLY 1-3 short sentences. MAX 40 words.\n"
    "- Talk about CONCRETE things: work, money, problems, gossip, complaints.\n"
    "- Try to advance your goals through the conversation.\n"
    "- You can ask others for help, negotiate, offer deals, lend/borrow money.\n"
    "- The village has a general store that sells anything for a price.\n"
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
    "or decisions taken. Be factual and specific. Write ONLY the summary."
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
    "- Use simple past tense. 8-12 sentences max."
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
    "Be direct and factual. No literary flourishes."
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
    "You create a setting for a village simulation.\n"
    "Write EXACTLY in this format, one field per line:\n\n"
    "VILLAGE: [a unique village name]\n"
    "REGION: [geography — e.g. coastal fishing town, mountain valley, desert oasis, "
    "river delta, forest clearing, volcanic island, arctic outpost]\n"
    "ERA: [time period flavor — e.g. 1920s rural, medieval, near-future, "
    "1800s frontier, present-day remote]\n"
    "MOOD: [one line setting the tone — e.g. 'drought threatens the harvest', "
    "'a gold rush draws strangers', 'the old mayor just died']\n"
    "STORE: [name of the general store, run by a local character]\n\n"
    "RULES:\n"
    "- Be specific and vivid. No generic fantasy.\n"
    "- The village must feel like a real, grounded place.\n"
    "- Write ONLY the fields above. Nothing else."
)

CITIZEN_SYSTEM = (
    "You create ONE villager for a village simulation.\n"
    "Write EXACTLY in this format, one field per line:\n\n"
    "NAME: [first name]\n"
    "ROLE: [job — e.g. Baker, Doctor, Farmer, Blacksmith, Librarian]\n"
    "AGE: [20-70]\n"
    "PERSONALITY: [2-3 sentences: who they are, how they talk, their flaws]\n"
    "GOAL1: [concrete goal with cost if applicable]\n"
    "GOAL2: [concrete goal with cost if applicable]\n"
    "GOAL3: [concrete goal with cost if applicable]\n"
    "MONEY: [starting money 20-400, based on role]\n"
    "INCOME: [daily income 0-60, based on role]\n\n"
    "RULES:\n"
    "- The character must be DIFFERENT from all existing characters listed.\n"
    "- Different name, different role, different personality.\n"
    "- Personality should include a flaw or quirk.\n"
    "- Goals must be concrete and actionable (not vague wishes).\n"
    "- Characters and goals should FIT the village setting described.\n"
    "- Write ONLY the fields above. Nothing else."
)

EVENT_SYSTEM = (
    "You generate ONE random event for a village simulation.\n"
    "The event should be a surprise that affects villagers and shakes up their routines.\n\n"
    "RULES:\n"
    "- Write ONE sentence, max 20 words.\n"
    "- Be specific and concrete — names, places, objects, dollar amounts.\n"
    "- The event must be NEW — different from previous events listed below.\n"
    "- It can be good (opportunity, visitor, gift) or bad (accident, theft, storm).\n"
    "- It should create a situation that villagers will want to discuss and react to.\n"
    "- NEVER repeat a previous event. NEVER be vague or abstract.\n"
    "- Write ONLY the event sentence. Nothing else."
)
