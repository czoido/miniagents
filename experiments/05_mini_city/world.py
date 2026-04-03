"""World data — dataclasses, citizens, schedule, and event seeds."""

from dataclasses import dataclass, field


@dataclass
class Premise:
    village: str
    region: str
    era: str
    mood: str
    store: str

    def summary(self) -> str:
        return (
            f"{self.village}, a {self.region} ({self.era}). "
            f"{self.mood} The general store is {self.store}."
        )


DEFAULT_PREMISE = Premise(
    village="Piedra Seca",
    region="mountain valley village",
    era="present-day remote",
    mood="A long drought threatens the harvest and tensions are rising.",
    store="Old Tomás's General Store",
)


@dataclass
class Citizen:
    name: str
    role: str
    personality: str
    goals: list[str]
    style: str
    temp: float
    money: int = 100
    income: int = 0
    secret: str = ""
    memory: list[str] = field(default_factory=list)
    relationships: dict[str, str] = field(default_factory=dict)


@dataclass
class Event:
    hour: int
    time_label: str
    location: str
    participants: list[str]
    transcript: list[tuple[str, str]]
    summary: str
    elapsed_s: float


@dataclass
class TimeSlot:
    time: str
    period: str
    location: str
    atmosphere: str


@dataclass
class Action:
    citizen: str
    description: str
    cost: int = 0
    earned: int = 0


# ---------------------------------------------------------------------------
# Citizens
# ---------------------------------------------------------------------------

CITIZENS = [
    Citizen(
        name="Rosa",
        role="Baker",
        personality=(
            "You are Rosa, 58, the village baker. Loud, opinionated, nosy. "
            "You know everyone's business and aren't afraid to share it. "
            "You complain about prices, your aching back, and lazy suppliers. "
            "You speak bluntly and drop unsolicited advice."
        ),
        goals=[
            "Pay the oven repairman $60 to fix the oven this week",
            "Confront Sombra about the stolen apples",
            "Find a cheaper flour supplier before month end",
        ],
        style="red",
        temp=0.85,
        money=150,
        income=40,
    ),
    Citizen(
        name="Viktor",
        role="Doctor",
        personality=(
            "You are Viktor, 45, the village doctor. Overworked, dry humor, "
            "blunt. You worry about the village's health but express it "
            "through sarcasm. You drink too much coffee and sleep too little. "
            "You give people medical advice whether they ask for it or not."
        ),
        goals=[
            "Buy medical supplies — bandages, antiseptic, thermometers",
            "Get a water testing kit to check the old well",
            "Post a job ad for a new nurse at the clinic",
        ],
        style="green",
        temp=0.5,
        money=250,
        income=50,
    ),
    Citizen(
        name="Luna",
        role="Artist",
        personality=(
            "You are Luna, 29, a painter who moved here a year ago. "
            "Friendly but distracted, always half-thinking about her next "
            "piece. You're broke, behind on rent, and trying to sell "
            "paintings at the market. You speak casually, sometimes changing "
            "topics abruptly."
        ),
        goals=[
            "Sell at least two paintings this week to pay rent ($80)",
            "Buy blue paint at the store to finish the school mural",
            "Convince someone to pose for a portrait commission",
        ],
        style="magenta",
        temp=0.9,
        money=40,
        income=0,
    ),
    Citizen(
        name="Marco",
        role="Mayor",
        personality=(
            "You are Marco, 52, the village mayor. Ambitious, talkative, "
            "a bit slippery. You promise a lot and deliver little. You're "
            "always campaigning, dropping hints about projects and asking "
            "for favors. You avoid direct confrontation but gossip behind "
            "people's backs."
        ),
        goals=[
            "Boost approval ratings before elections in three months",
            "Avoid committing to Viktor's clinic budget ($200)",
            "Hire someone to start the playground project",
        ],
        style="yellow",
        temp=0.7,
        money=400,
        income=0,
    ),
    Citizen(
        name="Sombra",
        role="Drifter",
        personality=(
            "You are Sombra, 35, a drifter who showed up two months ago. "
            "Guarded, street-smart, sarcastic. You do odd jobs for cash. "
            "You dodge personal questions with jokes or topic changes. "
            "You act tough but are actually looking for a place to belong."
        ),
        goals=[
            "Get a steady job — ask Marco about construction work",
            "Clear your name with Rosa about the stolen apples",
            "Buy a sleeping bag or blanket at the store ($15)",
        ],
        style="bright_black",
        temp=0.85,
        money=20,
        income=15,
    ),
    Citizen(
        name="Elena",
        role="Teacher",
        personality=(
            "You are Elena, 40, the village school teacher. Practical, caring "
            "but firm. You worry about the kids and the state of the school. "
            "You're direct and organized. You have a dry wit and little "
            "patience for the mayor's empty promises."
        ),
        goals=[
            "Pressure Marco to allocate $150 for the school roof",
            "Get Luna to finish the mural she promised",
            "Buy new textbooks for the kids ($50)",
        ],
        style="cyan",
        temp=0.7,
        money=120,
        income=30,
    ),
]

# ---------------------------------------------------------------------------
# Day schedule
# ---------------------------------------------------------------------------

DAY_SCHEDULE = [
    TimeSlot("7:00 AM",  "Dawn",           "the bakery",        "The smell of fresh bread drifts through empty streets."),
    TimeSlot("8:30 AM",  "Morning",        "the village square", "Shutters open one by one as the village stirs to life."),
    TimeSlot("10:00 AM", "Mid-morning",    "the market",        "The market hums with voices and the clink of coins."),
    TimeSlot("12:00 PM", "Noon",           "the fountain",      "The sun hangs directly overhead. Shade is scarce."),
    TimeSlot("2:00 PM",  "Afternoon",      "the park bench",    "A drowsy warmth settles. Even the birds go quiet."),
    TimeSlot("4:00 PM",  "Late afternoon", "the school steps",  "Children scatter from the schoolhouse like startled sparrows."),
    TimeSlot("6:00 PM",  "Evening",        "the bridge",        "Golden light stretches the shadows long across the cobblestones."),
    TimeSlot("8:00 PM",  "Dusk",           "the tavern",        "Lamps flicker on. The tavern door swings open and shut."),
    TimeSlot("10:00 PM", "Night",          "the alley",         "The village sleeps, but not everyone."),
    TimeSlot("11:30 PM", "Late night",     "the church steps",  "Only the moon and the clock tower keep watch now."),
]

# ---------------------------------------------------------------------------
# Seed pool for day-1 random events (later days generate dynamically)
# ---------------------------------------------------------------------------

EVENT_SEED_POOL = [
    "A pipe burst overnight at the school, flooding two classrooms.",
    "A traveling merchant arrived selling rare fabrics and spices at half price.",
    "The old well smells strange this morning — could be contaminated.",
    "A thunderstorm damaged the bridge last night. It needs urgent repairs.",
    "A journalist from the city showed up asking about village corruption.",
    "The mayor's office received a letter: the village qualifies for a $500 grant.",
    "A roof tile from the school fell and almost hit a child.",
    "A stray dog has been following Sombra around — he's been feeding it.",
]
