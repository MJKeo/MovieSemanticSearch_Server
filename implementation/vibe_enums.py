from enum import Enum


# -----------------------------
# Mood / atmosphere
# -----------------------------
class MoodAtmosphere(str, Enum):
    COZY = "cozy"
    WARM = "warm"
    WHOLESOME = "wholesome"
    GENTLE = "gentle"
    SERENE = "serene"
    DREAMY = "dreamy"
    WHIMSICAL = "whimsical"
    MAGICAL = "magical"
    GROUNDED = "grounded"
    GRITTY = "gritty"
    MOODY = "moody"
    OMINOUS = "ominous"
    BLEAK = "bleak"
    NIHILISTIC = "nihilistic"


# -----------------------------
# Tonal valence (warmth ↔ cynicism/bleakness)
# -----------------------------
class TonalValence(str, Enum):
    HOPEFUL = "hopeful"
    SINCERE = "sincere"
    BITTERSWEET = "bittersweet"
    DETACHED = "detached"
    CYNICAL = "cynical"
    JADED = "jaded"
    NIHILISTIC = "nihilistic"


# -----------------------------
# Pacing / momentum
# -----------------------------
class PacingMomentum(str, Enum):
    LANGUID = "languid"
    SLOW_BURN = "slow-burn"
    STEADY = "steady"
    BRISK = "brisk"
    PROPULSIVE = "propulsive"
    RELENTLESS = "relentless"


# -----------------------------
# Kinetic intensity (energy level)
# -----------------------------
class KineticIntensity(str, Enum):
    MELLOW = "mellow"
    LOW_KEY = "low-key"
    LIVELY = "lively"
    KINETIC = "kinetic"
    FRENETIC = "frenetic"


# -----------------------------
# Tension pressure
# -----------------------------
class TensionPressure(str, Enum):
    RELAXED = "relaxed"
    UNEASY = "uneasy"
    TENSE = "tense"
    GRIPPING = "gripping"
    NERVE_WRACKING = "nerve-wracking"
    SUFFOCATING = "suffocating"


# -----------------------------
# Unpredictability / twistiness
# -----------------------------
class UnpredictabilityTwistiness(str, Enum):
    STRAIGHTFORWARD = "straightforward"
    TELEGRAPHED = "telegraphed"
    TWISTY = "twisty"
    UNPREDICTABLE = "unpredictable"
    MIND_BENDING = "mind-bending"


# -----------------------------
# Scariness level
# -----------------------------
class ScarinessLevel(str, Enum):
    HARMLESS = "harmless"
    SPOOKY = "spooky"
    SCARY = "scary"
    TERRIFYING = "terrifying"
    NIGHTMARISH = "nightmarish"


# -----------------------------
# Fear mode
# -----------------------------
class FearMode(str, Enum):
    JUMPY = "jumpy"
    CREEPY = "creepy"
    EERIE = "eerie"
    DREAD_SOAKED = "dread-soaked"
    PANIC_INDUCING = "panic-inducing"


# -----------------------------
# Humor level
# -----------------------------
class HumorLevel(str, Enum):
    SERIOUS = "serious"
    WRY = "wry"
    FUNNY = "funny"
    HILARIOUS = "hilarious"
    RIOTOUS = "riotous"


# -----------------------------
# Humor flavor
# -----------------------------
class HumorFlavor(str, Enum):
    WHOLESOME = "wholesome"
    GOOFY = "goofy"
    DRY = "dry"
    SARCASTIC = "sarcastic"
    ABSURD = "absurd"
    DARKLY_FUNNY = "darkly-funny"
    CRINGE = "cringe"


# -----------------------------
# Violence intensity
# -----------------------------
class ViolenceIntensity(str, Enum):
    NONVIOLENT = "nonviolent"
    ROUGH = "rough"
    VIOLENT = "violent"
    BRUTAL = "brutal"
    SAVAGE = "savage"


# -----------------------------
# Gore / body grossness
# -----------------------------
class GoreBodyGrossness(str, Enum):
    CLEAN = "clean"
    BLOODY = "bloody"
    GORY = "gory"
    VISCERAL = "visceral"
    GROTESQUE = "grotesque"


# -----------------------------
# Romance prominence
# -----------------------------
class RomanceProminence(str, Enum):
    SUBTLE = "subtle"
    PRESENT = "present"
    PROMINENT = "prominent"
    CENTRAL = "central"


# -----------------------------
# Romance tone
# -----------------------------
class RomanceTone(str, Enum):
    SWEET = "sweet"
    FLIRTY = "flirty"
    TENDER = "tender"
    MESSY = "messy"
    VOLATILE = "volatile"
    BITTER = "bitter"


# -----------------------------
# Sexual explicitness
# -----------------------------
class SexualExplicitness(str, Enum):
    SUGGESTIVE = "suggestive"
    STEAMY = "steamy"
    EXPLICIT = "explicit"
    PORNOGRAPHIC = "pornographic"


# -----------------------------
# Erotic charge
# -----------------------------
class EroticCharge(str, Enum):
    NEUTRAL = "neutral"
    FLIRTY = "flirty"
    SENSUAL = "sensual"
    EROTIC = "erotic"
    LUSTY = "lusty"


# -----------------------------
# Sexual tone
# -----------------------------
class SexualTone(str, Enum):
    PLAYFUL = "playful"
    ROMANTIC = "romantic"
    CLINICAL = "clinical"
    AWKWARD = "awkward"
    UNCOMFORTABLE = "uncomfortable"
    TRANSGRESSIVE = "transgressive"


# -----------------------------
# Emotional heaviness
# -----------------------------
class EmotionalHeaviness(str, Enum):
    BREEZY = "breezy"
    MOVING = "moving"
    HEAVY = "heavy"
    DEVASTATING = "devastating"
    CRUSHING = "crushing"


# -----------------------------
# Emotional volatility
# -----------------------------
class EmotionalVolatility(str, Enum):
    EVEN_KEELED = "even-keeled"
    SIMMERING = "simmering"
    SWINGY = "swingy"
    TURBULENT = "turbulent"
    WHIPLASH = "whiplash"


# -----------------------------
# Weirdness / surrealism
# -----------------------------
class WeirdnessSurrealism(str, Enum):
    GROUNDED = "grounded"
    QUIRKY = "quirky"
    WEIRD = "weird"
    SURREAL = "surreal"
    HALLUCINATORY = "hallucinatory"
    FEVER_DREAM = "fever-dream"


# -----------------------------
# Attention demand
# -----------------------------
class AttentionDemand(str, Enum):
    BACKGROUNDABLE = "backgroundable"
    EASYWATCH = "easywatch"
    ATTENTIVE = "attentive"
    DEMANDING = "demanding"
    EXHAUSTING = "exhausting"


# -----------------------------
# Narrative complexity
# -----------------------------
class NarrativeComplexity(str, Enum):
    SIMPLE = "simple"
    STRAIGHTFORWARD = "straightforward"
    LAYERED = "layered"
    INTRICATE = "intricate"
    LABYRINTHINE = "labyrinthine"


# -----------------------------
# Ambiguity / interpretive-ness
# -----------------------------
class AmbiguityInterpretiveNess(str, Enum):
    LITERAL = "literal"
    CLEAR = "clear"
    SUGGESTIVE = "suggestive"
    AMBIGUOUS = "ambiguous"
    OPAQUE = "opaque"


# -----------------------------
# Sense of scale
# -----------------------------
class SenseOfScale(str, Enum):
    INTIMATE = "intimate"
    CONTAINED = "contained"
    EXPANSIVE = "expansive"
    GRAND = "grand"
    EPIC = "epic"
