"""Ground truth metadata normalization and category mappings for Mozilla Common Voice."""

from typing import Optional
from app.schemas.response import AgeBracketEnum, GenderPredictionEnum


def normalize_gender(raw_gender: Optional[str]) -> Optional[str]:
    """Map Mozilla Common Voice gender metadata to VoxPulse gender categories.

    Rules:
    - 'male', 'm', 'man' -> 'male'
    - 'female', 'f', 'woman' -> 'female'
    - 'other', 'unknown', empty, None -> None (excluded from binary accuracy evaluation)

    Args:
        raw_gender: Raw gender label from dataset metadata.

    Returns:
        'male', 'female', or None if unsupported/excluded.
    """
    if not raw_gender:
        return None

    cleaned = str(raw_gender).strip().lower()

    if cleaned in ("male", "m", "man", "male_masculine"):
        return GenderPredictionEnum.MALE.value
    elif cleaned in ("female", "f", "woman", "female_feminine"):
        return GenderPredictionEnum.FEMALE.value

    # 'other', 'unknown', or unrecognized labels are excluded from binary evaluation
    return None


def normalize_age(raw_age: Optional[str]) -> Optional[str]:
    """Map Mozilla Common Voice age metadata to VoxPulse age bracket categories.

    Rules:
    - 'twenties', '20s' -> '18-30'
    - 'thirties', 'forties', 'fourties', '30s', '40s' -> '31-45'
    - 'fifties', '50s' -> '46-60'
    - 'sixties', 'seventies', 'eighties', 'nineties', '60s+' -> '60+'
    - 'teens' -> None (explicitly excluded: VoxPulse starts at age 18)
    - empty, unknown, other -> None (excluded from age accuracy evaluation)

    Args:
        raw_age: Raw age label from dataset metadata.

    Returns:
        '18-30', '31-45', '46-60', '60+', or None if excluded/unsupported.
    """
    if not raw_age:
        return None

    cleaned = str(raw_age).strip().lower()

    # Explicitly exclude 'teens' as VoxPulse age brackets begin at 18
    if cleaned in ("teens", "teen", "teenager", "<20"):
        return None

    if cleaned in ("twenties", "20s", "20-29"):
        return AgeBracketEnum.AGE_18_30.value
    elif cleaned in ("thirties", "30s", "30-39", "forties", "fourties", "40s", "40-49"):
        return AgeBracketEnum.AGE_31_45.value
    elif cleaned in ("fifties", "50s", "50-59"):
        return AgeBracketEnum.AGE_46_60.value
    elif cleaned in (
        "sixties",
        "60s",
        "60-69",
        "seventies",
        "70s",
        "70-79",
        "eighties",
        "80s",
        "80-89",
        "nineties",
        "90s",
        "90-99",
        "hundreds",
        "elderly",
    ):
        return AgeBracketEnum.AGE_60_PLUS.value

    return None


map_gender = normalize_gender
map_age = normalize_age
