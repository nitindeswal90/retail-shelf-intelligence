from __future__ import annotations

from difflib import SequenceMatcher
import re


BRAND_ALIASES: list[tuple[str, tuple[str, ...]]] = [
    ("Minute Maid", ("minute maid", "minute", "maid")),
    ("Real", ("real", "rea", "rea)", "redq", "rlo")),
    ("Lay's", ("lays", "lay", "liays", "lIays", "({ays)", "(lays)", "lay's")),
    ("Doritos", ("doritos", "doritss", "doritzs")),
    ("Cheetos", ("cheetos", "geetos", "ceetos")),
    ("Kurkure", ("kurkure", "krkure")),
    ("Uncle Chips", ("uncle", "uncle chips", "udle", "undlle", "chipps")),
    ("Bingo", ("bingo", "bingol", "bingo}")),
    ("Pringles", ("pringles", "pricles")),
    ("Parle-G", ("parle", "parle g", "parle-g")),
    ("Good Day", ("good day", "good", "day")),
    ("Oreo", ("oreo",)),
    ("Dark Fantasy", ("dark fantasy", "fantasy", "choco fills")),
    ("Tiger Krunch", ("tiger", "krunch", "tger")),
    ("Hide & Seek", ("hide", "seek", "hide seek", "hide &seek")),
    ("Marie Gold", ("marie", "marie gold")),
    ("Malkist", ("malkist",)),
    ("Coca-Cola", ("coca-cola", "coca", "coke", "ccegdla", "cegdla", "gdla")),
    ("Sprite", ("sprite", "spritel")),
    ("Fanta", ("fanta", "fant")),
    ("Limca", ("limca", "limc", "limci")),
    ("Thums Up", ("thums", "thumbs", "up", "tupe")),
    ("Pepsi", ("pepsi",)),
    ("Mirinda", ("mirinda", "girinda", "airindi")),
    ("Mountain Dew", ("mountain dew", "dew")),
    ("7Up", ("7up", "up")),
    ("Nestea", ("nestea",)),
    ("Lipton", ("lipton", "lpton", "upton")),
    ("Gatorade", ("gatorade", "catorade", "oatorade")),
    ("Paper Boat", ("paper boat", "paper", "boat")),
    ("Red Bull", ("red bull", "redbul", "redbu")),
    ("Nescafe", ("nescafe",)),
    ("Amul", ("amul", "amule", "atnul")),
    ("Mother Dairy", ("mother dairy", "mother", "dairy")),
    ("Nestle", ("nestle", "nebtle", "nesite")),
    ("Activia", ("activia",)),
    ("Actimel", ("actimel",)),
    ("Yakult", ("yakult", "yakuld", "yakulo", "yakulb")),
    ("Epigamia", ("epigamia",)),
    ("Milky Mist", ("milky mist", "milkymist", "milkynlist")),
    ("Tropicana", ("tropicana",)),
    ("Hershey's", ("hershey", "hersheys", "hershey's")),
]


def map_brand(ocr_text: list[str]) -> str:
    # Map OCR text to known brand names using fuzzy matching and alias lists
    normalized_text = normalize_text(" ".join(ocr_text))
    tokens = normalized_text.split()
    compact_text = normalized_text.replace(" ", "")

    if not normalized_text:
        return "Other"

    for final_label, aliases in BRAND_ALIASES:
        if _matches_aliases(normalized_text, compact_text, tokens, aliases):
            return final_label

    return "Other"


def normalize_text(value: str) -> str:
    # Normalize text for comparison: lowercase, remove special chars, standardize spaces
    value = value.casefold()
    value = value.replace("'", "").replace("`", "").replace("’", "")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _matches_aliases(
    normalized_text: str,
    compact_text: str,
    tokens: list[str],
    aliases: tuple[str, ...],
) -> bool:
    # Check if any brand alias matches the normalized OCR text using various matching strategies
    for alias in aliases:
        normalized_alias = normalize_text(alias)
        if not normalized_alias:
            continue

        alias_compact = normalized_alias.replace(" ", "")
        if " " in normalized_alias:
            if _phrase_matches(normalized_text, compact_text, normalized_alias, alias_compact):
                return True
            continue

        if _token_matches(tokens, compact_text, alias_compact):
            return True

    return False


def _phrase_matches(
    normalized_text: str,
    compact_text: str,
    normalized_alias: str,
    alias_compact: str,
) -> bool:
    return (
        f" {normalized_alias} " in f" {normalized_text} "
        or alias_compact in compact_text
    )


def _token_matches(tokens: list[str], compact_text: str, alias: str) -> bool:
    if alias in tokens:
        return True

    # Useful for joined OCR such as "milkymist" or "redbul"; too risky for
    # very short aliases like "up", "rea", or "day".
    if len(alias) >= 5 and alias in compact_text:
        return True

    return any(_noisy_token_match(token, alias) for token in tokens)


def _noisy_token_match(token: str, alias: str) -> bool:
    if not token or not alias:
        return False

    if len(alias) >= 5 and alias in token:
        return True

    if len(alias) <= 3:
        return token == alias

    threshold = 0.88 if len(alias) == 4 else 0.78
    return SequenceMatcher(None, token, alias).ratio() >= threshold
