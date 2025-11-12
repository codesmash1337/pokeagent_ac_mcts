#!/usr/bin/env python3
"""
Compare two tokenized observation strings and print differences.

Usage:
    python compare_tokens.py

Or import and use the compare_observations function directly.
"""

import re
import sys
from pathlib import Path
from typing import Tuple, List, Optional

# Add metamon to path
REPO_ROOT = Path(__file__).resolve().parent
METAMON_ROOT = REPO_ROOT / "vendor" / "metamon"
if str(METAMON_ROOT) not in sys.path:
    sys.path.insert(0, str(METAMON_ROOT))

try:
    from metamon.tokenizer import PokemonTokenizer

    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    print("Warning: Could not import PokemonTokenizer, token names won't be shown")


def parse_old_format(text: str) -> Tuple[List[int], List[float]]:
    """
    Parse old format:
    Text tokens (length 106): [1454, 0, 24, ...]
    Numerical features (length 55): ["1.000", "1.000", ...]
    """
    text_tokens = []
    numerical_features = []

    # Parse text tokens
    text_match = re.search(r"Text tokens \(length \d+\): \[(.*?)\]", text)
    if text_match:
        tokens_str = text_match.group(1)
        text_tokens = [int(x.strip()) for x in tokens_str.split(",")]

    # Parse numerical features
    num_match = re.search(
        r"Numerical features \(length \d+\): \[(.*?)\]", text, re.DOTALL
    )
    if num_match:
        features_str = num_match.group(1)
        # Remove quotes and parse
        features_str = features_str.replace('"', "")
        numerical_features = [float(x.strip()) for x in features_str.split(",")]

    return text_tokens, numerical_features


def parse_new_format(text: str) -> Tuple[List[int], List[float]]:
    """
    Parse new format (with or without INFO prefix):

    Format 1 (with INFO):
    --- TOKENIZED OBSERVATION ---
    INFO     Text Tokens (106 tokens):
    INFO       [  0-  9]: [1454, 0, 24, ...]

    Format 2 (without INFO):
    Text Tokens (106 tokens):
      [  0-  9]: [1454, 0, 24, ...]

    Numerical Features (55 features):
      [ 0- 7]:  1.0000,  1.0000, ...
    """
    text_tokens = []
    numerical_features = []

    lines = text.split("\n")
    in_text_tokens = False
    in_numerical = False

    for line in lines:
        # Check if we're starting text tokens section
        if "Text Tokens" in line and "tokens):" in line:
            in_text_tokens = True
            in_numerical = False
            continue

        # Check if we're starting numerical features section
        if "Numerical Features" in line and "features):" in line:
            in_text_tokens = False
            in_numerical = True
            continue

        # Check if we're ending a section (empty line or section divider)
        stripped = line.strip()
        if stripped == "" or stripped.startswith("---") or stripped.startswith("==="):
            # Only end section if we see a clear section break
            # Don't end just because of any empty line
            if stripped.startswith("---") or stripped.startswith("==="):
                in_text_tokens = False
                in_numerical = False
            continue

        # Parse text token lines
        if in_text_tokens:
            # Match lines like: [  0-  9]: [1454, 0, 24, ...]
            # Works with or without INFO prefix
            match = re.search(r"\[.*?\]: \[(.*?)\]", line)
            if match:
                tokens_str = match.group(1)
                tokens = [int(x.strip()) for x in tokens_str.split(",") if x.strip()]
                text_tokens.extend(tokens)

        # Parse numerical feature lines
        if in_numerical:
            # Match lines like: [ 0- 7]:  1.0000,  1.0000, ...
            # Works with or without INFO prefix
            match = re.search(r"\[.*?\]:\s+(.*)", line)
            if match:
                nums_str = match.group(1)
                nums = [float(x.strip()) for x in nums_str.split(",") if x.strip()]
                numerical_features.extend(nums)

    return text_tokens, numerical_features


# Numerical feature descriptions for TeamPreviewObservationSpace
NUMERICAL_FEATURE_DESCRIPTIONS = [
    "Opponents remaining / 6.0",
    # Player Active Pokemon (15 features)
    "Player HP percentage",
    "Player Level / 100.0",
    "Player Base ATK / 255.0",
    "Player Base SpA / 255.0",
    "Player Base DEF / 255.0",
    "Player Base SpD / 255.0",
    "Player Base SPE / 255.0",
    "Player Base HP / 255.0",
    "Player ATK boost / 6.0",
    "Player SpA boost / 6.0",
    "Player DEF boost / 6.0",
    "Player SpD boost / 6.0",
    "Player SPE boost / 6.0",
    "Player Accuracy boost / 6.0",
    "Player Evasion boost / 6.0",
    # Player Move 1 (4 features)
    "Player Move 1 Base power / 200.0",
    "Player Move 1 Accuracy",
    "Player Move 1 Priority / 5.0",
    "Player Move 1 PP warning",
    # Player Move 2 (4 features)
    "Player Move 2 Base power / 200.0",
    "Player Move 2 Accuracy",
    "Player Move 2 Priority / 5.0",
    "Player Move 2 PP warning",
    # Player Move 3 (4 features)
    "Player Move 3 Base power / 200.0",
    "Player Move 3 Accuracy",
    "Player Move 3 Priority / 5.0",
    "Player Move 3 PP warning",
    # Player Move 4 (4 features)
    "Player Move 4 Base power / 200.0",
    "Player Move 4 Accuracy",
    "Player Move 4 Priority / 5.0",
    "Player Move 4 PP warning",
    # Player Switches (5 features)
    "Player Switch 1 HP percentage",
    "Player Switch 2 HP percentage",
    "Player Switch 3 HP percentage",
    "Player Switch 4 HP percentage",
    "Player Switch 5 HP percentage",
    # Opponent Active Pokemon (15 features)
    "Opponent HP percentage",
    "Opponent Level / 100.0",
    "Opponent Base ATK / 255.0",
    "Opponent Base SpA / 255.0",
    "Opponent Base DEF / 255.0",
    "Opponent Base SpD / 255.0",
    "Opponent Base SPE / 255.0",
    "Opponent Base HP / 255.0",
    "Opponent ATK boost / 6.0",
    "Opponent SpA boost / 6.0",
    "Opponent DEF boost / 6.0",
    "Opponent SpD boost / 6.0",
    "Opponent SPE boost / 6.0",
    "Opponent Accuracy boost / 6.0",
    "Opponent Evasion boost / 6.0",
    # Status Flags (3 features)
    "Any opponent asleep (history)",
    "Any opponent frozen (history)",
    "Can terastallize",
]


def get_token_description(idx: int) -> str:
    """Get description for token position."""
    if idx == 0:
        return "Battle format"
    elif idx == 1:
        return "Choice type"
    elif 2 <= idx <= 10:
        descs = [
            "Marker",
            "Species",
            "Item",
            "Ability",
            "Type 1",
            "Type 2",
            "Effect",
            "Status",
            "Tera Type",
        ]
        return f"Player Active: {descs[idx - 2]}"
    elif 11 <= idx <= 26:
        move_num = ((idx - 11) // 4) + 1
        pos = (idx - 11) % 4
        descs = ["Marker", "Move Name", "Move Type", "Category"]
        return f"Player Move {move_num}: {descs[pos]}"
    elif 27 <= idx <= 76:
        switch_num = ((idx - 27) // 10) + 1
        pos = (idx - 27) % 10
        descs = [
            "Marker",
            "Species",
            "Item",
            "Ability",
            "Moveset Marker",
            "Move 1",
            "Move 2",
            "Move 3",
            "Move 4",
            "Tera Type",
        ]
        return f"Player Switch {switch_num}: {descs[pos]}"
    elif 77 <= idx <= 85:
        descs = [
            "Marker",
            "Species",
            "Item",
            "Ability",
            "Type 1",
            "Type 2",
            "Effect",
            "Status",
            "Tera Type",
        ]
        return f"Opponent Active: {descs[idx - 77]}"
    elif 86 <= idx <= 89:
        descs = ["Marker", "Weather", "Player Conditions", "Opponent Conditions"]
        return f"Field Conditions: {descs[idx - 86]}"
    elif 90 <= idx <= 93:
        descs = [
            "Player Prev Marker",
            "Player Prev Move",
            "Opp Prev Marker",
            "Opp Prev Move",
        ]
        return f"Previous Moves: {descs[idx - 90]}"
    elif 94 <= idx <= 99:
        return f"Revealed Opponent {idx - 93}"
    elif 100 <= idx <= 105:
        return f"Team Preview {idx - 99}"
    else:
        return f"Unknown position {idx}"


def get_token_name(token_id: int, tokenizer: Optional[PokemonTokenizer] = None) -> str:
    """Get token name from ID using tokenizer."""
    if tokenizer is None or not TOKENIZER_AVAILABLE:
        return f"ID:{token_id}"

    try:
        # Use the tokenizer's decode method if available
        if hasattr(tokenizer, "id_to_word"):
            word = tokenizer.id_to_word.get(token_id)
            if word:
                return word

        # Fallback: Reverse lookup in tokenizer vocabulary
        if hasattr(tokenizer, "word_to_id"):
            for word, id_val in tokenizer.word_to_id.items():
                if id_val == token_id:
                    return word

        # Special handling for special tokens
        if token_id == -1:
            return "<unk_-1>"
        elif token_id == 0:
            return "<anychoice>"
        elif token_id == 1:
            return "<blank>"

        return f"<unk_{token_id}>"
    except Exception:
        return f"ID:{token_id}"


def compare_observations(
    obs1_text: str,
    obs2_text: str,
    format1: str = "auto",
    format2: str = "auto",
    tolerance: float = 1e-6,
):
    """
    Compare two observation strings and print differences.

    Args:
        obs1_text: First observation string
        obs2_text: Second observation string
        format1: Format of first observation ('old', 'new', or 'auto')
        format2: Format of second observation ('old', 'new', or 'auto')
        tolerance: Tolerance for numerical comparison
    """
    # Auto-detect formats
    if format1 == "auto":
        # New format has "Text Tokens" followed by bracketed ranges
        # Old format has "Text tokens (length 106): [1454, 0, ...]" all on one line
        if "Text Tokens" in obs1_text and "[  0-" in obs1_text:
            format1 = "new"
        elif "Text tokens (length" in obs1_text:
            format1 = "old"
        else:
            format1 = "new"  # Default to new

    if format2 == "auto":
        if "Text Tokens" in obs2_text and "[  0-" in obs2_text:
            format2 = "new"
        elif "Text tokens (length" in obs2_text:
            format2 = "old"
        else:
            format2 = "new"  # Default to new

    # Load tokenizer if available
    tokenizer = None
    if TOKENIZER_AVAILABLE:
        try:
            tokenizer = PokemonTokenizer()
            print("✓ Loaded PokemonTokenizer for token name lookup")
        except Exception:
            print("⚠ Could not load tokenizer")

    # Parse observations
    print(f"\nParsing Observation 1 (format: {format1})...")
    if format1 == "old":
        tokens1, nums1 = parse_old_format(obs1_text)
    else:
        tokens1, nums1 = parse_new_format(obs1_text)

    print(f"Parsing Observation 2 (format: {format2})...")
    if format2 == "old":
        tokens2, nums2 = parse_old_format(obs2_text)
    else:
        tokens2, nums2 = parse_new_format(obs2_text)

    print(
        f"\nObservation 1: {len(tokens1)} text tokens, {len(nums1)} numerical features"
    )
    print(f"Observation 2: {len(tokens2)} text tokens, {len(nums2)} numerical features")

    # Compare text tokens
    print("\n" + "=" * 80)
    print("TEXT TOKENS COMPARISON")
    print("=" * 80)

    if len(tokens1) != len(tokens2):
        print(f"⚠️  LENGTH MISMATCH: {len(tokens1)} vs {len(tokens2)}")

    max_len = max(len(tokens1), len(tokens2))
    token_diffs = []

    for i in range(max_len):
        val1 = tokens1[i] if i < len(tokens1) else None
        val2 = tokens2[i] if i < len(tokens2) else None

        if val1 != val2:
            token_diffs.append((i, val1, val2))

    if token_diffs:
        print(f"\n❌ Found {len(token_diffs)} differences in text tokens:\n")
        for idx, val1, val2 in token_diffs:
            desc = get_token_description(idx)
            name1 = get_token_name(val1, tokenizer) if val1 is not None else "<missing>"
            name2 = get_token_name(val2, tokenizer) if val2 is not None else "<missing>"

            print(f"  [{idx:3d}] {desc}")
            if val1 is None:
                print("        Obs1: <missing>")
                print(f"        Obs2: {name2:20s} (ID: {val2:5d})")
            elif val2 is None:
                print(f"        Obs1: {name1:20s} (ID: {val1:5d})")
                print("        Obs2: <missing>")
            else:
                print(f"        Obs1: {name1:20s} (ID: {val1:5d})")
                print(f"        Obs2: {name2:20s} (ID: {val2:5d})")
            print()
    else:
        print("\n✅ Text tokens are identical!")

    # Compare numerical features
    print("\n" + "=" * 80)
    print("NUMERICAL FEATURES COMPARISON")
    print("=" * 80)

    if len(nums1) != len(nums2):
        print(f"⚠️  LENGTH MISMATCH: {len(nums1)} vs {len(nums2)}")

    max_len = max(len(nums1), len(nums2))
    num_diffs = []

    for i in range(max_len):
        val1 = nums1[i] if i < len(nums1) else None
        val2 = nums2[i] if i < len(nums2) else None

        if val1 is None or val2 is None:
            num_diffs.append((i, val1, val2, None))
        elif abs(val1 - val2) > tolerance:
            diff = val2 - val1
            num_diffs.append((i, val1, val2, diff))

    if num_diffs:
        print(f"\n❌ Found {len(num_diffs)} differences in numerical features:\n")
        for idx, val1, val2, diff in num_diffs:
            desc = (
                NUMERICAL_FEATURE_DESCRIPTIONS[idx]
                if idx < len(NUMERICAL_FEATURE_DESCRIPTIONS)
                else f"Feature {idx}"
            )

            print(f"  [{idx:2d}] {desc}")
            if val1 is None:
                print("        Obs1: <missing>")
                print(f"        Obs2: {val2:10.6f}")
            elif val2 is None:
                print(f"        Obs1: {val1:10.6f}")
                print("        Obs2: <missing>")
            else:
                print(f"        Obs1: {val1:10.6f}")
                print(f"        Obs2: {val2:10.6f}")
                print(f"        Diff: {diff:+10.6f}")
            print()
    else:
        print(f"\n✅ Numerical features are identical (tolerance={tolerance})!")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"Text tokens differences: {len(token_diffs)}/{max(len(tokens1), len(tokens2))}"
    )
    print(f"Numerical differences: {len(num_diffs)}/{max(len(nums1), len(nums2))}")

    if not token_diffs and not num_diffs:
        print("\n✅ Observations are IDENTICAL!")
    else:
        print("\n❌ Observations are DIFFERENT")


def main():
    """Main function with command-line support for comparing observation files."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare two tokenized observation strings from files or use example data."
    )
    parser.add_argument(
        "file1",
        nargs="?",
        help="Path to first observation file (if not provided, uses example data)",
    )
    parser.add_argument(
        "file2",
        nargs="?",
        help="Path to second observation file (if not provided, uses example data)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Tolerance for numerical comparison (default: 1e-6)",
    )

    args = parser.parse_args()

    # If files provided, read them
    if args.file1 and args.file2:
        print(f"Reading file 1: {args.file1}")
        with open(args.file1, "r") as f:
            obs1 = f.read()

        print(f"Reading file 2: {args.file2}")
        with open(args.file2, "r") as f:
            obs2 = f.read()

        print("=" * 80)
        print(f"COMPARING {Path(args.file1).name} vs {Path(args.file2).name}")
        print("=" * 80)

        compare_observations(obs1, obs2, tolerance=args.tolerance)
        return

    # Otherwise use example data
    print("No files provided, using example data.")
    print("Usage: python compare_tokens.py <file1> <file2>")
    print("=" * 80)

    # Example observation 1 (old format)
    obs1 = """
Text Tokens (106 tokens):
  [  0-  9]: [1454, 0, 24, 1941, 660, 2373, 274, 1232, 1740, 857]
  [ 10- 19]: [859, 20, 600, 1232, 906, 20, 686, 398, 906, 20]
  [ 20- 29]: [1252, 274, 906, 20, 1279, 853, 1229, 26, 1863, 268]
  [ 30- 39]: [1997, 21, 278, 531, 2075, 2169, 859, 26, 1952, 1979]
  [ 40- 49]: [962, 21, 321, 582, 2268, 2409, 859, 26, 2527, 660]
  [ 50- 59]: [1646, 21, 1541, 263, 598, 1251, 859, 26, 1, 1]
  [ 60- 69]: [1, 1, 1, 1, 1, 1, 1, 26, 1, 1]
  [ 70- 79]: [1, 1, 1, 1, 1, 1, 1, 23, 1956, 1189]
  [ 80- 89]: [596, 428, 501, 848, 857, 859, 2, 860, 1231, 1231]
  [ 90- 99]: [25, 852, 22, 490, 1956, 1, 1, 1, 1, 1]
  [100-105]: [1863, 1925, 1941, 1952, 1956, 2527]

Numerical Features (55 features):
  [ 0- 7]:  0.8333,  0.9386,  1.0000,  0.5294,  0.2353,  0.4706,  0.3333,  0.1961
  [ 8-15]:  0.3922,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000
  [16-23]:  0.4000,  1.0000,  0.0000,  3.0000,  0.0000,  1.0000,  0.0000,  3.0000
  [24-31]:  0.3500,  1.0000,  0.2000,  3.0000,  0.0000,  1.0000,  0.0000,  3.0000
  [32-39]:  1.0000,  1.0000,  1.0000, -2.0000, -2.0000,  0.7196,  1.0000,  0.5686
  [40-47]:  0.4118,  0.3529,  0.3137,  0.3569,  0.3490, -0.1667,  0.0000,  0.0000
  [48-54]:  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000
  """
    # Example observation 2 (new format)
    obs2 = """
--- TOKENIZED OBSERVATION ---
INFO     Text Tokens (106 tokens):
INFO       [  0-  9]: [1454, 0, 24, 1941, 660, 2373, 274, 1232, 1740, 857]
INFO       [ 10- 19]: [859, 20, 600, 1232, 906, 20, 686, 398, 906, 20]
INFO       [ 20- 29]: [1252, 274, 906, 20, 1279, 853, 1229, 26, 1863, 268]
INFO       [ 30- 39]: [1997, 21, 278, 531, 2075, 2169, 859, 26, 1952, 1979]
INFO       [ 40- 49]: [962, 21, 321, 582, 2268, 2409, 859, 26, 2527, 660]
INFO       [ 50- 59]: [1646, 21, 1541, 263, 598, 1251, 859, 26, 1, 1]
INFO       [ 60- 69]: [1, 1, 1, 1, 1, 1, 1, 26, 1, 1]
INFO       [ 70- 79]: [1, 1, 1, 1, 1, 1, 1, 23, 1956, 1189]
INFO       [ 80- 89]: [596, 428, 501, 848, 857, 859, 2, 860, 1231, 1231]
INFO       [ 90- 99]: [25, 852, 22, 490, 1956, 1, 1, 1, 1, 1]
INFO       [100-105]: [1863, 1925, 1941, 1952, 1956, 2527]
INFO     
Numerical Features (55 features):
INFO       [ 0- 7]:  0.8333,  0.9386,  1.0000,  0.5294,  0.2353,  0.4706,  0.3333,  0.1961
INFO       [ 8-15]:  0.3922,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000
INFO       [16-23]:  0.4000,  1.0000,  0.0000,  3.0000,  0.0000,  1.0000,  0.0000,  3.0000
INFO       [24-31]:  0.3500,  1.0000,  0.2000,  3.0000,  0.0000,  1.0000,  0.0000,  3.0000
INFO       [32-39]:  1.0000,  1.0000,  1.0000, -2.0000, -2.0000,  0.7196,  1.0000,  0.5686
INFO       [40-47]:  0.4118,  0.3529,  0.3137,  0.3569,  0.3490, -0.1667,  0.0000,  0.0000
INFO       [48-54]:  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000
"""

    print("COMPARING SAMPLE OBSERVATIONS")
    print("=" * 80)

    compare_observations(obs1, obs2)


if __name__ == "__main__":
    main()
