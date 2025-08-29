import math
import time
import random
from collections import Counter

# Parameters
row_penalty = 1.0 #w1
alternate_hand_penalty = 2.0 #w2
finger_penalty = 0.5 #w3
# Define stagger offsets
row_stagger_offsets = {
    1: 0.25,   # Top row (QWERTY row)
    2: 0.75,   # Home row (ASDF row)
    3: 1.85    # Bottom row (ZXCV row)
}
# Finger strength penalties (weaker finger = higher cost)
finger_strength_penalty = {0: 3.0, 1: 2.0, 2: 1.5, 3: 1.0}  # pinky=3.0, index=1.0
corpus = "the quick brown fox jumps over the lazy dog"


# Load corpus similar to cost.py (use NLTK Reuters if available, else fallback)

try:
    import nltk  # type: ignore
    try:
        nltk.download("reuters", quiet=True)
        from nltk.corpus import reuters  # type: ignore
        corpus = " ".join(reuters.words())
    except Exception:
        pass
except ImportError:
    pass


# Timer utility (reused from cost.py)
def timed_cost(text, layout, name=""):
    start = time.time()
    value = layout_cost(text, layout)
    end = time.time()
    print(f"{name}: {value:.6f} (time: {end - start:.4f} sec)")
    return value


def generate_layout_from_rows(rows, row_stagger_offsets, randomize=False, seed=None):
    """
    Generate a layout dictionary mapping letters -> (x, y) coordinates.

    Args:
        rows (list of str): Each string is a row of the keyboard.
        row_stagger_offsets (dict): Row index (starting at 1) -> stagger offset.
        randomize (bool): If True, randomly shuffle letters before assigning.
        seed (int or None): Random seed for reproducibility.

    Returns:
        dict: {letter: (x, y)}
    """
    # Collect all letters from rows
    letters = [ch for row in rows for ch in row]

    if randomize:
        if seed is not None:
            random.seed(seed)
        random.shuffle(letters)

    layout = {}
    for y, row in enumerate(rows, start=1):
        offset = row_stagger_offsets.get(y, 0.0)
        for x, ch in enumerate(row):
            layout[letters.pop(0)] = (x + offset, y)

    return layout


def assign_fingers(layout):
    """
    Assign fingers dynamically based on coordinates (QWERTY-like split).
    Works for any layout where x ~ [0..10].
    Args:
        layout (dict): {letter: (x,y)}
    Returns:
        dict: {letter: (finger, hand)}
    """
    finger_assignment = {}
    for letter, (x, y) in layout.items():
        if x < 5:  # Left hand
            hand = 'L'
            if x < 1.0:
                finger = 0  # pinky
            elif x < 2.0:
                finger = 1  # ring
            elif x < 3.0:
                finger = 2  # middle
            elif x < 5.0:
                finger = 3  # index
            else:
                finger = 3  # index continues

        else:  # Right hand
            hand = 'R'
            if x >= 9.5:
                finger = 0  # pinky
            elif x >= 8.5:
                finger = 1  # ring
            elif x >= 7.5:
                finger = 2  # middle
            else:
                finger = 3  # index

        finger_assignment[letter] = (finger, hand)

    return finger_assignment



# QWERTY rows
qwerty_rows = [
    "qwertyuiop",
    "asdfghjkl",
    "zxcvbnm"
]

# Dvorak rows
dvorak_rows = [
    "pyfgcrl",
    "aoeuidhtns",
    "qjkxbmwvz"
]

# Colemak rows
colemak_rows = [
    "qwfpgjluy",
    "arstdhneio",
    "zxcvbkm"
]


# Build layouts via the generator
qwerty_layout = generate_layout_from_rows(qwerty_rows, row_stagger_offsets)
dvorak_layout = generate_layout_from_rows(dvorak_rows, row_stagger_offsets)
colemak_layout = generate_layout_from_rows(colemak_rows, row_stagger_offsets)
random_layout = generate_layout_from_rows(qwerty_rows, row_stagger_offsets, randomize=True, seed=42)

# Debug: Check finger assignments for each layout
# print("=== QWERTY Finger Assignments ===")
# qwerty_fingers = assign_fingers(qwerty_layout)
# for letter, (finger, hand) in qwerty_fingers.items():
#     print(f"{letter}: finger={finger}, hand={hand}, pos={qwerty_layout[letter]}")

# print("\n=== Dvorak Finger Assignments ===")
# dvorak_fingers = assign_fingers(dvorak_layout)
# for letter, (finger, hand) in dvorak_fingers.items():
#     print(f"{letter}: finger={finger}, hand={hand}, pos={dvorak_layout[letter]}")

# print("\n=== Colemak Finger Assignments ===")
# colemak_fingers = assign_fingers(colemak_layout)
# for letter, (finger, hand) in colemak_fingers.items():
#     print(f"{letter}: finger={finger}, hand={hand}, pos={colemak_layout[letter]}")

# print("\n=== Random Layout Finger Assignments ===")
# random_fingers = assign_fingers(random_layout)
# for letter, (finger, hand) in random_fingers.items():
#     print(f"{letter}: finger={finger}, hand={hand}, pos={random_layout[letter]}")

# print("\n" + "="*50 + "\n")


# Cost function (improved version from cost.py)
def layout_cost(text, layout, w1=row_penalty, w2=alternate_hand_penalty, w3=finger_penalty):
    text = text.lower()
    letters = [ch for ch in text if ch in layout]

    letter_freq = Counter(letters)
    bigram_freq = Counter(zip(letters, letters[1:]))

    cost = 0.0
    finger_assignments = assign_fingers(layout)

    # Letter position cost (vertical distance from home row) + finger strength
    home_row_y = 2
    for letter, freq in letter_freq.items():
        x, y = layout[letter]
        dist_from_home = abs(y - home_row_y)
        finger, hand = finger_assignments.get(letter, (0, 'R'))

        cost += w1 * dist_from_home * freq
        cost += w3 * finger_strength_penalty[finger] * freq

    # Bigram travel + same-hand penalties
    for (l1, l2), freq in bigram_freq.items():
        if l1 in layout and l2 in layout:
            x1, y1 = layout[l1]
            x2, y2 = layout[l2]

            # Manhattan distance for travel cost
            travel = abs(x1 - x2) + abs(y1 - y2)
            cost += (w1 * travel * freq)

            # Same-hand penalty
            if l1 in finger_assignments and l2 in finger_assignments:
                if finger_assignments[l1][1] == finger_assignments[l2][1]:
                    cost += (w2 * freq) / (1 + travel)


    return cost / len(letters) if letters else cost



if __name__ == "__main__":
    timed_cost(corpus, qwerty_layout, "QWERTY")
    timed_cost(corpus, dvorak_layout, "Dvorak")
    timed_cost(corpus, colemak_layout, "Colemak")
    timed_cost(corpus, random_layout, "Random Layout")


