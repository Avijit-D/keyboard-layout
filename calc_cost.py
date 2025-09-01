import math
import time
import random
from collections import Counter

# Parameters
row_penalty = 1 #w1
alternate_hand_penalty = 0.5 #w2
finger_penalty = 0.4 #w3
# Define stagger offsets
row_stagger_offsets = {
    1: 0.25,   # Top row (QWERTY row)
    2: 0.75,   # Home row (ASDF row)
    3: 1.85    # Bottom row (ZXCV row)
}
# Finger strength penalties (weaker finger = higher cost)
finger_strength_penalty = {0: 2.5, 1: 2.0, 2: 1.5, 3: 1.0}  # pinky=2.5, index=1.0

# Optional: load a large corpus via NLTK; fall back to a sample if unavailable
corpus = "the quick brown fox jumps over the lazy dog"
corpus_file = "corpus_data.txt"

# Try to load from local file first, then NLTK, then fallback
try:
    # First try to load from local file
    with open(corpus_file, 'r', encoding='utf-8') as f:
        corpus = f.read()
        print(f"Loaded corpus from local file: {len(corpus)} characters")
except FileNotFoundError:
    # If local file doesn't exist, try NLTK (commented out for offline use)
    # try:
    #     import nltk  # type: ignore
    #     try:
    #         nltk.download("reuters", quiet=True)
    #         from nltk.corpus import reuters  # type: ignore
    #         corpus = " ".join(reuters.words())
    #     except Exception:
    #         pass
    # except ImportError:
    #     pass
    print("Corpus file not found, using fallback corpus")
else:
    print(f"Using cached corpus: {len(corpus)} characters")


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

my_rows = [
    "zkutcdghwj",
    "frvameiyq",
    "bolsnpx"
]

# Build layouts via the generator
qwerty_layout = generate_layout_from_rows(qwerty_rows, row_stagger_offsets)
dvorak_layout = generate_layout_from_rows(dvorak_rows, row_stagger_offsets)
colemak_layout = generate_layout_from_rows(colemak_rows, row_stagger_offsets)
random_layout = generate_layout_from_rows(qwerty_rows, row_stagger_offsets, randomize=True, seed=42)
my_layout = generate_layout_from_rows(my_rows, row_stagger_offsets)
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
    """Cost function for layout evaluation with proper normalization"""
    text = text.lower()
    letters = [ch for ch in text if ch in layout]
   
    if not letters:
        return float('inf')
   
    letter_freq = Counter(letters)
    bigram_freq = Counter(zip(letters, letters[1:]))
   
    # Separate cost components for normalization
    position_cost = 0.0
    finger_strength_cost = 0.0
    travel_cost = 0.0
    same_hand_cost = 0.0
    
    finger_assignments = assign_fingers(layout)
    home_row_y = 2
   
    # Letter position cost and finger strength cost
    total_letters = sum(letter_freq.values())
    for letter, freq in letter_freq.items():
        x, y = layout[letter]
        dist_from_home = abs(y - home_row_y)
        finger, hand = finger_assignments.get(letter, (0, 'R'))
       
        # Normalize by frequency weight
        position_cost += dist_from_home * (freq / total_letters)
        finger_strength_cost += finger_strength_penalty[finger] * (freq / total_letters)
   
    # Bigram costs
    total_bigrams = sum(bigram_freq.values())
    if total_bigrams > 0:
        for (l1, l2), freq in bigram_freq.items():
            if l1 in layout and l2 in layout:
                x1, y1 = layout[l1]
                x2, y2 = layout[l2]
               
                # Euclidean distance for travel cost
                travel = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
                travel_cost += travel * (freq / total_bigrams)
               
                # Same-hand penalty
                if l1 in finger_assignments and l2 in finger_assignments:
                    if finger_assignments[l1][1] == finger_assignments[l2][1]:
                        same_hand_cost += (freq / total_bigrams) / (1 + travel)
   
    # Combine normalized components with weights
    finger_strength_cost=finger_strength_cost/10
    print(f'position_cost:{position_cost},finger_strenth_cost{finger_strength_cost},same_hand_cost{same_hand_cost}')

    total_cost = (w1 * position_cost + 
                  w3 * finger_strength_cost + 
                  w1 * travel_cost + 
                  w2 * same_hand_cost)
    
    return total_cost


if __name__ == "__main__":
    timed_cost(corpus, qwerty_layout, "QWERTY")
    timed_cost(corpus, dvorak_layout, "Dvorak")
    timed_cost(corpus, colemak_layout, "Colemak")
    timed_cost(corpus, random_layout, "Random Layout")
    timed_cost(corpus, my_layout, "My Layout")