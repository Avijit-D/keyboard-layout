import math
from collections import Counter

# Optional: load a large corpus via NLTK; fall back to a sample if unavailable
corpus = "the quick brown fox jumps over the lazy dog"
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

import time

def timed_cost(text, layout, name=""):
    start = time.time()
    cost = layout_cost(text, layout)
    end = time.time()
    print(f"{name}: {cost:.2f} (time: {end - start:.4f} sec)")
    return cost





# -----------------------
# 1. Define Keyboard Layout (QWERTY coordinates for now)
# -----------------------
# Coordinates are arbitrary grid positions
qwerty_layout = {
    'q': (0,0), 'w': (1,0), 'e': (2,0), 'r': (3,0), 't': (4,0), 'y': (5,0), 'u': (6,0), 'i': (7,0), 'o': (8,0), 'p': (9,0),
    'a': (0.5,1), 's': (1.5,1), 'd': (2.5,1), 'f': (3.5,1), 'g': (4.5,1), 'h': (5.5,1), 'j': (6.5,1), 'k': (7.5,1), 'l': (8.5,1),
    'z': (1,2), 'x': (2,2), 'c': (3,2), 'v': (4,2), 'b': (5,2), 'n': (6,2), 'm': (7,2)
}

dvorak_layout = {
    'p': (3,0), 'y': (4,0), 'f': (5,0), 'g': (6,0), 'c': (7,0), 'r': (8,0), 'l': (9,0),
    'a': (0.5,1), 'o': (1.5,1), 'e': (2.5,1), 'u': (3.5,1), 'i': (4.5,1), 'd': (5.5,1), 'h': (6.5,1), 't': (7.5,1), 'n': (8.5,1), 's': (9.5,1),
     'q': (2,2), 'j': (3,2), 'k': (4,2), 'x': (5,2), 'b': (6,2), 'm': (7,2), 'w': (8,2), 'v': (9,2), 'z': (10,2)
}

colemak_layout = {
    'q': (0,0), 'w': (1,0), 'f': (2,0), 'p': (3,0), 'g': (4,0), 'j': (5,0), 'l': (6,0), 'u': (7,0), 'y': (8,0),
    'a': (0.5,1), 'r': (1.5,1), 's': (2.5,1), 't': (3.5,1), 'd': (4.5,1), 'h': (5.5,1), 'n': (6.5,1), 'e': (7.5,1), 'i': (8.5,1), 'o': (9.5,1),
    'z': (1,2), 'x': (2,2), 'c': (3,2), 'v': (4,2), 'b': (5,2), 'k': (6,2), 'm': (7,2)
}

workman_layout = {
    'q': (0,0), 'd': (1,0), 'r': (2,0), 'w': (3,0), 'b': (4,0), 'j': (5,0), 'f': (6,0), 'u': (7,0), 'p': (8,0),
    'a': (0.5,1), 's': (1.5,1), 'h': (2.5,1), 't': (3.5,1), 'g': (4.5,1), 'y': (5.5,1), 'n': (6.5,1), 'e': (7.5,1), 'o': (8.5,1), 'i': (9.5,1),
    'z': (1,2), 'x': (2,2), 'm': (3,2), 'c': (4,2), 'v': (5,2), 'k': (6,2), 'l': (7,2)
}

import random

def generate_random_layout():
    # Define a simple 3-row keyboard grid like QWERTY
    rows = [
        "qwertyuiop",
        "asdfghjkl",
        "zxcvbnm"
    ]

    # Flatten positions (x, y) for all slots
    positions = []
    for y, row in enumerate(rows):
        for x, _ in enumerate(row):
            positions.append((x, y))

    # Take only A-Z letters
    letters = list("abcdefghijklmnopqrstuvwxyz")
    random.seed(42)
    random.shuffle(letters)

    # Map shuffled letters to positions
    layout = {}
    i = 0
    for y, row in enumerate(rows):
        for x, _ in enumerate(row):
            if i < len(letters):
                layout[letters[i]] = (x, y)
                i += 1

    return layout

# Example usage:
random_layout = generate_random_layout()
print("Random Layout:", random_layout)

# Compare cost




# Assign hands (left/right) and fingers (0 = pinky, 3 = index)
finger_assignments = {
    'q': (0,'L'), 'w': (1,'L'), 'e': (2,'L'), 'r': (3,'L'), 't': (3,'L'),
    'a': (0,'L'), 's': (1,'L'), 'd': (2,'L'), 'f': (3,'L'), 'g': (3,'L'),
    'z': (0,'L'), 'x': (1,'L'), 'c': (2,'L'), 'v': (3,'L'),
    'b': (3,'L'), 'y': (3,'R'), 'u': (3,'R'), 'i': (2,'R'), 'o': (1,'R'), 'p': (0,'R'),
    'h': (3,'R'), 'j': (3,'R'), 'k': (2,'R'), 'l': (1,'R'),
    'n': (3,'R'), 'm': (3,'R')
}

#print(len(qwerty_layout), len(dvorak_layout), len(colemak_layout), len(workman_layout))

# Finger strength penalties (weaker finger = higher cost)
finger_penalty = {0: 3.0, 1: 2.0, 2: 1.5, 3: 1.0}  # pinky=3.0, index=1.0


# -----------------------
# 2. Cost Function
# -----------------------
def layout_cost(text, layout=qwerty_layout, w1=1.0, w2=2.0, w3=0.5):
    text = text.lower()
    letters = [ch for ch in text if ch in layout]
    
    # Letter and bigram frequencies
    letter_freq = Counter(letters)
    bigram_freq = Counter(zip(letters, letters[1:]))
    
    cost = 0
    
    # -----------------
    # Letter "position" cost (distance from home row)
    # -----------------
    home_row_y = 1  # assume row=1 is the home row
    for letter, freq in letter_freq.items():
        x, y = layout[letter]
        dist = abs(y - home_row_y)   # vertical distance from home row
        finger, hand = finger_assignments[letter]
        
        cost += w1 * dist * freq            # reaching away from home row
        cost += w3 * finger_penalty[finger] * freq  # finger load penalty
    
    # -----------------
    # Bigram costs (movement + same-hand penalty)
    # -----------------
    for (l1, l2), freq in bigram_freq.items():
        if l1 in layout and l2 in layout:
            x1, y1 = layout[l1]
            x2, y2 = layout[l2]
            
            # Euclidean distance between consecutive keys
            dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            cost += w1 * dist * freq   # travel distance
            
            # Same hand penalty
            if finger_assignments[l1][1] == finger_assignments[l2][1]:
                cost += w2 * freq
    
    return cost



# -----------------------
# 3. Quick Test
# -----------------------
# Run all layouts with timing
timed_cost(corpus, qwerty_layout, "QWERTY")
timed_cost(corpus, dvorak_layout, "Dvorak")
timed_cost(corpus, colemak_layout, "Colemak")
timed_cost(corpus, workman_layout, "Workman")
timed_cost(corpus, random_layout, "Random Layout")