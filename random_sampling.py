import math
import time
import random
from collections import Counter
import json


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



# Load corpus from local file
corpus = "the quick brown fox jumps over the lazy dog"
corpus_file = "corpus_data.txt"

try:
    with open(corpus_file, 'r', encoding='utf-8') as f:
        corpus = f.read()
        print(f"Loaded corpus from local file: {len(corpus)} characters")
except FileNotFoundError:
    print("Corpus file not found, using fallback corpus")
else:
    print(f"Using cached corpus: {len(corpus)} characters")

# QWERTY rows
qwerty_rows = [
    "qwertyuiop",
    "asdfghjkl",
    "zxcvbnm"
]

# Define stagger offsets
row_stagger_offsets = {
    1: 0.25,   # Top row (QWERTY row)
    2: 0.75,   # Home row (ASDF row)
    3: 1.85    # Bottom row (ZXCV row)
}


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

def timed_cost(text, layout, name=""):
    """Timer utility for cost calculation"""
    start = time.time()
    value = layout_cost(text, layout)
    end = time.time()
    print(f"{name}: {value:.6f} (time: {end - start:.4f} sec)")
    return value

# Files for tracking progress and results
progress_file = "random_sampling_progress.json"
results_file = "random_sampling_results.txt"

def save_progress(last_seed, min_cost, best_layout_seed, best_layout):
    """Save progress to file - save minimum data and best layout in row format"""
    # Convert best layout to row format
    rows = convert_layout_to_rows(best_layout)
    
    with open(progress_file, 'w') as f:
        json.dump({
            'last_seed': last_seed,
            'min_cost': min_cost,
            'best_layout_seed': best_layout_seed,
            'best_layout_rows': {
                'first_row': rows[0],
                'home_row': rows[1],
                'last_row': rows[2]
            }
        }, f, indent=2)

def save_result(seed, cost, layout, is_new_min=False):
    """Save result to results file - flattened vector format"""
    # Convert layout to flattened 52-dimensional vector [x_a, y_a, x_b, y_b, ..., x_z, y_z]
    letters = 'abcdefghijklmnopqrstuvwxyz'
    vector = []
    
    for letter in letters:
        if letter in layout:
            x, y = layout[letter]
            vector.extend([x, y])
        else:
            # If letter not in layout, use default coordinates
            vector.extend([0.0, 0.0])
    
    with open(results_file, 'a', encoding='utf-8') as f:
        f.write(f"{vector}, Cost: {cost:.6f}\n")

def convert_layout_to_rows(layout):
    """Convert coordinate-based layout back to row format"""
    # Group letters by y-coordinate (row)
    rows = {1: [], 2: [], 3: []}
    
    for letter, (x, y) in layout.items():
        if y in rows:
            rows[y].append((x, letter))
    
    # Sort each row by x-coordinate and extract just the letters
    row_strings = []
    for row_num in [1, 2, 3]:
        sorted_letters = [letter for _, letter in sorted(rows[row_num])]
        row_strings.append(''.join(sorted_letters))
    
    return row_strings

def load_progress():
    """Load progress from file or return default values"""
    try:
        with open(progress_file, 'r') as f:
            content = f.read().strip()
            if not content:  # File is empty
                return 0, float('inf'), 0
            
            data = json.loads(content)
            
            # Handle both old and new format
            if 'best_layout' in data:
                # Old format - extract seed from the layout data
                best_layout_seed = data.get('last_seed', 0)  # Use last_seed as best for now
                print(f"Found old progress format, converting...")
            else:
                # New format
                best_layout_seed = data.get('best_layout_seed', 0)
            
            return data.get('last_seed', 0), data.get('min_cost', float('inf')), best_layout_seed
            
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"Progress file error: {e}, starting fresh")
        return 0, float('inf'), 0

def generate_random_layout_with_seed(seed):
    """Generate a random layout with a specific seed"""
    random.seed(seed)
    return generate_layout_from_rows(qwerty_rows, row_stagger_offsets, randomize=True, seed=seed)

# def layout_cost(text, layout, w1=1.0, w2=2.0, w3=0.5):
#     """Cost function for layout evaluation"""
#     text = text.lower()
#     letters = [ch for ch in text if ch in layout]
    
#     if not letters:
#         return float('inf')
    
#     letter_freq = Counter(letters)
#     bigram_freq = Counter(zip(letters, letters[1:]))
    
#     cost = 0.0
#     finger_assignments = assign_fingers(layout)
    
#     # Letter position cost (vertical distance from home row) + finger strength
#     home_row_y = 2
#     for letter, freq in letter_freq.items():
#         x, y = layout[letter]
#         dist_from_home = abs(y - home_row_y)
#         finger, hand = finger_assignments.get(letter, (0, 'R'))
        
#         cost += w1 * dist_from_home * freq
#         cost += w3 * finger_strength_penalty[finger] * freq
    
#     # Bigram travel + same-hand penalties
#     for (l1, l2), freq in bigram_freq.items():
#         if l1 in layout and l2 in layout:
#             x1, y1 = layout[l1]
#             x2, y2 = layout[l2]
            
#             # Manhattan distance for travel cost
#             travel = abs(x1 - x2) + abs(y1 - y2)
#             cost += (w1 * travel * freq)
            
#             # Same-hand penalty
#             if l1 in finger_assignments and l2 in finger_assignments:
#                 if finger_assignments[l1][1] == finger_assignments[l2][1]:
#                     cost += (w2 * freq) / (1 + travel)
    
#     return cost / len(letters) if letters else cost

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
    finger_strength_cost = finger_strength_cost/10
    #print(f'position_cost:{position_cost},finger_strenth_cost{finger_strength_cost},same_hand_cost{same_hand_cost}')

    total_cost = (w1 * position_cost + 
                  w3 * finger_strength_cost + 
                  w1 * travel_cost + 
                  w2 * same_hand_cost)
    
    return total_cost

def run_random_sampling(start_seed=None, num_layouts=100):
    """Run random layout sampling"""
    # Load previous progress
    last_seed, min_cost, best_layout_seed = load_progress()
    
    # Use provided start_seed or continue from last
    current_seed = start_seed if start_seed is not None else last_seed + 1
    
    print(f"Starting random sampling from seed {current_seed}")
    print(f"Previous minimum cost: {min_cost:.6f}")
    print(f"Previous best layout seed: {best_layout_seed}")
    print(f"Target layouts to generate: {num_layouts}")
    print("-" * 50)
    
    # Initialize results file if it doesn't exist
    try:
        with open(results_file, 'r') as f:
            pass  # File exists
    except FileNotFoundError:
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("Random Keyboard Layout Sampling Results\n")
            f.write("=" * 50 + "\n")
    
    best_layout = None
    best_seed = best_layout_seed
    
    for i in range(num_layouts):
        seed = current_seed + i
        
        # Generate random layout
        layout = generate_random_layout_with_seed(seed)
        
        # Calculate cost
        start_time = time.time()
        cost = layout_cost(corpus, layout)
        end_time = time.time()
        
        # Print progress
        print(f"Seed {seed:4d}: Cost = {cost:.6f} (Time: {end_time - start_time:.4f}s)")
        
        # Check if this is a new minimum
        is_new_min = cost < min_cost
        if is_new_min:
            min_cost = cost
            best_layout = layout
            best_seed = seed
            print(f"🎉 NEW MINIMUM! Cost: {cost:.6f}")
            
            # Save progress immediately
            save_progress(seed, min_cost, best_layout_seed, best_layout)
        
        # Save result to file
        save_result(seed, cost, layout, is_new_min)
    
    # Final summary
    print("\n" + "=" * 50)
    print("SAMPLING COMPLETED!")
    print(f"Best layout found: Seed {best_seed} (Cost: {min_cost:.6f})")
    print(f"Results saved to: {results_file}")
    print(f"Progress saved to: {progress_file}")
    
    return min_cost, best_layout

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate random keyboard layouts and find minimum cost')
    parser.add_argument('--start-seed', type=int, help='Starting seed number (default: continue from last)')
    parser.add_argument('--num-layouts', type=int, default=10, help='Number of layouts to generate (default: 10)')
    
    args = parser.parse_args()
    
    # Run sampling
    min_cost, best_layout = run_random_sampling(
        start_seed=args.start_seed,
        num_layouts=args.num_layouts
    )
