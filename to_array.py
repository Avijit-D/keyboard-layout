import math
import time
import random
from collections import Counter
import json
import numpy as np
import warnings
import os

# Suppress TensorFlow warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Import TensorFlow after setting environment variables
from tensorflow import keras
import joblib

# Parameters for cost calculation
row_penalty = 1  # w1
alternate_hand_penalty = 0.5  # w2
finger_penalty = 0.4  # w3

# Finger strength penalties (weaker finger = higher cost)
finger_strength_penalty = {0: 2.5, 1: 2.0, 2: 1.5, 3: 1.0}  # pinky=2.5, index=1.0

row_stagger_offsets = {
    1: 0.25,   # Top row (QWERTY row)
    2: 0.75,   # Home row (ASDF row)
    3: 1.85    # Bottom row (ZXCV row)
}

# Load corpus from local file
corpus = "the quick brown fox jumps over the lazy dog"
corpus_file = "corpus_data.txt"

try:
    with open(corpus_file, 'r', encoding='utf-8') as f:
        corpus = f.read()
        print(f"Loaded corpus from local file: {len(corpus)} characters")
except FileNotFoundError:
    print("Corpus file not found, using fallback corpus")
    print(f"Using fallback corpus: {len(corpus)} characters")

# Load the scaler used during training
try:
    scaler_X = joblib.load("scaler_X.pkl")
    print("✓ Loaded scaler_X.pkl successfully")
except FileNotFoundError:
    print("⚠ Warning: scaler_X.pkl not found. Predictions may be inaccurate.")
    scaler_X = None


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

def layout_to_vector(layout):
    """
    Convert a keyboard layout to a flattened 52-dimensional vector.
    
    Args:
        layout (dict): {letter: (x, y)} coordinate mapping
        
    Returns:
        list: 52-dimensional vector [x_a, y_a, x_b, y_b, ..., x_z, y_z]
    """
    letters = 'abcdefghijklmnopqrstuvwxyz'
    vector = []
    
    for letter in letters:
        if letter in layout:
            x, y = layout[letter]
            vector.extend([x, y])
        else:
            # If letter not in layout, use default coordinates
            vector.extend([0.0, 0.0])
    
    return vector

def rows_to_vector(rows):
    """
    Convert keyboard rows to a flattened 52-dimensional vector.
    
    Args:
        rows (list of str): List of 3 strings representing keyboard rows
        
    Returns:
        list: 52-dimensional vector [x_a, y_a, x_b, y_b, ..., x_z, y_z]
    """
    layout = generate_layout_from_rows(rows, row_stagger_offsets, randomize=False)
    return layout_to_vector(layout)

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

    total_cost = (w1 * position_cost + 
                  w3 * finger_strength_cost + 
                  w1 * travel_cost + 
                  w2 * same_hand_cost)
    
    return total_cost



def print_layout_visualization(layout):
    """
    Print a visual representation of the keyboard layout.
    
    Args:
        layout (dict): {letter: (x, y)} coordinate mapping
    """
    # Create a grid representation
    grid = {}
    for letter, (x, y) in layout.items():
        grid[(x, y)] = letter
    
    # Find bounds
    if not grid:
        print("Empty layout")
        return
    
    min_x = min(x for x, y in grid.keys())
    max_x = max(x for x, y in grid.keys())
    min_y = min(y for x, y in grid.keys())
    max_y = max(y for x, y in grid.keys())
    
    # Print the layout
    for y in range(int(min_y), int(max_y) + 1):
        row_str = ""
        for x in range(int(min_x), int(max_x) + 1):
            # Check for exact float matches
            found_letter = None
            for (gx, gy), letter in grid.items():
                if abs(gx - x) < 0.1 and abs(gy - y) < 0.1:
                    found_letter = letter
                    break
            if found_letter:
                row_str += found_letter.upper()
            else:
                row_str += " "
        print(row_str)

def predict_layout_cost(vector, model_path="cost_prediction_model.keras"):
    """
    Predict the cost of a keyboard layout using the trained neural network model.
    
    Args:
        vector (list): 52-dimensional vector [x_a, y_a, x_b, y_b, ..., x_z, y_z]
        model_path (str): Path to the trained model file
        
    Returns:
        float: Predicted cost value
    """
    try:
        # Load the trained model
        model = keras.models.load_model(model_path, compile=False)
        
        # Convert vector to numpy array and reshape for prediction
        # The model expects input shape (batch_size, 52)
        vector_array = np.array(vector).reshape(1, -1)
        
        # Scale the input vector using the same scaler used during training
        if scaler_X is not None:
            vector_scaled = scaler_X.transform(vector_array)
        else:
            print("⚠ Warning: No scaler available, using unscaled input")
            vector_scaled = vector_array
        
        # Make prediction
        predicted_cost_scaled = model.predict(vector_scaled, verbose=0)
        
        # Scale back the prediction (divide by 1000 as used in training)
        predicted_cost = float(predicted_cost_scaled[0][0]) / 1000
        
        return predicted_cost
        
    except FileNotFoundError:
        print(f"❌ Model file {model_path} not found")
        return None
    except Exception as e:
        print(f"❌ Error predicting cost: {e}")
        return None

def predict_cost_from_rows(rows, model_path="cost_prediction_model.keras"):
    """
    Predict the cost of a keyboard layout given as rows.
    
    Args:
        rows (list): List of 3 strings representing keyboard rows
        model_path (str): Path to the trained model file
        
    Returns:
        tuple: (predicted_cost, actual_cost) or (None, None) if error
    """
    # Convert rows to vector
    vector = rows_to_vector(rows)
    
    # Predict cost using the vector
    predicted_cost = predict_layout_cost(vector, model_path)
    
    # Calculate actual cost
    layout = generate_layout_from_rows(rows, row_stagger_offsets, randomize=False)
    actual_cost = layout_cost(corpus, layout)
    
    return predicted_cost, actual_cost



if __name__ == "__main__":
    print("🎹 Keyboard Layout Cost Predictor")
    print("=" * 40)
    
    # 1. Convert QWERTY rows to vector
    qwerty_rows = [
    "qwertyuiop",
        "asdfghjkl",
        "zxcvbnm"
    ]
    
    # print("\n📊 QWERTY Layout Analysis")
    # print("-" * 25)
    # qwerty_predicted, qwerty_actual = predict_cost_from_rows(qwerty_rows)
    # if qwerty_predicted is not None and qwerty_actual is not None:
    #     accuracy = 100 * (1 - abs(qwerty_predicted - qwerty_actual) / qwerty_actual)
    #     print(f"🤖 Predicted: {qwerty_predicted:.4f}")
    #     print(f"📈 Actual:    {qwerty_actual:.4f}")
    #     print(f"📊 Difference: {abs(qwerty_predicted - qwerty_actual):.4f}")
    #     print(f"🎯 Accuracy:   {accuracy:.1f}%")
    # else:
    #     print("❌ Failed to predict/calculate cost")
    
    # 2. Example with a custom layout
    custom_rows = [
        "juvwthrpkc",
        "bmoasedly",
        "xgfqniz"
    ]
    
    print("\n📊 Custom Layout Analysis")
    print("-" * 25)
    custom_predicted, custom_actual = predict_cost_from_rows(custom_rows)
    if custom_predicted is not None and custom_actual is not None:
        accuracy = 100 * (1 - abs(custom_predicted - custom_actual) / custom_actual)
        print(f"🤖 Predicted: {custom_predicted:.4f}")
        print(f"📈 Actual:    {custom_actual:.4f}")
        print(f"📊 Difference: {abs(custom_predicted - custom_actual):.4f}")
        print(f"🎯 Accuracy:   {accuracy:.1f}%")
    else:
        print("❌ Failed to predict/calculate cost")
    
    print("\n" + "=" * 40)
    print("✅ Analysis Complete!")