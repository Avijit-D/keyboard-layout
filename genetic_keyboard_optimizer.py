import random
import numpy as np
from collections import Counter
import warnings
import os
import time
import json
import matplotlib.pyplot as plt

# Suppress TensorFlow warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Import TensorFlow after setting environment variables
from tensorflow import keras
import joblib

# Parameters for cost calculation (copied from random_sampling.py)
row_penalty = 1  # w1
alternate_hand_penalty = 0.5  # w2
finger_penalty = 0.4  # w3
finger_strength_penalty = {0: 2.5, 1: 2.0, 2: 1.5, 3: 1.0}  # pinky=2.5, index=1.0

row_stagger_offsets = {
    1: 0.25,   # Top row
    2: 0.75,   # Home row
    3: 1.85    # Bottom row
}

# Load corpus and scaler
corpus = "the quick brown fox jumps over the lazy dog"
corpus_file = "corpus_data.txt"

try:
    with open(corpus_file, 'r', encoding='utf-8') as f:
        corpus = f.read()
        print(f"✓ Loaded corpus: {len(corpus)} characters")
except FileNotFoundError:
    print("⚠ Using fallback corpus")
    print(f"Using fallback corpus: {len(corpus)} characters")

try:
    scaler_X = joblib.load("scaler_X.pkl")
    print("✓ Loaded scaler_X.pkl successfully")
except FileNotFoundError:
    print("❌ scaler_X.pkl not found!")
    scaler_X = None

# Load the trained model
try:
    model = keras.models.load_model("cost_prediction_model.keras", compile=False)
    print("✓ Loaded cost_prediction_model.keras successfully")
except FileNotFoundError:
    print("❌ cost_prediction_model.keras not found!")
    model = None


class KeyboardLayout:
    """Represents a keyboard layout as a permutation of letters"""
    
    def __init__(self, letters=None):
        if letters is None:
            # Default QWERTY layout
            self.letters = list("qwertyuiopasdfghjklzxcvbnm")
        else:
            self.letters = list(letters)
    
    def __str__(self):
        # Display as 3 rows
        rows = [
            self.letters[:10],  # Top row
            self.letters[10:19],  # Middle row
            self.letters[19:]   # Bottom row
        ]
        return "\n".join(["".join(row) for row in rows])
    
    def get_rows(self):
        """Return the layout as 3 rows for vector conversion"""
        return [
            "".join(self.letters[:10]),   # Top row
            "".join(self.letters[10:19]), # Middle row
            "".join(self.letters[19:])    # Bottom row
        ]


def generate_layout_from_rows(rows, row_stagger_offsets, randomize=False, seed=None):
    """Generate a layout dictionary mapping letters -> (x, y) coordinates"""
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


def layout_to_vector(layout):
    """Convert a keyboard layout to a flattened 52-dimensional vector"""
    letters = 'abcdefghijklmnopqrstuvwxyz'
    vector = []
    
    for letter in letters:
        if letter in layout:
            x, y = layout[letter]
            vector.extend([x, y])
        else:
            vector.extend([0.0, 0.0])
    
    return vector


def rows_to_vector(rows):
    """Convert keyboard rows to a flattened 52-dimensional vector"""
    layout = generate_layout_from_rows(rows, row_stagger_offsets, randomize=False)
    return layout_to_vector(layout)


def evaluate_fitness(layout):
    """
    Evaluate the fitness of a keyboard layout using the trained neural network.
    Lower cost = higher fitness (better layout).
    """
    if model is None or scaler_X is None:
        print("❌ Model or scaler not available for fitness evaluation")
        return float('inf')
    
    try:
        # Convert layout to vector
        vector = rows_to_vector(layout.get_rows())
        
        # Scale the input vector
        vector_array = np.array(vector).reshape(1, -1)
        vector_scaled = scaler_X.transform(vector_array)
        
        # Make prediction using the neural network
        predicted_cost_scaled = model.predict(vector_scaled, verbose=0)
        predicted_cost = float(predicted_cost_scaled[0][0]) / 1000
        
        return predicted_cost
        
    except Exception as e:
        print(f"❌ Error evaluating fitness: {e}")
        return float('inf')


def generate_random_layout():
    """Generate a random valid keyboard layout (permutation of letters)"""
    letters = list("qwertyuiopasdfghjklzxcvbnm")
    random.shuffle(letters)
    
    # Ensure the layout is valid (all 26 letters present)
    letters = ensure_valid_layout(letters)
    return KeyboardLayout(letters)


def tournament_selection(population, fitness_scores, tournament_size=3):
    """Select parents using tournament selection"""
    selected = []
    for _ in range(2):  # Select 2 parents
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
        selected.append(population[winner_idx])
    return selected


def crossover(parent1, parent2):
    """
    Perform structure-aware crossover between two parent layouts.
    - 70%: Row-based order crossover (OX) within rows
    - 20%: Row-swap or column-shift operations
    - 10%: Fallback to whole-genome order crossover
    """
    operation = random.random()
    
    if operation < 0.7:  # 70% chance: Row-based crossover
        return row_based_crossover(parent1, parent2)
    elif operation < 0.9:  # 20% chance: Structure operations
        return structure_aware_crossover(parent1, parent2)
    else:  # 10% chance: Fallback to whole-genome
        return whole_genome_crossover(parent1, parent2)


def row_based_crossover(parent1, parent2):
    """Row-based order crossover within keyboard rows"""
    # Define row boundaries: top(0-9), middle(10-18), bottom(19-25)
    row_boundaries = [(0, 9), (10, 18), (19, 25)]
    
    child_letters = [''] * len(parent1.letters)
    
    for start, end in row_boundaries:
        # Choose crossover points within this row
        if end - start > 1:  # Only if row has more than 1 letter
            point1, point2 = sorted(random.sample(range(start, end + 1), 2))
        else:
            point1, point2 = start, end
        
        # Copy segment from parent1
        child_letters[point1:point2 + 1] = parent1.letters[point1:point2 + 1]
        
        # Fill remaining positions in this row from parent2
        remaining_in_row = [letter for letter in parent2.letters[start:end + 1] 
                           if letter not in child_letters[point1:point2 + 1]]
        
        # Safety check: if no remaining letters in this row, use parent1's letters
        if not remaining_in_row:
            remaining_in_row = [letter for letter in parent1.letters[start:end + 1] 
                               if letter not in child_letters[point1:point2 + 1]]
        
        # If still no letters, fill with any available letters
        if not remaining_in_row:
            all_letters = list('abcdefghijklmnopqrstuvwxyz')
            used_letters = set(child_letters[point1:point2 + 1])
            remaining_in_row = [letter for letter in all_letters if letter not in used_letters]
        
        # Fill before point1 in this row
        for i in range(start, point1):
            if child_letters[i] == '' and remaining_in_row:
                child_letters[i] = remaining_in_row.pop(0)
        
        # Fill after point2 in this row
        for i in range(point2 + 1, end + 1):
            if child_letters[i] == '' and remaining_in_row:
                child_letters[i] = remaining_in_row.pop(0)
    
    # Ensure the layout is valid (all 26 letters present)
    child_letters = ensure_valid_layout(child_letters)
    return KeyboardLayout(child_letters)


def structure_aware_crossover(parent1, parent2):
    """Row-swap or column-shift operations"""
    if random.random() < 0.5:  # 50% chance for each operation
        return row_swap_crossover(parent1, parent2)
    else:
        return column_shift_crossover(parent1, parent2)


def row_swap_crossover(parent1, parent2):
    """Swap entire rows between parents"""
    # Define row boundaries
    row_boundaries = [(0, 9), (10, 18), (19, 25)]
    
    # Randomly choose which rows to swap
    swap_rows = random.sample(range(3), 2)
    
    child_letters = parent1.letters.copy()
    
    for row_idx in swap_rows:
        start, end = row_boundaries[row_idx]
        # Replace row from parent2
        child_letters[start:end + 1] = parent2.letters[start:end + 1]
    
    # Ensure the layout is valid (all 26 letters present)
    child_letters = ensure_valid_layout(child_letters)
    return KeyboardLayout(child_letters)


def column_shift_crossover(parent1, parent2):
    """Shift columns between parents"""
    # Define column positions (0-9 for top row, 0-8 for middle, 0-6 for bottom)
    column_ranges = [(0, 9), (0, 8), (0, 6)]
    
    # Choose random column to shift
    row_idx = random.randint(0, 2)
    start, end = column_ranges[row_idx]
    col = random.randint(start, end)
    
    # Map column to actual position in parent1
    if row_idx == 0:  # Top row
        pos1 = col
    elif row_idx == 1:  # Middle row
        pos1 = 10 + col
    else:  # Bottom row
        pos1 = 19 + col
    
    # Map column to actual position in parent2
    if row_idx == 0:  # Top row
        pos2 = col
    elif row_idx == 1:  # Middle row
        pos2 = 10 + col
    else:  # Bottom row
        pos2 = 19 + col
    
    child_letters = parent1.letters.copy()
    # Swap the letters at these column positions
    child_letters[pos1], child_letters[pos2] = child_letters[pos2], child_letters[pos1]
    
    # Ensure the layout is valid (all 26 letters present)
    child_letters = ensure_valid_layout(child_letters)
    return KeyboardLayout(child_letters)


def whole_genome_crossover(parent1, parent2):
    """Original whole-genome order crossover as fallback"""
    # Choose two random crossover points
    size = len(parent1.letters)
    point1, point2 = sorted(random.sample(range(size), 2))
    
    # Create child
    child_letters = [''] * size
    
    # Copy the segment from parent1
    child_letters[point1:point2] = parent1.letters[point1:point2]
    
    # Fill remaining positions with letters from parent2 in order
    remaining_letters = [letter for letter in parent2.letters if letter not in child_letters[point1:point2]]
    
    # Safety check: if no remaining letters, use parent1's letters instead
    if not remaining_letters:
        remaining_letters = [letter for letter in parent1.letters if letter not in child_letters[point1:point2]]
    
    # If still no letters, fill with random letters to ensure validity
    if not remaining_letters:
        all_letters = list('abcdefghijklmnopqrstuvwxyz')
        used_letters = set(child_letters[point1:point2])
        remaining_letters = [letter for letter in all_letters if letter not in used_letters]
    
    # Fill before point1
    for i in range(point1):
        if child_letters[i] == '' and remaining_letters:
            child_letters[i] = remaining_letters.pop(0)
    
    # Fill after point2
    for i in range(point2, size):
        if child_letters[i] == '' and remaining_letters:
            child_letters[i] = remaining_letters.pop(0)
    
    # Ensure the layout is valid (all 26 letters present)
    child_letters = ensure_valid_layout(child_letters)
    return KeyboardLayout(child_letters)


def crossover_with_tracking(parent1, parent2):
    """Crossover with method tracking for reporting"""
    operation = random.random()
    
    if operation < 0.7:  # 70% chance: Row-based crossover
        return row_based_crossover(parent1, parent2), 'row_based'
    elif operation < 0.9:  # 20% chance: Structure operations
        return structure_aware_crossover(parent1, parent2), 'structure'
    else:  # 10% chance: Fallback to whole-genome
        return whole_genome_crossover(parent1, parent2), 'whole_genome'


def hamming_distance(layout1, layout2):
    """Calculate Hamming distance between two layouts"""
    return sum(1 for a, b in zip(layout1.letters, layout2.letters) if a != b)


def inject_diversity(population, fitness_scores, best_layout):
    """Inject diversity by replacing worst individuals with fresh random layouts"""
    population_size = len(population)
    
    # Determine how many individuals to replace (5-10% of population)
    replace_count = random.randint(
        max(1, int(population_size * 0.05)),  # At least 5%
        max(2, int(population_size * 0.10))   # At most 10%
    )
    
    # Find indices of worst individuals
    worst_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:replace_count]
    
    diversity_injected = 0
    
    for idx in worst_indices:
        # Generate fresh random layout
        new_layout = generate_random_layout()
        
        # Check Hamming distance from best layout to avoid clones
        if hamming_distance(new_layout, best_layout) >= 5:  # At least 5 different positions
            population[idx] = new_layout
            diversity_injected += 1
        else:
            # If too similar, generate another one
            attempts = 0
            while attempts < 10:  # Try up to 10 times
                new_layout = generate_random_layout()
                if hamming_distance(new_layout, best_layout) >= 5:
                    population[idx] = new_layout
                    diversity_injected += 1
                    break
                attempts += 1
            
            # If still can't find diverse enough layout, use original but with some mutations
            if attempts >= 10:
                mutated_layout = mutate(best_layout, 0.3)  # High mutation rate
                population[idx] = mutated_layout
                diversity_injected += 1
    
    return diversity_injected


def save_best_candidate(best_layout, best_cost, run_params, filename=None):
    """Save the best candidate to a JSON file"""
    if filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"best_candidate_{timestamp}.json"
    
    data = {
        'best_layout': best_layout.letters,
        'best_cost': best_cost,
        'run_params': run_params,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'layout_visualization': str(best_layout)
    }
    
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Best candidate saved to: {filename}")
        return filename
    except Exception as e:
        print(f"❌ Failed to save best candidate: {e}")
        return None


def mutate(layout, mutation_rate=0.1):
    """Create a mutated copy of the layout"""
    new_letters = layout.letters.copy()  # Create a copy!
    for _ in range(int(len(new_letters) * mutation_rate)):
        idx1, idx2 = random.sample(range(len(new_letters)), 2)
        new_letters[idx1], new_letters[idx2] = new_letters[idx2], new_letters[idx1]
    
    # Ensure the layout is valid (all 26 letters present)
    new_letters = ensure_valid_layout(new_letters)
    return KeyboardLayout(new_letters)


def validate_layout(letters):
    """Validate that layout contains exactly 26 unique letters a-z"""
    expected = set('abcdefghijklmnopqrstuvwxyz')
    actual = set(letters)
    return actual == expected


def ensure_valid_layout(letters):
    """Ensure layout contains exactly 26 unique letters a-z, fixing any issues"""
    if len(letters) != 26:
        # If length is wrong, create a new valid layout
        return list('abcdefghijklmnopqrstuvwxyz')
    
    # Check for duplicates or missing letters
    letter_set = set(letters)
    expected_set = set('abcdefghijklmnopqrstuvwxyz')
    
    if letter_set == expected_set:
        return letters  # Already valid
    
    # Find missing letters
    missing_letters = expected_set - letter_set
    
    # Find duplicate letters (letters that appear more than once)
    letter_counts = {}
    for letter in letters:
        letter_counts[letter] = letter_counts.get(letter, 0) + 1
    
    duplicate_letters = []
    for letter, count in letter_counts.items():
        if count > 1:
            duplicate_letters.extend([letter] * (count - 1))
    
    # Replace duplicates with missing letters
    fixed_letters = letters.copy()
    missing_list = list(missing_letters)
    duplicate_list = duplicate_letters.copy()
    
    for i, letter in enumerate(fixed_letters):
        if letter in duplicate_list and missing_list:
            fixed_letters[i] = missing_list.pop(0)
            duplicate_list.remove(letter)
    
    return fixed_letters


def batch_evaluate_fitness(layouts):
    """Evaluate fitness for multiple layouts in batch"""
    if model is None or scaler_X is None:
        return [float('inf')] * len(layouts)
    
    vectors = [rows_to_vector(layout.get_rows()) for layout in layouts]
    vectors_array = np.array(vectors)
    vectors_scaled = scaler_X.transform(vectors_array)
    
    predicted_costs = model.predict(vectors_scaled, verbose=0)
    return [float(cost[0]) / 1000 for cost in predicted_costs]


def plot_optimization_progress(best_history, avg_history, qwerty_cost, diversity_history=None, 
                             unique_individuals_history=None, run_params=None, save_plot=True):
    """Plot the optimization progress with enhanced annotations"""
    
    # Safety check: ensure all history arrays have the same length
    min_length = min(len(best_history), len(avg_history))
    if diversity_history is not None:
        min_length = min(min_length, len(diversity_history))
    if unique_individuals_history is not None:
        min_length = min(min_length, len(unique_individuals_history))
    
    # Truncate all arrays to the same length
    best_history = best_history[:min_length]
    avg_history = avg_history[:min_length]
    if diversity_history is not None:
        diversity_history = diversity_history[:min_length]
    if unique_individuals_history is not None:
        unique_individuals_history = unique_individuals_history[:min_length]
    
    # Create figure with subplots
    if diversity_history is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(14, 6))
    
    generations = range(len(best_history))
    
    # Main fitness plot
    ax1.plot(generations, best_history, 'b-', linewidth=2, label='Best Fitness')
    ax1.plot(generations, avg_history, 'r--', linewidth=2, label='Average Fitness')
    ax1.axhline(y=qwerty_cost, color='g', linestyle=':', linewidth=2, label='QWERTY Cost')
    
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Predicted Cost (Lower is Better)')
    
    # Enhanced title with hyperparameters
    title = 'Genetic Algorithm Optimization Progress'
    if run_params:
        title += f'\nPopulation: {run_params["population_size"]}, Generations: {run_params["generations"]}, Tournament: {run_params["tournament_size"]}'
        if run_params.get("adaptive_mutation"):
            title += ', Adaptive Mutation'
        else:
            title += f', Mutation Rate: {run_params.get("mutation_rate", "N/A")}'
    
    # Calculate total improvement
    if best_history:
        total_improvement = ((best_history[0] - best_history[-1]) / best_history[0] * 100)
        title += f'\nTotal Improvement: {total_improvement:.1f}%'
    
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Diversity plot (if data provided)
    if diversity_history is not None:
        ax2.plot(generations, diversity_history, 'purple', linewidth=2, label='Mean Hamming Distance from Best')
        if unique_individuals_history:
            ax2_twin = ax2.twinx()
            ax2_twin.plot(generations, unique_individuals_history, 'orange', linewidth=2, label='Unique Individuals %')
            ax2_twin.set_ylabel('Unique Individuals (%)', color='orange')
            ax2_twin.tick_params(axis='y', labelcolor='orange')
        
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Mean Hamming Distance', color='purple')
        ax2.tick_params(axis='y', labelcolor='purple')
        ax2.set_title('Population Diversity Metrics')
        ax2.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        if unique_individuals_history:
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax2.legend()
    
    plt.tight_layout()
    
    if save_plot:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f'optimization_progress_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Saved optimization progress plot to '{filename}'")
    
    plt.show()


def get_adaptive_mutation_rate(generation, max_generations, start_rate=0.12, end_rate=0.04, decay_generations=40):
    """
    Calculate adaptive mutation rate that linearly decays from start_rate to end_rate.
    
    Args:
        generation: Current generation (0-indexed)
        max_generations: Maximum number of generations
        start_rate: Starting mutation rate (default: 0.12)
        end_rate: Final mutation rate (default: 0.04)
        decay_generations: Number of generations to decay over (default: 40)
    
    Returns:
        float: Current mutation rate
    """
    if generation < decay_generations:
        # Linear decay from start_rate to end_rate
        decay_factor = generation / decay_generations
        current_rate = start_rate - (start_rate - end_rate) * decay_factor
    else:
        # Keep at end_rate after decay_generations
        current_rate = end_rate
    
    return current_rate


def genetic_algorithm(population_size=50, generations=200, mutation_rate=None, tournament_size=3, 
                     plot_progress=True, patience=15, adaptive_mutation=True, load_best_candidate=None):
    """
    Main genetic algorithm for optimizing keyboard layouts.
    
    Args:
        population_size: Size of the population
        generations: Maximum number of generations
        mutation_rate: Fixed mutation rate (if None and adaptive_mutation=True, uses adaptive rate)
        tournament_size: Size of tournament for selection
        plot_progress: Whether to plot optimization progress
        patience: Number of generations without improvement before early stopping
        adaptive_mutation: Whether to use adaptive mutation rates
        load_best_candidate: Path to saved best candidate file to resume from
    """
    print("🧬 Genetic Algorithm for Keyboard Layout Optimization")
    print("=" * 60)
    print(f"Population size: {population_size}")
    print(f"Max generations: {generations}")
    print(f"Tournament size: {tournament_size}")
    print(f"Patience: {patience} generations")
    if adaptive_mutation:
        print(f"Mutation rate: Adaptive (0.12 → 0.04 over 40 generations)")
    else:
        print(f"Mutation rate: Fixed at {mutation_rate}")
    if load_best_candidate:
        print(f"🔄 Resuming from saved candidate: {load_best_candidate}")
    print()
    
    # Initialize population
    print("🎯 Initializing population with random layouts...")
    population = [generate_random_layout() for _ in range(population_size)]
    print(f"✓ Created {population_size} random keyboard layouts")
    print()
    
    best_fitness_history = []
    avg_fitness_history = []
    diversity_history = []
    unique_individuals_history = []
    
    # Enhanced tracking variables
    best_fitness_ever = float('inf')
    generations_without_improvement = 0
    mutation_improvements = 0
    crossover_improvements = 0
    total_mutations = 0
    total_crossovers = 0
    
    # Load best candidate if specified
    if load_best_candidate and os.path.exists(load_best_candidate):
        try:
            with open(load_best_candidate, 'r') as f:
                saved_data = json.load(f)
                best_candidate_letters = saved_data['best_layout']
                print(f"✓ Loaded best candidate with cost: {saved_data['best_cost']:.4f}")
                # Insert the best candidate into the population
                population[0] = KeyboardLayout(best_candidate_letters)
        except Exception as e:
            print(f"⚠ Failed to load best candidate: {e}")
            print("   Starting with random population instead")
    
    # Calculate actual QWERTY cost for comparison
    print("🎯 Calculating QWERTY baseline cost...")
    qwerty_layout = KeyboardLayout()  # Default QWERTY layout
    qwerty_cost = evaluate_fitness(qwerty_layout)
    print(f"✓ QWERTY cost: {qwerty_cost:.4f}")
    print()
    
    print("🔄 Starting evolution process...")
    print("-" * 40)
    
    for generation in range(generations):
        # Calculate current mutation rate
        if adaptive_mutation and mutation_rate is None:
            current_mutation_rate = get_adaptive_mutation_rate(generation, generations)
        else:
            current_mutation_rate = mutation_rate or 0.1
        
        # Show mutation rate every 10 generations
        if generation % 10 == 0:
            print(f"   🧬 Current mutation rate: {current_mutation_rate:.3f}")
        print(f"\n📊 Generation {generation + 1}/{generations}")
        print("   🤖 Using neural network to evaluate fitness for all layouts...")
        
        # Validate all layouts before fitness evaluation
        for i, layout in enumerate(population):
            if not validate_layout(layout.letters):
                print(f"   ⚠️  Invalid layout detected at index {i}, fixing...")
                layout.letters = ensure_valid_layout(layout.letters)
        
        # Evaluate fitness for all layouts using the trained model
        fitness_scores = [evaluate_fitness(layout) for layout in population]
        
        # Track best and average fitness
        best_fitness = min(fitness_scores)
        avg_fitness = np.mean(fitness_scores)
        worst_fitness = max(fitness_scores)
        best_layout = population[fitness_scores.index(best_fitness)]
        
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        
        # Early stopping logic
        if best_fitness < best_fitness_ever:
            best_fitness_ever = best_fitness
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1
        
        # Check for early stopping
        if generations_without_improvement >= patience:
            print(f"\n   🛑 Early stopping triggered!")
            print(f"   📉 No improvement for {patience} generations")
            print(f"   🎯 Best fitness: {best_fitness_ever:.4f}")
            # Ensure all history arrays have the same length before breaking
            while len(diversity_history) < len(best_fitness_history):
                diversity_history.append(diversity_history[-1] if diversity_history else 0.0)
            while len(unique_individuals_history) < len(best_fitness_history):
                unique_individuals_history.append(unique_individuals_history[-1] if unique_individuals_history else 100.0)
            break
        
        # Progress update
        if generation % 5 == 0 or generation == generations - 1:
            print(f"   🏆 Best: {best_fitness:.4f} | 📊 Avg: {avg_fitness:.4f} | 📉 Worst: {worst_fitness:.4f}")
            print(f"   🎯 Best layout so far:")
            print(f"   {str(best_layout).replace(chr(10), chr(10) + '   ')}")
            print(f"   🧬 Mutation rate: {current_mutation_rate:.3f}")
            print(f"   ⏰ Generations without improvement: {generations_without_improvement}")
            print(f"   👑 Elite preservation: Top 3 individuals protected from mutation/crossover")
        
        # Create new population
        print("   🔄 Creating new population...")
        new_population = []
        
        # Enhanced Elitism: keep the top 3 individuals unchanged
        elite_count = 3
        elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[:elite_count]
        elite_layouts = [population[i] for i in elite_indices]
        elite_costs = [fitness_scores[i] for i in elite_indices]
        
        new_population.extend(elite_layouts)
        print(f"   ✓ Elitism: Preserved top {elite_count} individuals:")
        for i, (layout, cost) in enumerate(zip(elite_layouts, elite_costs)):
            print(f"      {i+1}. Cost: {cost:.4f}")
        
        # Calculate enhanced diversity metrics (after elite_layouts is defined)
        diversity_scores = [hamming_distance(layout, best_layout) for layout in population]
        avg_diversity = np.mean(diversity_scores)
        
        # Calculate diversity from elite (top 3)
        elite_diversity_scores = [hamming_distance(layout, elite_layouts[0]) for layout in population]
        avg_elite_diversity = np.mean(elite_diversity_scores)
        
        # Calculate percentage of unique individuals
        unique_layouts = set(tuple(layout.letters) for layout in population)
        unique_percentage = (len(unique_layouts) / len(population)) * 100
        
        # Store metrics for plotting
        diversity_history.append(avg_diversity)
        unique_individuals_history.append(unique_percentage)
        
        # Enhanced progress update with diversity metrics
        if generation % 5 == 0 or generation == generations - 1:
            print(f"   🌱 Population diversity: {avg_diversity:.1f} avg Hamming distance from best")
            print(f"   🏆 Elite diversity: {avg_elite_diversity:.1f} avg Hamming distance from elite")
            print(f"   🔢 Unique individuals: {unique_percentage:.1f}%")
            print(f"   📈 Genetic operations: Mutations {mutation_improvements}/{total_mutations} improved, Crossovers {crossover_improvements}/{total_crossovers} improved")
        
        # Generate rest of population through selection, crossover, and mutation
        children_needed = population_size - elite_count
        children_created = 0
        
        # Track crossover method usage for this generation
        crossover_methods = {'row_based': 0, 'structure': 0, 'whole_genome': 0}
        
        while len(new_population) < population_size:
            # Select parents
            parents = tournament_selection(population, fitness_scores, tournament_size)
            
            # Perform crossover and track method
            child, method_used = crossover_with_tracking(parents[0], parents[1])
            crossover_methods[method_used] += 1
            total_crossovers += 1
            
            # Store pre-mutation fitness for comparison
            pre_mutation_fitness = evaluate_fitness(child)
            
            # Apply mutation with current rate
            child = mutate(child, current_mutation_rate)
            total_mutations += 1
            
            # Check if mutation improved the child
            post_mutation_fitness = evaluate_fitness(child)
            if post_mutation_fitness < pre_mutation_fitness:
                mutation_improvements += 1
            
            # Check if crossover produced improvement over parents
            parent_fitness = min(evaluate_fitness(parents[0]), evaluate_fitness(parents[1]))
            if pre_mutation_fitness < parent_fitness:
                crossover_improvements += 1
            
            new_population.append(child)
            children_created += 1
        
        # Show crossover method distribution
        if children_created > 0:
            print(f"   🔀 Crossover methods used: Row-based: {crossover_methods['row_based']}, Structure: {crossover_methods['structure']}, Whole-genome: {crossover_methods['whole_genome']}")
        
        print(f"   ✓ Created {children_created} children through crossover and mutation")
        
        # Replace old population
        population = new_population
        print(f"   ✓ Population updated ({len(population)} individuals)")
        
        # Diversity injection every 10 generations
        if generation > 0 and generation % 10 == 0:
            diversity_injected = inject_diversity(population, fitness_scores, best_layout)
            if diversity_injected > 0:
                print(f"   🌱 Diversity injection: Replaced {diversity_injected} worst individuals with fresh random layouts")
        
        # Show improvement over QWERTY
        qwerty_improvement = ((qwerty_cost - best_fitness) / qwerty_cost * 100)
        print(f"   📈 Improvement over QWERTY: {qwerty_improvement:.1f}%")
    
    # Final evaluation
    print(f"\n🎯 Final evaluation of best layouts...")
    final_fitness_scores = [evaluate_fitness(layout) for layout in population]
    best_final_fitness = min(final_fitness_scores)
    best_final_layout = population[final_fitness_scores.index(best_final_fitness)]
    
    print("\n" + "=" * 60)
    print("🏆 OPTIMIZATION COMPLETE!")
    print("=" * 60)
    print(f"🎯 Best layout found after {generations} generations:")
    print(f"   {str(best_final_layout).replace(chr(10), chr(10) + '   ')}")
    # Determine actual generations completed
    actual_generations = len(best_fitness_history)
    early_stopped = actual_generations < generations
    
    print(f"\n📊 Final Results:")
    print(f"   🤖 Predicted cost: {best_final_fitness:.4f}")
    print(f"   📈 Improvement over QWERTY: {((qwerty_cost - best_final_fitness) / qwerty_cost * 100):.1f}%")
    print(f"   🎯 QWERTY cost: {qwerty_cost:.4f}")
    print(f"   💡 Cost reduction: {qwerty_cost - best_final_fitness:.4f}")
    
    if early_stopped:
        print(f"   ⏰ Early stopping: Stopped at generation {actual_generations}")
    
    # Save best candidate
    run_params = {
        'population_size': population_size,
        'generations': generations,
        'mutation_rate': mutation_rate,
        'tournament_size': tournament_size,
        'patience': patience,
        'adaptive_mutation': adaptive_mutation
    }
    saved_file = save_best_candidate(best_final_layout, best_final_fitness, run_params)
    
    # Plot optimization progress
    if plot_progress:
        print(f"\n📊 Generating optimization progress plot...")
        plot_optimization_progress(
            best_fitness_history, avg_fitness_history, qwerty_cost,
            diversity_history, unique_individuals_history, run_params
        )
    
    # Print optimization progress summary
    print(f"\n📈 Optimization Progress Summary:")
    print(f"   🚀 Starting best cost: {best_fitness_history[0]:.4f}")
    print(f"   🏆 Final best cost: {best_final_fitness:.4f}")
    print(f"   📊 Total improvement: {((best_fitness_history[0] - best_final_fitness) / best_fitness_history[0] * 100):.1f}%")
    print(f"   🔄 Generations completed: {actual_generations}/{generations}")
    print(f"   👥 Population size: {population_size}")
    print(f"   👑 Elite preservation: Top 3 individuals protected each generation")
    print(f"   🌱 Diversity injection: Every 10 generations, 5-10% worst individuals replaced")
    print(f"   📈 Genetic operations: Mutations {mutation_improvements}/{total_mutations} improved ({mutation_improvements/total_mutations*100:.1f}%)")
    print(f"   🔀 Crossover operations: {crossover_improvements}/{total_crossovers} improved ({crossover_improvements/total_crossovers*100:.1f}%)")
    if early_stopped:
        print(f"   🛑 Early stopping: {patience} generations without improvement")
    
    return best_final_layout, best_final_fitness, best_fitness_history, avg_fitness_history, qwerty_cost


if __name__ == "__main__":
    # Set random seed for reproducibility
    s = 100
    random.seed(s)
    np.random.seed(s)
    
    # Example: Resume from a previous best candidate (uncomment to use)
    load_best_candidate = "best_candidate_20241201_143022.json"
    #load_best_candidate = None
    
    # Run genetic algorithm
    best_layout, best_cost, best_history, avg_history, qwerty_cost = genetic_algorithm(
        population_size=50,
        generations=200,
        mutation_rate=None,  # Use adaptive mutation
        tournament_size=10,
        plot_progress=True,
        patience=15,  # Early stopping after 15 generations without improvement
        adaptive_mutation=True,
        load_best_candidate=load_best_candidate
    )
    
    print("\n📊 Optimization Summary:")
    print(f"Final best cost: {best_cost:.4f}")
    print(f"QWERTY cost: {qwerty_cost:.4f}")
    print(f"Improvement: {((qwerty_cost - best_cost) / qwerty_cost * 100):.1f}%")
    
    # Example of how to resume from saved candidate:
    # print(f"\n💡 To resume from this run, use:")
    # print(f"   load_best_candidate='{saved_file}'")
