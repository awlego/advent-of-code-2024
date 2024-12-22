from typing import List, Tuple
from dataclasses import dataclass
from functools import cache
from tqdm import tqdm


def parse_input(filepath: str) -> str:
    codes = []
    with open(filepath, 'r') as openfile:
        for line in openfile:
            code = line.strip()
            codes.append(code)
    return codes


class NumericKeypad():
    '''
    +---+---+---+
    | 7 | 8 | 9 |
    +---+---+---+
    | 4 | 5 | 6 |
    +---+---+---+
    | 1 | 2 | 3 |
    +---+---+---+
        | 0 | A |
        +---+---+
    '''
    def __init__(self):
        # coordinates start from upper left corner
        self.keypad = {
            (0, 0): 7, (0, 1): 8, (0, 2): 9,
            (1, 0): 4, (1, 1): 5, (1, 2): 6,
            (2, 0): 1, (2, 1): 2, (2, 2): 3,
            (3, 0): "Error", (3, 1): 0, (3, 2): 'A',
        } 
        self.key_map = {
            "7": (0, 0), "8": (0, 1), "9": (0, 2),
            "4": (1, 0), "5": (1, 1), "6": (1, 2),
            "1": (2, 0), "2": (2, 1), "3": (2, 2),
            "Error": (3, 0), "0": (3, 1), "A": (3, 2),
        }
        self.start = self.key_map['A']
        self.loc = self.start


class DirectionalKeypad():
    '''
        +---+---+
        | ^ | A |
    +---+---+---+
    | < | v | > |
    +---+---+---+
    '''    
    def __init__(self):
        # coordinates are start from upper left corner
        self.keypad = {
            (0, 0): "Error", (0, 1): "^", (0, 2): "A",
            (1, 0): "<", (1, 1): "v", (1, 2): ">",
        }
        self.key_map = {
            "Error": (0, 0),
            "^": (0, 1),
            "A": (0, 2),
            "<": (1, 0),
            "v": (1, 1),
            ">": (1, 2),
        }
        self.start = self.key_map['A']
        self.loc = self.start

@cache
def fastest_paths(cur: tuple[int, int], target: tuple[int, int], type) -> list[str]:
    '''Solves the fastest path(s) from cur to target on the numeric keypad.
    Returns both possible paths: vertical-then-horizontal and horizontal-then-vertical
        if they are the same length, otherwise returns the shortest path.
    cur and target are both tuples.

    Returns a list of strings, each representing a path from cur to target.'''
    banned_location = None
    if type == 'numeric':
        banned_location = (3, 0)
    elif type == 'directional':
        banned_location = (0, 0)
    else:
        raise ValueError(f"Invalid type: {type}")

    dy = cur[0] - target[0]
    dx = cur[1] - target[1]
    paths = set()

    def path_hits_banned(path: str, start: tuple[int, int]) -> bool:
        """Check if a path from start position hits the banned location"""
        pos = list(start)
        for move in path:
            if move == '^': pos[0] -= 1
            elif move == 'v': pos[0] += 1
            elif move == '<': pos[1] -= 1
            elif move == '>': pos[1] += 1
            elif move == 'A': break
            
            if tuple(pos) == banned_location:
                return True
        return False

    # Path 1: vertical then horizontal
    seq1 = ""
    if dy > 0:
        seq1 += "^" * abs(dy)
    if dy < 0:
        seq1 += "v" * abs(dy)
    if dx > 0:
        seq1 += "<" * abs(dx)
    if dx < 0:
        seq1 += ">" * abs(dx)
    if not path_hits_banned(seq1, cur):
        paths.add(seq1 + "A")

    # Path 2: horizontal then vertical
    seq2 = ""
    if dx > 0:
        seq2 += "<" * abs(dx)
    if dx < 0:
        seq2 += ">" * abs(dx)
    if dy > 0:
        seq2 += "^" * abs(dy)
    if dy < 0:
        seq2 += "v" * abs(dy)
    if not path_hits_banned(seq2, cur):
        paths.add(seq2 + "A")

    # print(f"Cur, target: {cur}, {target}, paths: {paths}")
    shortest_seq_len = len(min(paths))
    return [p for p in paths if len(p) == shortest_seq_len]

@cache
def solve_numeric(code: str) -> list[str]:
    sequences = [""]
    nk = NumericKeypad()
    
    for i, char in enumerate(code):
        target = nk.key_map[char]
        paths = fastest_paths(nk.loc, target, 'numeric')        
        
        new_sequences = []
        for existing_seq in sequences:
            for path in paths:
                new_sequences.append(existing_seq + path)

        sequences = new_sequences
        nk.loc = target  # Update the location after each move
    
    shortest_len = len(min(sequences, key=len))
    return [s for s in sequences if len(s) == shortest_len]

@cache
def solve_directional_chunk(chunk: str) -> list[str]:
    dk = DirectionalKeypad()
    # print(f"\nSolving chunk: {chunk}")
    sequences = [""]
    for char in chunk:
        target = dk.key_map[char]
        paths = fastest_paths(dk.loc, target, 'directional')
        # print(f"Current loc: {dk.loc}, target: {target}, paths: {paths}")

        new_sequences = []
        for existing_seq in sequences:
            for path in paths:
                new_sequences.append(existing_seq + path)

        sequences = new_sequences
        dk.loc = target
        # print(f"After char {char}, sequences: {sequences}")

    shortest_len = len(min(sequences, key=len))
    return [s for s in sequences if len(s) == shortest_len]

@cache
def get_min_sequence_length_dp(target_sequence: str, total_robots: int) -> int:
    # dp[i][seq] represents min length needed for i robots to generate seq
    dp = [{} for _ in range(total_robots + 1)]
    dp[0][target_sequence] = len(target_sequence)
    
    for robots in range(1, total_robots + 1):
        for seq in dp[robots-1]:
            possible_sequences = solve_directional_chunk(seq)
            for possible_seq in possible_sequences:
                if possible_seq not in dp[robots]:
                    dp[robots][possible_seq] = float('inf')
                dp[robots][possible_seq] = min(
                    dp[robots][possible_seq],
                    len(possible_seq)
                )
    
    return min(dp[total_robots].values()) if dp[total_robots] else float('inf')

def solve_with_dp(code: str, num_robots: int) -> str:
    """
    Solves for the shortest sequence using dynamic programming.
    """
    numeric_sequences = solve_numeric(code)
    
    # For each possible numeric sequence, find min length needed to generate it
    min_length = float('inf')
    best_numeric_seq = None
    for seq in numeric_sequences:
        length = get_min_sequence_length_dp(seq, num_robots)
        if length < min_length:
            min_length = length
            best_numeric_seq = seq
            
    def reconstruct_sequence(target: str, robots_left: int) -> str:
        """Reconstructs the sequence that generates target with given number of robots"""
        if robots_left == 0:
            return target
            
        possible_sequences = solve_directional_chunk(target)
        for seq in possible_sequences:
            # Try this sequence
            next_result = reconstruct_sequence(seq, robots_left - 1)
            if next_result is not None:
                # Verify the total length matches our expected minimum
                total_length = sum(len(s) for s in [next_result])
                if total_length == min_length:
                    return next_result
                
        return None

    result = reconstruct_sequence(best_numeric_seq, num_robots)
    if result is None:
        raise ValueError("Failed to reconstruct sequence")
    return result

def main():
    codes = parse_input('inputs/day21_input.txt')
    num_robots = 3
    total_complexity = 0 

    for code in codes:
        numeric_code = int(code.strip('A'))
        shortest_sequence = solve_with_dp(code, num_robots)
        print(f"Final sequence for code {code}: {shortest_sequence} ({len(shortest_sequence)})")
        total_complexity += len(shortest_sequence) * numeric_code

    print(f"Total complexity: {total_complexity}")
    return total_complexity

def verify_sequence(sequence: str, num_robots: int = 2) -> str:
    """
    Verifies a sequence by simulating each robot's actions.
    Returns the final numeric keypad sequence or raises an error if invalid.
    """
    def simulate_directional_keypad(sequence: str) -> str:
        """Simulates pressing buttons on directional keypad and returns resulting sequence"""
        result = ""
        # Start at 'A' position
        pos = (0, 2)  # Upper right corner for directional keypad
        
        for move in sequence:
            # Update position based on movement
            if move == '^' and pos[0] > 0:
                new_pos = (pos[0] - 1, pos[1])
            elif move == 'v' and pos[0] < 1:
                new_pos = (pos[0] + 1, pos[1])
            elif move == '<' and pos[1] > 0:
                new_pos = (pos[0], pos[1] - 1)
            elif move == '>' and pos[1] < 2:
                new_pos = (pos[0], pos[1] + 1)
            elif move == 'A':
                # Add the button being pressed to result
                if pos == (0, 0):  # Gap position
                    raise ValueError(f"Invalid position: tried to press button at gap {pos}")
                button_map = {
                    (0, 1): '^',
                    (0, 2): 'A',
                    (1, 0): '<',
                    (1, 1): 'v',
                    (1, 2): '>'
                }
                result += button_map[pos]
                continue
            else:
                raise ValueError(f"Invalid move: {move}")
            
            # Check if new position is valid (not the gap)
            if new_pos == (0, 0):  # Gap position
                raise ValueError(f"Invalid move: would cross gap at {new_pos}")
            pos = new_pos
        
        return result

    def simulate_numeric_keypad(sequence: str) -> str:
        """Simulates pressing buttons on numeric keypad and returns resulting sequence"""
        result = ""
        # Start at 'A' position
        pos = (3, 2)  # Bottom right corner for numeric keypad
        
        for move in sequence:
            # Update position based on movement
            if move == '^' and pos[0] > 0:
                new_pos = (pos[0] - 1, pos[1])
            elif move == 'v' and pos[0] < 3:
                new_pos = (pos[0] + 1, pos[1])
            elif move == '<' and pos[1] > 0:
                new_pos = (pos[0], pos[1] - 1)
            elif move == '>' and pos[1] < 2:
                new_pos = (pos[0], pos[1] + 1)
            elif move == 'A':
                # Add the button being pressed to result
                if pos == (3, 0):  # Gap position
                    raise ValueError(f"Invalid position: tried to press button at gap {pos}")
                button_map = {
                    (0, 0): '7', (0, 1): '8', (0, 2): '9',
                    (1, 0): '4', (1, 1): '5', (1, 2): '6',
                    (2, 0): '1', (2, 1): '2', (2, 2): '3',
                    (3, 1): '0', (3, 2): 'A'
                }
                result += button_map[pos]
                continue
            else:
                raise ValueError(f"Invalid move: {move}")
            
            # Check if new position is valid (not the gap)
            if new_pos == (3, 0):  # Gap position
                raise ValueError(f"Invalid move: would cross gap at {new_pos}")
            pos = new_pos
     
        return result

    current_sequence = sequence
    print("\nStarting verification chain:")
    print(f"Initial sequence: {current_sequence}")
    
    # Process through each robot
    for i in range(num_robots):
        next_sequence = simulate_directional_keypad(current_sequence)
        print(f"\nRobot {i+1} output: {next_sequence}")
        current_sequence = next_sequence
    
    print("\nFinal robot (numeric keypad):")
    return simulate_numeric_keypad(current_sequence)


if __name__ == "__main__":
    main()