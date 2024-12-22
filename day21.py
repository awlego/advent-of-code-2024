from typing import List, Tuple
from dataclasses import dataclass
from functools import cache


def manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

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
    cache_hits = 0
    cache_misses = 0
    _path_cache = {}  # Single cache for paths
    _solve_cache = {}  # Keep this for full sequences
    
    @staticmethod
    def reset_cache_stats():
        DirectionalKeypad.cache_hits = 0
        DirectionalKeypad.cache_misses = 0
    
    @staticmethod
    def print_cache_stats():
        total = DirectionalKeypad.cache_hits + DirectionalKeypad.cache_misses
        hit_rate = (DirectionalKeypad.cache_hits / total * 100) if total > 0 else 0
        print(f"\nCache Statistics:")
        print(f"Hits: {DirectionalKeypad.cache_hits}")
        print(f"Misses: {DirectionalKeypad.cache_misses}")
        print(f"Hit Rate: {hit_rate:.2f}%")
        print(f"Total Calls: {total}")
    
    def __init__(self):
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
        
    def reset(self):
        self.loc = self.start
  
    @staticmethod
    def solve_cached(instructions: str, keypad_type: str = 'directional') -> list[str]:
        # Create a fresh keypad instance
        keypad = DirectionalKeypad() if keypad_type == 'directional' else NumericKeypad()
        
        def find_all_paths(curr_loc: tuple[int, int], target_loc: tuple[int, int]) -> list[str]:
            # Check path cache
            cache_key = (curr_loc, target_loc)
            if cache_key in DirectionalKeypad._path_cache:
                DirectionalKeypad.cache_hits += 1
                return DirectionalKeypad._path_cache[cache_key]
            
            DirectionalKeypad.cache_misses += 1
            
            if curr_loc == target_loc:
                return ["A"]
            
            paths = []
            dr, dc = target_loc[0] - curr_loc[0], target_loc[1] - curr_loc[1]
            
            possible_moves = []
            if dc > 0:
                possible_moves.append((">", (curr_loc[0], curr_loc[1] + 1)))
            if dc < 0:
                possible_moves.append(("<", (curr_loc[0], curr_loc[1] - 1)))
            if dr > 0:
                possible_moves.append(("v", (curr_loc[0] + 1, curr_loc[1])))
            if dr < 0:
                possible_moves.append(("^", (curr_loc[0] - 1, curr_loc[1])))
                
            for move, new_loc in possible_moves:
                if new_loc in keypad.keypad and keypad.keypad[new_loc] != "Error":
                    for subpath in find_all_paths(new_loc, target_loc):
                        paths.append(move + subpath)
            
            # Cache the result
            DirectionalKeypad._path_cache[cache_key] = paths
            return paths

        # Rest of solve_cached remains the same
        all_sequences = [""]
        curr_loc = keypad.start
        
        for instruction in instructions:
            target_loc = keypad.key_map[instruction]
            if target_loc == curr_loc:
                all_sequences = [seq + "A" for seq in all_sequences]
                continue
                
            new_sequences = []
            for base_seq in all_sequences:
                paths = find_all_paths(curr_loc, target_loc)
                new_sequences.extend([base_seq + path for path in paths])
            
            all_sequences = new_sequences
            curr_loc = target_loc
        
        return all_sequences
    
    def solve(self, instructions: str, keypad) -> list[str]:
        keypad_type = 'numeric' if isinstance(keypad, NumericKeypad) else 'directional'
        return self.solve_cached(instructions, keypad_type)

    
def generate_sequence(code: str) -> str:
    robot_1_keypad = NumericKeypad()
    num_robots = 5
    robots = [DirectionalKeypad() for _ in range(num_robots)]
    human_keypad = DirectionalKeypad()
    print(f"code: {code}")

    robot_instructions = robots[0].solve(code, robot_1_keypad)
    print(f"Found {len(robot_instructions)} possible R1 sequences")
    
    shortest_human_instructions = None
    shortest_length = float('inf')
    
    # Chain through all robots sequentially
    for sequence in robot_instructions:
        current_sequences = [sequence]
        
        # Pass through each robot in sequence
        for robot_idx in range(1, num_robots):
            new_sequences = []
            for seq in current_sequences:
                robot_results = robots[robot_idx].solve(seq, DirectionalKeypad())
                new_sequences.extend(robot_results)
            current_sequences = new_sequences
            
            if not current_sequences:
                break
        
        # If we made it through all robots, try human instructions
        for final_robot_seq in current_sequences:
            human_instructions = human_keypad.solve(final_robot_seq, DirectionalKeypad())
            
            # Update shortest if we found a better sequence
            for h_inst in human_instructions:
                if len(h_inst) < shortest_length:
                    shortest_length = len(h_inst)
                    shortest_human_instructions = h_inst
    
    print(f"Found shortest sequence (length {shortest_length})")
    return shortest_human_instructions

def run_directional_sequence(instructions: str) -> str:
    """Takes a sequence of directional pad instructions and returns what buttons get pressed."""
    keypad = DirectionalKeypad()
    pressed = []
    
    loc = keypad.start
    
    for instruction in instructions:
        if instruction == 'A':
            pressed.append(keypad.keypad[loc])
        elif instruction == '^':
            loc = (loc[0] - 1, loc[1])
        elif instruction == 'v':
            loc = (loc[0] + 1, loc[1])
        elif instruction == '<':
            loc = (loc[0], loc[1] - 1)
        elif instruction == '>':
            loc = (loc[0], loc[1] + 1)
    
    return ''.join(str(x) for x in pressed)

def run_numeric_sequence(instructions: str) -> str:
    """Takes a sequence of directional pad instructions and returns what numbers get pressed."""
    keypad = NumericKeypad()
    pressed = []
    
    # Start at A position
    loc = keypad.start
    
    for instruction in instructions:
        if instruction == 'A':
            # Record the number we're pointing at
            pressed.append(keypad.keypad[loc])
        elif instruction == '^':
            loc = (loc[0] - 1, loc[1])
        elif instruction == 'v':
            loc = (loc[0] + 1, loc[1])
        elif instruction == '<':
            loc = (loc[0], loc[1] - 1)
        elif instruction == '>':
            loc = (loc[0], loc[1] + 1)
    
    return ''.join(str(x) for x in pressed)

def verify_sequence(code: str) -> bool:
    """Verifies that a sequence correctly generates the target code."""
    keypad = DirectionalKeypad()
    robot_1_keypad = NumericKeypad()
    
    sequence = generate_sequence(code)
    
    robot_2_presses = run_directional_sequence(sequence)
    robot_1_presses = run_directional_sequence(robot_2_presses)
    final_code = run_numeric_sequence(robot_1_presses)
    
    print(f"Original code: {code}")
    print(f"Human sees: ({len(sequence)}) {sequence}")
    print(f"Robot 2 sees: ({len(robot_2_presses)}) {robot_2_presses}")
    print(f"Robot 1 sees: ({len(robot_1_presses)}) {robot_1_presses}")
    print(f"Final code: ({len(final_code)}) {final_code}")
    
    return code == final_code

def run_sequence_with_debug(instructions: str, keypad_type='directional') -> tuple[str, list]:
    """Takes a sequence of directional pad instructions and returns what buttons get pressed + path."""
    keypad = DirectionalKeypad() if keypad_type == 'directional' else NumericKeypad()
    pressed = []
    path = []
    
    loc = keypad.start
    path.append((loc, keypad.keypad[loc]))
    
    for instruction in instructions:
        if instruction == 'A':
            pressed.append(keypad.keypad[loc])
            path.append((loc, f"PRESS {keypad.keypad[loc]}"))
        elif instruction == '^':
            new_loc = (loc[0] - 1, loc[1])
            if new_loc in keypad.keypad:
                loc = new_loc
                path.append((loc, keypad.keypad[loc]))
            else:
                path.append((new_loc, "OUT OF BOUNDS"))
        elif instruction == 'v':
            new_loc = (loc[0] + 1, loc[1])
            if new_loc in keypad.keypad:
                loc = new_loc
                path.append((loc, keypad.keypad[loc]))
            else:
                path.append((new_loc, "OUT OF BOUNDS"))
        elif instruction == '<':
            new_loc = (loc[0], loc[1] - 1)
            if new_loc in keypad.keypad:
                loc = new_loc
                path.append((loc, keypad.keypad[loc]))
            else:
                path.append((new_loc, "OUT OF BOUNDS"))
        elif instruction == '>':
            new_loc = (loc[0], loc[1] + 1)
            if new_loc in keypad.keypad:
                loc = new_loc
                path.append((loc, keypad.keypad[loc]))
            else:
                path.append((new_loc, "OUT OF BOUNDS"))
    
    return ''.join(str(x) for x in pressed), path

def compare_sequences_zipped(seq1: str, seq2: str, keypad_type='directional'):
    """Compare two sequences side by side to highlight divergences"""
    def chunk_sequence(seq):
        chunks = []
        current_chunk = ""
        for char in seq:
            current_chunk += char
            if char == 'A':
                chunks.append(current_chunk)
                current_chunk = ""
        return chunks
    
    chunks1 = chunk_sequence(seq1)
    chunks2 = chunk_sequence(seq2)
    
    print(f"\nComparing sequences on {keypad_type} keypad (split by 'A' presses):")
    print(f"{'Seq1 (' + str(len(seq1)) + ')'.ljust(20)} | {'Seq2 (' + str(len(seq2)) + ')'.ljust(20)} | Same?")
    print("-" * 50)
    
    for i in range(max(len(chunks1), len(chunks2))):
        chunk1 = chunks1[i] if i < len(chunks1) else ""
        chunk2 = chunks2[i] if i < len(chunks2) else ""
        same = "✓" if chunk1 == chunk2 else "✗"
        print(f"{chunk1.ljust(20)} | {chunk2.ljust(20)} | {same}")
    
    # Also get what buttons are actually pressed
    result1, path1 = run_sequence_with_debug(seq1, keypad_type)
    result2, path2 = run_sequence_with_debug(seq2, keypad_type)
    
    print("\nButtons pressed:")
    print(f"Seq1: {result1}")
    print(f"Seq2: {result2}")
    print(f"Number of '<' in each sequence: {seq1.count('<')} {seq2.count('<')}")

def print_path_with_keypad(path, keypad_type='directional'):
    """Prints the path taken with the keypad layout for reference"""
    if keypad_type == 'directional':
        print("""    +---+---+
    | ^ | A |
+---+---+---+
| < | v | > |
+---+---+---+""")
    else:
        print("""+---+---+---+
| 7 | 8 | 9 |
+---+---+---+
| 4 | 5 | 6 |
+---+---+---+
| 1 | 2 | 3 |
+---+---+---+
    | 0 | A |
    +---+---+""")
    
    for loc, action in path:
        if "PRESS" in str(action):
            print(f"At {loc}: {action}")

def parse_input(filepath: str) -> str:
    codes = []
    with open(filepath, 'r') as openfile:
        for line in openfile:
            code = line.strip()
            codes.append(code)
    return codes

def main():
    DirectionalKeypad.reset_cache_stats()
    codes = parse_input('inputs/day21_input.txt')
    complexity_sum = 0
    for code in codes:
        seq = generate_sequence(code)
        complexity = int(code[:-1]) * len(seq)
        print(f"Code: {code}: {seq}, {complexity}")
        complexity_sum += complexity
        print()
    print(f"Total complexity: {complexity_sum}")
    DirectionalKeypad.print_cache_stats()


if __name__ == "__main__":
    # verify_sequence("37")
    # seq1 = "v<<A^>>AvA^Av<<A^>>AAv<A<A^>>AA<Av>AA^Av<A^>AA<A>Av<A<A^>>AAA<Av>A^A"
    # seq2 = "<v<A>>^AvA^A<vA<AA>>^AAvA<^A>AAvA^A<vA>^AA<A>A<v<A>A>^AAAvA<^A>A"
    # compare_sequences_zipped(seq1, seq2)

    main()

