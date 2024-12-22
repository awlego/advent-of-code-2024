from typing import List, Tuple
from dataclasses import dataclass
from functools import cache



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
def numeric_chunk_map(cur: tuple[int, int], target: tuple[int, int]) -> list[str]:
    '''Solves the fastest path(s) from cur to target on the numeric keypad.
    Returns both possible paths: vertical-then-horizontal and horizontal-then-vertical
        if they are the same length, otherwise returns the shortest path.
    cur and target are both tuples.

    Returns a list of strings, each representing a path from cur to target.'''

    dy = cur[0] - target[0]
    dx = cur[1] - target[1]
    paths = set()

    # Path 1: vertical then horizontal
    seq1 = ""
    if dy > 0:
        seq1 += "v" * abs(dy)
    if dy < 0:
        seq1 += "^" * abs(dy)
    if dx > 0:
        seq1 += ">" * abs(dx)
    if dx < 0:
        seq1 += "<" * abs(dx)
    paths.add(seq1 + "A")

    # Path 2: horizontal then vertical
    seq2 = ""
    if dx > 0:
        seq2 += ">" * abs(dx)
    if dx < 0:
        seq2 += "<" * abs(dx)
    if dy > 0:
        seq2 += "v" * abs(dy)
    if dy < 0:
        seq2 += "^" * abs(dy)
    paths.add(seq2 + "A")

    print(f"Cur, target: {cur}, {target}, paths: {paths}")
    shortest_seq_len = len(min(paths))
    return [p for p in paths if len(p) == shortest_seq_len]


def solve_numeric(code: str) -> list[str]:
    sequences = [""]  # Start with one empty sequence
    nk = NumericKeypad()
    
    for char in code:
        target = nk.key_map[char]
        chunks = numeric_chunk_map(nk.loc, target)
        print(f"resulting Chunks: {chunks}")
        
        # Create new sequences for each chunk
        new_sequences = []
        for existing_seq in sequences:
            for chunk in chunks:
                new_sequences.append(existing_seq + chunk)
        
        sequences = new_sequences
    
    shortest_seq_len = len(min(sequences))
    return [s for s in sequences if len(s) == shortest_seq_len]




def main():
    codes = parse_input('inputs/day21_input.txt')
    num_robots = 2
    for code in codes:
        numeric_code = int(code.strip('A'))
        numeric_seq = solve_numeric(code)
        print(f"Numeric sequence: {numeric_seq}")
        for i in range(num_robots):
            sequence = TODO(sequence)
            complexity += len(sequence) * numeric_code
    print(complexity)


if __name__ == "__main__":
    main()

