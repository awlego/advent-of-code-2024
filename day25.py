from dataclasses import dataclass
from itertools import product

@dataclass
class Lock:
   pin_heights: list[int]
   id: int = 0

@dataclass
class Key:
    key_heights: list[int]
    id: int = 0

def shape_from_chunk(chunk: list[str], shape_class: type[Key | Lock]) -> Key | Lock:
    relevant_lines = chunk[1:-1]
    pin_heights = []
    for i in range(len(relevant_lines)):
        pin_height = 0
        for j in range(len(relevant_lines[0])):
            if relevant_lines[j][i] == "#":
                pin_height += 1
        pin_heights.append(pin_height)

    return shape_class(key_heights=pin_heights) if shape_class is Key else shape_class(pin_heights=pin_heights)

def parse_input(input_file: str) -> tuple[list[Lock], list[Key]]:
    chunks = []
    locks = []
    keys = []
    key_id = 0
    lock_id = 0
    with open(input_file, "r") as f:
        # Read all lines and add an extra newline
        lines = f.readlines() + ["\n"]
        chunk = []
        for line in lines:
            if line.strip() == "":
                chunks.append(chunk)
                chunk = []
            else:
                chunk.append(line.strip())
    
    print(f"Found {len(chunks)} chunks")
    for chunk in chunks:
        if chunk[0].startswith("#"):
            lock = shape_from_chunk(chunk, Lock)
            lock.id = lock_id
            locks.append(lock)
            lock_id += 1
        if chunk[0].startswith("."):
            key = shape_from_chunk(chunk, Key)
            key.id = key_id
            keys.append(key)
            key_id += 1

    return locks, keys

def check_if_key_fits_lock(key: Key, lock: Lock, exact_fit: bool = False) -> bool:
    lock_height = 5
    for i in range(len(key.key_heights)):
        if exact_fit:
            if key.key_heights[i] + lock.pin_heights[i] != lock_height:
                return False
        else:
            if key.key_heights[i] + lock.pin_heights[i] > lock_height:
                return False
    return True

locks, keys = parse_input("inputs/day25_input.txt")
print(locks)
print(keys)

unique_pairs = set()
for lock, key in product(locks, keys):
    if check_if_key_fits_lock(key, lock, exact_fit=False):
        print(f"Key {key} fits lock {lock}")
        unique_pairs.add((lock.id, key.id))

print(f"Found {len(unique_pairs)} unique pairs")
