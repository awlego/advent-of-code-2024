from typing import List, Tuple
from functools import wraps
from tqdm import tqdm
from collections  import defaultdict
import numpy as np

class CacheStats:
    def __init__(self, name):
        self.name = name
        self.hits = 0
        self.misses = 0
        self.unique_inputs = set()
    
    def record_access(self, key, is_hit):
        if is_hit:
            self.hits += 1
        else:
            self.misses += 1
            self.unique_inputs.add(str(key))
    
    def __str__(self):
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return f"{self.name} Cache Stats:\n" + \
               f"Hits: {self.hits}\n" + \
               f"Misses: {self.misses}\n" + \
               f"Hit rate: {hit_rate:.2f}%\n" + \
               f"Unique inputs: {len(self.unique_inputs)}\n"
            #    f"Unique inputs: {self.unique_inputs}"

# Global stats objects
rules_cache_stats = CacheStats("Rules")
blink_cache_stats = CacheStats("Blink")



def parse_input(filename: str) -> str:
    with open(filename, 'r') as openfile:
        input = openfile.readline()
    return [int(stone) for stone in input.split(" ")]

def apply_rule_one(stone: int) -> Tuple[bool, int]:
    if stone == 0:
        stone = 1
        return True, [stone]
    return False, [stone]

def apply_rule_two(stone: int) -> Tuple[bool, any]:
    even_digits = len(str(stone)) % 2
    if even_digits == 1:
        return False, stone
    
    num_digits = len(str(stone))
    stone1 = int(str(stone)[0:num_digits//2])
    stone2 = int(str(stone)[num_digits//2:])
    return True, [stone1, stone2]

def apply_rule_three(stone: int) -> Tuple[bool, int]:
    return True, [stone * 2024]

def cache_rules(func):
    cache = {}
    @wraps(func)
    def wrapper(stone):
        key = stone
        is_hit = key in cache
        rules_cache_stats.record_access(key, is_hit)
        if not is_hit:
            cache[key] = func(stone)
        return cache[key]
    return wrapper

@cache_rules
def apply_rules(stone):
    applied, stones = apply_rule_one(stone)
    if not applied:
        applied, stones = apply_rule_two(stone)
    if not applied:
        applied, stones = apply_rule_three(stone)
    return stones

def cache_blinks(func):
    cache = {}
    @wraps(func)
    def wrapper(stones):
        key = tuple(sorted(stones))
        is_hit = key in cache
        blink_cache_stats.record_access(key, is_hit)
        if not is_hit:
            cache[key] = func(stones)
        return cache[key]
    return wrapper


@cache_blinks
def base_blink(stones: List[int]) -> List[int]:
    new_stones = []
    for stone in stones:
       new_stones.extend([stone for stone in apply_rules(stone)])
    return new_stones


@cache_blinks
def blink(stone_counts: defaultdict[int]) -> defaultdict[int]:
    ''''a dictionary of stones. 
    key: stone number
    value: count of stones numbered that number'''
    keys_to_process = list(stone_counts.keys())
    new_stone_counts = defaultdict(int)
    for key in keys_to_process:
        stones = apply_rules(key)
        # we made a new stone
        if len(stones) == 2:
            new_stone_counts[stones[1]] += stone_counts[key]
        new_stone_counts[stones[0]] += stone_counts[key]
        
    return new_stone_counts


def solve(stones, num_blinks) -> int:
    stones = {x: 1 for x in stones}
    for i in tqdm(range(num_blinks)):
        stones = blink(stones)
        print(f"blink number: {i}, num unique stones: {len(stones)}, total stones: {sum(stones.values())}")
        # print(rules_cache_stats)
        # print(blink_cache_stats)

    return sum(stones.values())


def solve_base(stones: List[int]) -> set:
    '''let's see if the base case is a repeating sequence'''

    base_stones = {0: stones}
    prev_stones = set()
    blink_count = 0
    # stones = blink(stones)
    # blink_count += 1
    new_numbers = set(stones)
    while (len(new_numbers) > 0):
        stones = base_blink(stones)
        base_stones[blink_count] = stones
        blink_count += 1
        new_numbers = set(stones) - prev_stones
        prev_stones.update(new_numbers)
    print(f"{blink_count-1} blinks to reach a repeating state.")
    return stones, base_stones

def main():
    input = parse_input("inputs/day11_input.txt")
    print("Initial stones:", input)

    print("\nSolving the default case for a 0...")
    base_case, base_stones = solve_base([0])
    base_case = set(base_case)
    print(f"Answer to part base_case: {len(base_case)}, {base_case}")
    print(f"base_stones:")
    for key, value in base_stones.items():
        print(key, value)
    base_stone_counts = {}
    for key, value in base_stones.items():
        base_stone_counts[key] = len(value)
    print(f"base_stone_counts: {base_stone_counts}")

    # okay so now I know that after 17 blinks, all new numbers generated from 0s will be stable...
    # we might be making more numbers, but they aren't UNIQUE ones we haven't seen yet.
    # so we should be able to pair down our input and keep solving for the unique values while
    # keeping track of how large our known stuff is getting each loop.

    # every time I would add a 0, instead add an age counter.

    print("\nSolving part A (25 blinks)...")
    a = solve(input, 25)
    print(f"Answer to part A: {a}")
    print("\nCache statistics after part A:")
    print(rules_cache_stats)
    print(blink_cache_stats)

    print("\nSolving part B (75 blinks)...")
    b = solve(input, 75)
    print(f"Answer to part B: {b}")
    print("\nFinal cache statistics:")
    print(rules_cache_stats)
    print(blink_cache_stats)

if __name__ == "__main__":
    main()


