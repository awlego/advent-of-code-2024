from typing import List, Tuple, Dict
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations


def parse_input(filepath: str) -> List[str]:
    lines = []
    with open(filepath, 'r') as openfile:
        for line in openfile:
            line = line.strip()
            row = []
            for char in line:
                row.append(int(char))
            lines.append(row)
    return lines

def find_trailheads(input) -> List[Tuple[int, int]]:
    trailheads = []
    for row_index, row in enumerate(input):
        for col_index, col in enumerate(row):
            if col == 0:
                trailheads.append((row_index, col_index))
    return trailheads

def score_trailhead(topo_map, trailhead: tuple[int, int]) -> int:
    '''Starting at the trailhead (row, col), look in the four cardinal directions for paths
    
    Valid paths always increase by exactly one, and trailheads are guaranteed to start at
    a height of 0.
        
    Max mountain height is 9
    
    The score is the number of unique 9 height peaks able to be reached from the trailhead.'''
    
    def is_valid(row: int, col: int) -> bool:
        """Check if a position is within bounds of the map"""
        return (0 <= row < len(topo_map) and 
                0 <= col < len(topo_map[0]))
    
    def dfs(row: int, col: int, current_height: int, visited: set, peaks: set) -> None:
        """Depth first search from current position, collects reachable peaks in peaks set"""
        # If we've reached a peak, add it to our peaks set
        if current_height == 9:
            peaks.add((row, col))
            return
            
        # Try all four cardinal directions
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        
        for dx, dy in directions:
            new_row, new_col = row + dy, col + dx
            
            # Check if new position is valid and not visited
            if (is_valid(new_row, new_col) and 
                (new_row, new_col) not in visited and
                topo_map[new_row][new_col] == current_height + 1):
                
                # Mark as visited, explore path, then backtrack
                visited.add((new_row, new_col))
                dfs(new_row, new_col, current_height + 1, visited, peaks)
                visited.remove((new_row, new_col))
    
    # Start DFS from trailhead
    visited = {trailhead}
    peaks = set()  # Set to collect unique peak positions
    dfs(trailhead[0], trailhead[1], 0, visited, peaks)
    return len(peaks)

def solve_input(filepath):
    input = parse_input(filepath)

    trailheads = find_trailheads(input)
    print(f"Found {len(trailheads)} trailheads")

    scores = {}
    for trailhead in trailheads:
        score = score_trailhead(input, trailhead)
        scores[trailhead] = score
        # print("location", trailhead, "score: ", score)


    result = sum(scores.values())
    print(f"Sum of scores of all trailheads: {result}")
    return result

def rate_trailhead(topo_map, trailhead: tuple[int, int], require_nine: bool = False) -> int:
    '''Starting at the trailhead (row, col), look in the four cardinal directions for paths
    
    Valid paths always increase by exactly one, and trailheads are guaranteed to start at
    a height of 0.
        
    A valid path must increase by exactly one each step and continue until it can't go higher.
    If require_nine is True, only paths reaching height 9 are counted.
    The score is the number of such unique paths from the trailhead.'''
    
    def is_valid(row: int, col: int) -> bool:
        """Check if a position is within bounds of the map"""
        return (0 <= row < len(topo_map) and 
                0 <= col < len(topo_map[0]))
    
    def can_climb_higher(row: int, col: int, current_height: int, visited: set) -> bool:
        """Check if there's any valid next step up from current position"""
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            new_row, new_col = row + dy, col + dx
            if (is_valid(new_row, new_col) and 
                (new_row, new_col) not in visited and
                topo_map[new_row][new_col] == current_height + 1):
                return True
        return False
    
    def dfs(row: int, col: int, current_height: int, visited: set) -> int:
        """Depth first search from current position, returns number of valid paths"""
        can_climb = can_climb_higher(row, col, current_height, visited)
        
        # If we require 9 and can't reach it, this isn't a valid path
        if require_nine and not can_climb and current_height < 9:
            return 0
            
        # If we can't climb higher and (we don't require 9 OR we reached 9), this is a valid path
        if not can_climb or current_height == 9:
            return 1
            
        total_paths = 0
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        
        for dx, dy in directions:
            new_row, new_col = row + dy, col + dx
            
            # Check if new position is valid and not visited
            if (is_valid(new_row, new_col) and 
                (new_row, new_col) not in visited and
                topo_map[new_row][new_col] == current_height + 1):
                
                # Mark as visited, explore path, then backtrack
                visited.add((new_row, new_col))
                total_paths += dfs(new_row, new_col, current_height + 1, visited)
                visited.remove((new_row, new_col))
        
        return total_paths
    
    # Start DFS from trailhead
    visited = {trailhead}
    return dfs(trailhead[0], trailhead[1], 0, visited)

def solve_b(filepath):
    input = parse_input(filepath)

    trailheads = find_trailheads(input)
    print(f"Found {len(trailheads)} trailheads")

    ratings = {}
    for trailhead in trailheads:
        rating = rate_trailhead(input, trailhead, True)
        ratings[trailhead] = rating

    result = sum(ratings.values())
    print(f"Sum of ratings of all trailheads: {result}")
    return result

def main():
    test_answer = solve_input("inputs/day10_input_test.txt")
    assert test_answer == 36
    solve_input("inputs/day10_input.txt")
    
    test_answer = solve_b("inputs/day10_input_test.txt")
    assert test_answer == 81
    solve_b("inputs/day10_input.txt")
    

if __name__ == "__main__":
    main()