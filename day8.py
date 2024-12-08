from typing import List, Tuple, Dict
from collections import defaultdict
from itertools import combinations

def parse_input(filepath: str) -> List[str]:
    lines = []
    with open(filepath, 'r') as openfile:
        for line in openfile:
            line = line.strip()
            row = []
            for char in line:
                row.append(char)
            lines.append(row)
    return lines

def find_antennas_by_frequency(input_map: List[str]) -> Dict[str, List[Tuple[int, int]]]:
    """Creates a dictionary mapping each frequency to a list of antenna positions"""
    frequency_positions = defaultdict(list)
    for row in range(len(input_map)):
        for col in range(len(input_map[row])):
            char = input_map[row][col]
            if char != '.': 
                frequency_positions[char].append((row, col))
    return frequency_positions

def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two points"""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def find_anti_node_positions(ant1: Tuple[int, int], ant2: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Calculate the exact positions of anti-nodes for a pair of antennas"""
    delta_row = ant2[0] - ant1[0]
    delta_col = ant2[1] - ant1[1]
    
    pos1 = (
        ant1[0] - delta_row,
        ant1[1] - delta_col
    )
    
    pos2 = (
        ant2[0] + delta_row,
        ant2[1] + delta_col
    )
    
    return [pos1, pos2]

def find_anti_node_positions_part2(ant1: Tuple[int, int], ant2: Tuple[int, int], rows: int, cols: int) -> List[Tuple[int, int]]:
    """Calculate all positions on the line through a pair of antennas that lie within bounds"""
    anti_nodes = []
    
    delta_row = (ant2[0] - ant1[0])
    delta_col = (ant2[1] - ant1[1])
    
    for direction in [-1, 1]:
        curr_row, curr_col = ant1[0], ant1[1]
        
        while 0 <= curr_row < rows and 0 <= curr_col < cols:
            anti_nodes.append((curr_row, curr_col))
            curr_row += direction * delta_row
            curr_col += direction * delta_col

    return list(set(anti_nodes))


def find_anti_nodes(input_map: List[str], frequency: str, antenna_positions: List[Tuple[int, int]]) -> set[Tuple[int, int]]:
    """Finds all anti-nodes for a given frequency"""
    anti_nodes = set()
    anti_nodes2 = set()
    rows, cols = len(input_map), len(input_map[0])
    
    for ant1, ant2 in combinations(antenna_positions, 2):
        print(frequency, ant1, ant2)
        potential_positions = find_anti_node_positions(ant1, ant2)
        potential_positions2 = find_anti_node_positions_part2(ant1, ant2, len(input_map), len(input_map[0]))
        
        for pos in potential_positions:
            if (0 <= pos[0] < rows and 0 <= pos[1] < cols):
                anti_nodes.add(pos)
        
        for pos in potential_positions2:
            anti_nodes2.add(pos)
    
    return anti_nodes, anti_nodes2


def solution(input_map: List[str]) -> int:
    frequency_positions = find_antennas_by_frequency(input_map)
    
    all_anti_nodes = set()
    all_anti_nodes2 = set()
    for frequency, positions in frequency_positions.items():
        if len(positions) >= 2:
            frequency_anti_nodes, frequency_anti_nodes2 = find_anti_nodes(input_map, frequency, positions)
            all_anti_nodes.update(frequency_anti_nodes)
            all_anti_nodes2.update(frequency_anti_nodes2)
    
    return all_anti_nodes, all_anti_nodes2

def graph_antinodes(anti_nodes, input_map):
    for node in anti_nodes:
        input_map[node[0]][node[1]] = "#"

    pretty_print(input_map)

def pretty_print(maze):
    '''Makes a maze easy to see via printing in the terminal
    
    Args:
        maze: 2D list/array representing the maze
        
    Example output:
    ┌───────┐
    │ # # # │
    │ . . . │
    │ # . # │
    └───────┘
    '''
    if not maze or not maze[0]:
        return
    
    width = len(maze[0])
    
    print('┌' + '─' * (width * 2) + '─' + '┐')
    
    for row in maze:
        print('│ ' + ' '.join(str(cell) for cell in row) + ' │')
    
    print('└' + '─' * (width * 2) + '─' + '┘')

def main():
    input_map = parse_input("inputs/day8_input.txt")
    pretty_print(input_map)
    result_a, result_b = solution(input_map)
    print(f"There are {len(result_a)} unique locations within the bounds of the map that contain an antinode")
    print(f"There are {len(result_b)} unique locations within the bounds of the map that contain an antinode after accounting for resonant harmonics.")

    graph_antinodes(result_b, input_map)

if __name__ == "__main__":
    main()