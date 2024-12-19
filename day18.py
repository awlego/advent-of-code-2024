from heapq import heappush, heappop
from typing import List, Tuple, Dict, Set

def parse_coordinates(filename, timestamp=None):
    # Read coordinates from file
    coordinates = []
    with open(filename, 'r') as file:
        for line in file:
            if ',' in line:
                x, y = map(int, line.strip().split(','))
                coordinates.append((x, y))
    
    # If timestamp is provided, return the coordinate at that timestamp
    if timestamp is not None:
        if 0 <= timestamp < len(coordinates):
            return coordinates[timestamp]
        else:
            return None
            
    # Otherwise return all coordinates with their timestamps
    return list(enumerate(coordinates))


def create_grid(blocked_cells: List[Tuple[int, int]] = None, size: int = 6) -> List[List[str]]:
    """Create a grid of specified size with optional blocked cells.
    
    Args:
        blocked_cells: List of (x,y) coordinates for blocked cells
        size: Size of the grid (creates a size x size grid). Defaults to 6
        
    Returns:
        List[List[str]]: Grid where '.' represents empty cells and '#' represents blocked cells
    """
    grid = [['.' for _ in range(size)] for _ in range(size)]
    if blocked_cells:
        for x, y in blocked_cells:
            if 0 <= x < size and 0 <= y < size:
                grid[y][x] = '#'
    return grid


def print_grid(grid: List[List[str]], path: List[Tuple[int, int]] = None):
    """Print the grid with optional path marked with '*'."""
    if path:
        # Create a copy of the grid to avoid modifying the original
        temp_grid = [row[:] for row in grid]
        for x, y in path:
            temp_grid[y][x] = '*'
        grid = temp_grid
    
    for row in grid:
        print(' '.join(row))

def get_neighbors(x: int, y: int, grid: List[List[str]]) -> List[Tuple[int, int]]:
    """Get valid neighboring cells (up, down, left, right)."""
    neighbors = []
    size = len(grid)  # Assuming square grid
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        new_x, new_y = x + dx, y + dy
        if (0 <= new_x < size and 0 <= new_y < size and 
            grid[new_y][new_x] != '#'):
            neighbors.append((new_x, new_y))
    return neighbors

def dijkstra(grid: List[List[str]], start: Tuple[int, int], 
             end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Find shortest path using Dijkstra's algorithm."""
    distances: Dict[Tuple[int, int], float] = {start: 0}
    previous: Dict[Tuple[int, int], Tuple[int, int]] = {}
    pq = [(0, start)]
    visited: Set[Tuple[int, int]] = set()

    while pq:
        current_dist, current = heappop(pq)
        
        if current in visited:
            continue
            
        visited.add(current)
        
        if current == end:
            # Reconstruct path
            path = []
            while current in previous:
                path.append(current)
                current = previous[current]
            path.append(start)
            return path[::-1]
        
        for next_cell in get_neighbors(*current, grid):
            if next_cell in visited:
                continue
                
            distance = current_dist + 1
            
            if distance < distances.get(next_cell, float('inf')):
                distances[next_cell] = distance
                previous[next_cell] = current
                heappush(pq, (distance, next_cell))
    
    return []  # No path found

def main():
    falling_positions = parse_coordinates("inputs/day18_input.txt")

    blocked_cells = [pos for (t, pos) in falling_positions if t<1024]
    # print(blocked_cells)
    # blocked_cells = []
    size = 70
    grid = create_grid(blocked_cells, size+1)
    print(len(grid))
    
    start = (0, 0)  # Top-left corner
    end = (size, size)    # Bottom-right corner
    
    print("Initial grid:")
    print_grid(grid)
    print("\nFinding path from", start, "to", end)
    
    path = dijkstra(grid, start, end)
    
    if path:
        print("\nPath found! Grid with path marked as '*':")
        print_grid(grid, path)
        # print("Path:", path)
    else:
        print("\nNo path found!")

    print(len(path) - 1) # minus 1 since we start on the grid and our pathfinding algorithm says that's the first step

    # part 2
    blocked_cells = []
    grid = create_grid(blocked_cells, size+1)
    path = dijkstra(grid, start, end)
    t = 0
    print(path)

    print(falling_positions[0][1])

    while path:
        blocked_cells.append(falling_positions[t][1])
        grid = create_grid(blocked_cells, size+1)
        path = dijkstra(grid, start, end)
        print_grid(grid, path)
        print()
        if path:
            t += 1

    print(t)
    print(falling_positions[t][1])
    

if __name__ == "__main__":
    main()
