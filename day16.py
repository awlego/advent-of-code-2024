from collections import deque
from dataclasses import dataclass
from typing import List, Set, Tuple, Dict, Optional
import heapq
from functools import total_ordering
from pathlib import Path

@dataclass(frozen=True)
@total_ordering
class State:
    """
    Represents a state in the maze navigation problem.
    
    Each state consists of a position (x, y) and a direction of movement.
    The state is immutable and comparable for use in collections.
    
    Attributes:
        x: X-coordinate in the maze
        y: Y-coordinate in the maze
        direction: Tuple of (dx, dy) representing direction of movement
    """
    x: int
    y: int
    direction: Tuple[int, int]

    def __lt__(self, other) -> bool:
        return (self.x, self.y, self.direction) < (other.x, other.y, other.direction)
    
    @property
    def position(self) -> Tuple[int, int]:
        """Returns the current (x, y) position."""
        return (self.x, self.y)


class MazeSolver:
    """
    Solves a maze navigation problem with directional movement constraints.
    
    The maze contains a start point 'S', end point 'E', and walls '#'.
    Movement includes forward steps (cost 1) and turns (cost 1000).
    """
    
    DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # right, down, left, up
    TURN_COST = 1000
    MOVE_COST = 1
    
    def __init__(self, maze: List[str]):
        """
        Initialize the maze solver with a maze layout.

        Args:
            maze: List of strings representing maze rows
        """
        self.maze = maze
        self.width = len(maze[0])
        self.height = len(maze)
        self.start_pos = self._find_position('S')
        self.end_pos = self._find_position('E')
        
        if not all((self.start_pos, self.end_pos)):
            raise ValueError("Maze must contain both 'S' and 'E' positions")

    def _find_position(self, marker: str) -> Optional[Tuple[int, int]]:
        """Finds the position of a given marker in the maze."""
        for y, row in enumerate(self.maze):
            for x, cell in enumerate(row):
                if cell == marker:
                    return (x, y)
        return None
    
    def _is_valid_position(self, x: int, y: int) -> bool:
        """Checks if a position is within bounds and not a wall."""
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                self.maze[y][x] != '#')

    def solve(self) -> int:
        """
        Finds the lowest cost path from start to end.
        
        Returns:
            int: Minimum cost to reach the end, or float('inf') if no path exists
        """
        start_state = State(self.start_pos[0], self.start_pos[1], (1, 0))
        pq = [(0, start_state)]
        visited = set()
        
        while pq:
            cost, state = heapq.heappop(pq)
            
            if state in visited:
                continue
            visited.add(state)
            
            if state.position == self.end_pos:
                return cost
            
            curr_dir_idx = self.DIRECTIONS.index(state.direction)
            
            # Forward movement
            new_x = state.x + state.direction[0]
            new_y = state.y + state.direction[1]
            if self._is_valid_position(new_x, new_y):
                new_state = State(new_x, new_y, state.direction)
                if new_state not in visited:
                    heapq.heappush(pq, (cost + self.MOVE_COST, new_state))
            
            # Turns (left and right)
            for turn in (-1, 1):
                new_dir_idx = (curr_dir_idx + turn) % 4
                new_dir = self.DIRECTIONS[new_dir_idx]
                new_state = State(state.x, state.y, new_dir)
                if new_state not in visited:
                    heapq.heappush(pq, (cost + self.TURN_COST, new_state))
        
        return float('inf')

    def find_optimal_paths(self) -> Tuple[int, Set[Tuple[int, int]]]:
        """
        Finds all tiles that are part of any optimal (lowest-cost) path.
        
        Returns:
            Tuple containing:
            - Minimum cost to reach the end
            - Set of (x, y) coordinates that are part of any optimal path
        """
        start_state = State(self.start_pos[0], self.start_pos[1], (1, 0))
        costs: Dict[State, int] = {start_state: 0}
        pq = [(0, start_state)]
        min_end_cost = float('inf')
        
        # Forward pass to find minimum cost
        while pq:
            cost, state = heapq.heappop(pq)
            
            if cost > costs[state]:
                continue
                
            if state.position == self.end_pos:
                min_end_cost = min(min_end_cost, cost)
                continue
                
            if cost > min_end_cost:
                continue
            
            curr_dir_idx = self.DIRECTIONS.index(state.direction)
            
            # Process moves
            self._process_moves(state, cost, costs, pq, curr_dir_idx)
        
        return min_end_cost, self._backtrack_optimal_paths(costs, min_end_cost)
    
    def _process_moves(self, state: State, cost: int, costs: Dict[State, int], 
                      pq: List, curr_dir_idx: int) -> None:
        """Processes possible moves from current state and updates the priority queue."""
        # Forward movement
        new_x = state.x + state.direction[0]
        new_y = state.y + state.direction[1]
        if self._is_valid_position(new_x, new_y):
            new_state = State(new_x, new_y, state.direction)
            new_cost = cost + self.MOVE_COST
            if new_cost < costs.get(new_state, float('inf')):
                costs[new_state] = new_cost
                heapq.heappush(pq, (new_cost, new_state))
        
        # Turns
        for turn in (-1, 1):
            new_dir_idx = (curr_dir_idx + turn) % 4
            new_dir = self.DIRECTIONS[new_dir_idx]
            new_state = State(state.x, state.y, new_dir)
            new_cost = cost + self.TURN_COST
            if new_cost < costs.get(new_state, float('inf')):
                costs[new_state] = new_cost
                heapq.heappush(pq, (new_cost, new_state))

    def _backtrack_optimal_paths(self, costs: Dict[State, int], 
                               min_end_cost: int) -> Set[Tuple[int, int]]:
        """Backtrack from end states to find all tiles in optimal paths."""
        optimal_tiles = set()
        end_states = [(state, cost) for state, cost in costs.items() 
                     if state.position == self.end_pos and cost == min_end_cost]
        
        stack = deque(end_states)
        visited = set()
        
        while stack:
            state, cost = stack.popleft()
            if (state, cost) in visited:
                continue
            visited.add((state, cost))
            
            optimal_tiles.add(state.position)
            
            if state.position == self.start_pos:
                continue
            
            self._add_previous_states(state, cost, costs, stack)
        
        return optimal_tiles
    
    def _add_previous_states(self, state: State, cost: int, 
                           costs: Dict[State, int], stack: deque) -> None:
        """Adds valid previous states to the processing stack."""
        for prev_dir in self.DIRECTIONS:
            prev_x = state.x - state.direction[0]
            prev_y = state.y - state.direction[1]
            
            if self._is_valid_position(prev_x, prev_y):
                turn_cost = self.TURN_COST if prev_dir != state.direction else 0
                prev_state = State(prev_x, prev_y, prev_dir)
                expected_cost = cost - self.MOVE_COST - turn_cost
                
                if costs.get(prev_state, float('inf')) == expected_cost:
                    stack.append((prev_state, expected_cost))

    def visualize_path(self, optimal_tiles: Set[Tuple[int, int]]) -> List[str]:
        """
        Creates a visual representation of the maze with optimal paths marked.
        
        Args:
            optimal_tiles: Set of (x, y) coordinates in optimal paths
            
        Returns:
            List of strings representing the visualization
        """
        return [''.join('#' if cell == '#' else 
                       'O' if (x, y) in optimal_tiles else '.'
                       for x, cell in enumerate(row))
                for y, row in enumerate(self.maze)]


def parse_input(filepath: str) -> List[str]:
    """
    Reads and parses a maze from a file.
    
    Args:
        filepath: Path to the input file
        
    Returns:
        List of strings representing maze rows
    """
    return Path(filepath).read_text().splitlines()


def main():
    """Main execution function."""
    maze = parse_input("inputs/day16_input.txt")
    solver = MazeSolver(maze)
    
    # Find minimum cost path
    min_cost = solver.solve()
    print(f"Lowest possible score: {min_cost}")
    
    # Find and visualize all optimal paths
    min_cost, optimal_tiles = solver.find_optimal_paths()
    print(f"Number of tiles in optimal paths: {len(optimal_tiles)}")
    
    result_maze = solver.visualize_path(optimal_tiles)
    for row in result_maze:
        print(row)


if __name__ == "__main__":
    main()