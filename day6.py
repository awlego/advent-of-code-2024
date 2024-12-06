from typing import List, Tuple
from enum import Enum
from line_profiler import profile

class Direction(Enum):
    # let's put the origin in the upper left
    # that way we can read it like an array 
    # (row, col)
    UP = ("^", (-1, 0))
    RIGHT = (">", (0, 1))
    DOWN = ("v", (1, 0))
    LEFT = ("<", (0, -1))

    def __init__(self, symbol, delta):
        self.symbol = symbol
        self.delta = delta

    @classmethod
    def from_symbol(cls, symbol: str) -> "Direction":
        for direction in cls:
            if direction.symbol == symbol:
                return direction
        raise ValueError(f"Invalid direction symbol: {symbol}")

    def turn_right(self) -> "Direction":
        """turns right 90 degrees"""
        directions = list(Direction)
        current_index = directions.index(self)
        return directions[(current_index + 1) % 4]

def profile_methods(cls):
    for attr in cls.__dict__:
        if callable(getattr(cls, attr)) and not attr.startswith("__"):
            setattr(cls, attr, profile(getattr(cls, attr)))
    return cls

@profile_methods
class Guard:
    def __init__(self, maze: List[str]) -> None:
        self.direction = Direction.UP  # or whatever starting direction
        self.position = self.find_guard_location(maze)
        self.history = set()

    def find_guard_location(self, maze: List[str]) -> Tuple[int, int]:
        '''returns the row and column of the guard's location'''
        guard_chars = [d.symbol for d in Direction]
        
        for row_index, row in enumerate(maze):
            for col_index, col in enumerate(row):
                if col in guard_chars:
                    return (row_index, col_index)

    def turn_right(self) -> None:
        self.direction = self.direction.turn_right()

    def turn_left(self) -> None:
        self.direction = self.direction.turn_left()

    def get_next_position(self) -> Tuple[int, int]:
        dr, dc = self.direction.delta
        return (self.position[0] + dr, self.position[1] + dc)
    
    def move(self, maze):
        '''move in the current direction and update the maze'''
        old_row, old_col = self.position
        self.position = self.get_next_position()
        maze[self.position[0]][self.position[1]] = self.direction.symbol
        maze[old_row][old_col] = "X"
        self.history.add((self.position, self.direction))
        return maze
    
    def check_in_loop(self):
        if (self.get_next_position(), self.direction) in self.history:
            return True
        return False
    
    def timestep(self, maze):
        dr, dc = self.direction.delta
        next_row = self.position[0] + dr
        next_col = self.position[1] + dc
        
        # Check bounds first
        if not (0 <= next_row < len(maze) and 0 <= next_col < len(maze[0])):
            # print("going out of bounds")
            return maze, True
        
        next_cell = maze[next_row][next_col]
        if next_cell not in [".", "X"]:
            self.turn_right()
        else:
            maze = self.move(maze)
        return maze, False
    
@profile
def parse_input(filepath: str) -> List[str]:
    maze = []
    with open(filepath, 'r') as open_file:
        for line in open_file:
            maze.append([char for char in line.strip()])
    return maze

@profile
def count_visited_spots(maze):
    count = 1 # start at one since the current location also counts
    for row in maze:
        for col in row:
            if col == "X":
                count += 1
    return count

@profile
def loop_detection(maze, maze_history):
    ''''Finds if a loop has happened.'''
    if maze in maze_history:
        return True
    else:
        maze_history.add(maze)
    return False

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
    
    # Print top border
    print('┌' + '─' * (width * 2 - 1) + '┐')
    
    # Print maze contents with side borders
    for row in maze:
        print('│ ' + ' '.join(str(cell) for cell in row) + ' │')
    
    # Print bottom border
    print('└' + '─' * (width * 2 - 1) + '┘')

def part1():
    maze = parse_input('inputs/day6_input.txt')
    pretty_print(maze)
    guard = Guard(maze)
    print(guard.find_guard_location(maze))

    done = False
    while not done:
        maze, done = guard.timestep(maze)

    pretty_print(maze)
    num_visited_spots = count_visited_spots(maze)
    total_spaces = len(maze) * len(maze[0])
    print(f"number of spots guard visits: {num_visited_spots} out of {total_spaces} total spots")

@profile
def check_for_loops(maze):
    guard = Guard(maze)
    done = False
    found_loop = False
    while not done:
        maze, done = guard.timestep(maze)
        found_loop = guard.check_in_loop()
        if found_loop:
            # print("Found loop!")
            done = True

    # pretty_print(maze)
    return found_loop

@profile
def part2():
    # it's not fast but it works
    input_file_name = 'inputs/day6_input.txt'
    maze = parse_input(input_file_name)
    loop_count = 0
    for row_index, row in enumerate(maze):
        for col_index, col in enumerate(row):
            if maze[row_index][col_index] == ".":
                maze[row_index][col_index] = "O"
                if check_for_loops(maze):
                    loop_count += 1
                # reset the maze
                maze = parse_input(input_file_name)
    print(loop_count)

@profile
def main():
    # part1()
    part2()


if __name__ == "__main__":
    main()


# how might I optimize part 2?
# 1. save the input text with a deepcopy so I can restore from memory instead of reconstructing from disk every time
# 2. memoize the timesteps?
# 3. run on multiple processes
# 4. instead of taking one step at a time per timestamp, "slide" the full distance that the guard would go.
# 5 only check putting blocks on squares that we know the guard will visit (via part 1)