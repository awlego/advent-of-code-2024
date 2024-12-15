from dataclasses import dataclass
from collections import namedtuple

def parse_input(filename):
    warehouse = []
    robot_instructions = ""
    with open(filename, 'r') as openfile:
        wearhouse_input = True
        while wearhouse_input:
            line = openfile.readline()
            if line == ('\n'):
                wearhouse_input = False
            else:
                row = []
                for char in line.strip():
                    row.append(char)
                warehouse.append(row)
        for line in openfile:
            robot_instructions += line.strip()

    return warehouse, robot_instructions


instruction_to_direction_map = {
    "^": (-1, 0),
    ">": (0, 1),
    "v": (1, 0),
    "<": (0, -1)
}


@dataclass
class MapObject():
    moveable: bool


@dataclass
class Wall(MapObject):
    moveable: False
    token = "#"


@dataclass
class Box(MapObject):
    moveable = True
    token = "O"


@dataclass
class Robot(MapObject):
    moveable = True
    token = "@"


@dataclass
class Empty(MapObject):
    moveable = True
    token = "."


def move(state, obj_pos: tuple[int, int], move: tuple[int, int]):
    
    planned_moves = []
    def recursive_checks(state, obj_pos, move):
        r = obj_pos[0] + move[0]
        c = obj_pos[1] + move[1]
        if state[r][c] == Empty.token:
            planned_moves.append((obj_pos, move))
            return True
        elif state[r][c] == Wall.token:
            return False
        else:
            check = recursive_checks(state, (r, c), move)
            if check:
                planned_moves.append((obj_pos, move))
            return check
            
    recursive_checks(state, obj_pos, move)
    
    if planned_moves:
        for (obj_pos, move) in planned_moves:
            r = obj_pos[0] + move[0]
            c = obj_pos[1] + move[1]
            state[r][c] = state[obj_pos[0]][obj_pos[1]]
            state[obj_pos[0]][obj_pos[1]] = Empty.token

    return state


def move_big(state, obj_pos: tuple[int, int], move: tuple[int, int]):
    checked_locs = {}
    planned_moves = []
    temp_moves = []
    depth = 0
    MoveWithDepth = namedtuple('MoveWithDepth', ['move', 'depth'])
    def recursive_checks(state, obj_pos, move, depth=0):
        r = obj_pos[0] + move[0]
        c = obj_pos[1] + move[1]
        if state[r][c] == Empty.token:
            # only add the moves if all the recursive moves don't fail
            temp_moves.append(MoveWithDepth((obj_pos, move), depth))
            return True
        elif state[r][c] == Wall.token:
            return False
        else:
            # we have a box, sometimes we will need to branch
            # we can go normally if we are going left/right.
            if move == (0, 1) or move == (0, -1):
                check = recursive_checks(state, (r, c), move, depth+1)
                if check:
                    planned_moves.append(MoveWithDepth((obj_pos, move), depth))
                return check
            
            # but if we go up down, then we need to branch.
            # if we see the left half of box, then first try to move the right half
            if state[r][c] == "[":
                obj2_pos = (r, c+1) # right half is one to the right
                right_check = recursive_checks(state, obj2_pos, move, depth+1)
                left_check = recursive_checks(state, (r, c), move, depth+1)
            
            elif state[r][c] == "]":
                # if we see the right half of box though, then try to move the left half
                obj2_pos = (r, c-1) # left half is one to the left
                left_check = recursive_checks(state, obj2_pos, move, depth+1)
                right_check = recursive_checks(state, (r, c), move, depth+1)

            else:
                raise ValueError("I never expected to be here")

            if left_check and right_check:
                # if both halves of the box can move, move both halves of the box
                temp_moves.append(MoveWithDepth((obj_pos, move), depth))

            return left_check and right_check
                
    if recursive_checks(state, obj_pos, move):
        planned_moves.extend(temp_moves)
    # deduplicate moves
    planned_moves = list(set(planned_moves))

    print(f"planned_moves:")
    planned_moves.sort(key=lambda x: x.depth, reverse=True)
    for move in planned_moves:
        print(move.move[0], move.depth)

    if planned_moves:
        for (obj_pos, move) in [move.move for move in planned_moves]:
            r = obj_pos[0] + move[0]
            c = obj_pos[1] + move[1]
            state[r][c] = state[obj_pos[0]][obj_pos[1]]
            state[obj_pos[0]][obj_pos[1]] = Empty.token
            # print(f"moved: {(obj_pos, move)}")
            # show_warehouse(state)

    return state


def find_robot(warehouse):
    for row_index, row in enumerate(warehouse):
        for col_index, char in enumerate(row):
            if char == Robot.token:
                return (row_index, col_index)
    raise LookupError("Robot token not found in warehouse")


def show_warehouse(warehouse):
    for line in warehouse:
        print(''.join(line))


def calc_gps(box):
    return 100 * box[0] + box[1]


def sum_gps(warehouse):
    gps_sum = 0
    for row_index, row in enumerate(warehouse):
        for col_index, char in enumerate(row):
            if char == Box.token or char == '[':
                gps_sum += calc_gps((row_index, col_index))
    return gps_sum



def make_wide_map(warehouse):
    new_warehouse = []
    for row in warehouse:
        new_row = []
        for char in row:
            if char == "#":
                new_row.append("#")
                new_row.append("#")
            if char == "O":
                new_row.append("[")
                new_row.append("]")
            if char == ".":
                new_row.append(".")
                new_row.append(".")
            if char == "@":
                new_row.append("@")
                new_row.append(".")
        new_warehouse.append(new_row)
    return new_warehouse


def part_a(warehouse, robot_instructions):
    for instruction in robot_instructions:
        robot_location = find_robot(warehouse)  
        # print(f"Move {instruction}")
        warehouse = move(warehouse, robot_location, instruction_to_direction_map[instruction])
        # show_warehouse(warehouse)
        # print()

    part_a = sum_gps(warehouse)

    print(f"Sum of all boxes' GPS coordinates for part a: {part_a}")
    return warehouse


def part_b(warehouse, robot_instructions):
    warehouse = make_wide_map(warehouse)
    for instruction in robot_instructions:
        robot_location = find_robot(warehouse)  
        print(f"Move {instruction}")
        warehouse = move_big(warehouse, robot_location, instruction_to_direction_map[instruction])
        show_warehouse(warehouse)
        print()

    part_b = sum_gps(warehouse)

    print(f"Sum of all boxes' GPS coordinates for part b: {part_b}")
    return warehouse



def main():
    # warehouse, robot_instructions = parse_input("inputs/day15_input_test.txt")
    # part_a(warehouse, robot_instructions)

    warehouse, robot_instructions = parse_input("inputs/day15_input.txt")
    warehouse = part_b(warehouse, robot_instructions)
    show_warehouse(warehouse)



if __name__ == "__main__":
    main()