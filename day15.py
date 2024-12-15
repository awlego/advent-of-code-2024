from dataclasses import dataclass

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
        # print(f"Checking: obj_pos: {obj_pos}, move: {move}")
        # state[r][c]
        r = obj_pos[0] + move[0]
        c = obj_pos[1] + move[1]
        if state[r][c] == Empty.token:
            planned_moves.append((obj_pos, move))
            # print("found empty spot")
            # print(planned_moves)
            return True
        elif state[r][c] == Wall.token:
            # print("found wall")
            return False
        else:
            # print("found a box")
            check = recursive_checks(state, (r, c), move)
            if check:
                planned_moves.append((obj_pos, move))
                # print(planned_moves)
            return check
            

    recursive_checks(state, obj_pos, move)
    # print(f"planned_moves: {planned_moves}")
    
    # make sure planned moves are in the order that they need to happen in.
    if planned_moves:
        for (obj_pos, move) in planned_moves:
            # swap in place
            # state[obj_pos[0]][obj_pos[1]], state[obj_pos[0] + move[0]][obj_pos[1] + move[1]] = \
            # state[obj_pos[0] + move[0]][obj_pos[1] + move[1]], state[obj_pos[0]][obj_pos[1]]
            r = obj_pos[0] + move[0]
            c = obj_pos[1] + move[1]
            state[r][c] = state[obj_pos[0]][obj_pos[1]]
            state[obj_pos[0]][obj_pos[1]] = Empty.token

    return state


def find_robot(warehouse):
    for row_index, row in enumerate(warehouse):
        for col_index, char in enumerate(row):
            if char == Robot.token:
                return (row_index, col_index)
    raise LookupError("Robot token not found in warehouse")


# need to do recursive checking before applying the recursive action
shoves = []

def show_warehouse(warehouse):
    for line in warehouse:
        print(''.join(line))


def calc_gps(box):
    return 100 * box[0] + box[1]

def main():
    warehouse, robot_instructions = parse_input("inputs/day15_input.txt")
    # print(f"warehouse: {warehouse}, robot_instructions: {robot_instructions}")

    for instruction in robot_instructions:
        robot_location = find_robot(warehouse)  
        print(f"Move {instruction}")
        warehouse = move(warehouse, robot_location, instruction_to_direction_map[instruction])
        show_warehouse(warehouse)
        print()

    part_a = 0
    for row_index, row in enumerate(warehouse):
        for col_index, char in enumerate(row):
            if char == Box.token:
                part_a += calc_gps((row_index, col_index))

    print(f"Sum of all boxes' GPS coordinates for part a: {part_a}")

if __name__ == "__main__":
    main()