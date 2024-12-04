from typing import Tuple

def parse_input(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(line.strip())

    return data


def main():
    input_data = parse_input('inputs/day4_input.txt')
    padded_input_data = pad(input_data)

    for line in padded_input_data:
        print(line)    
    
    count = 0
    num_places_checked = 0
    for row_index in range(len(padded_input_data)):
       for col_index in range(len(padded_input_data[row_index])):
            if padded_input_data[row_index][col_index] != ".":
                count += check_directions(padded_input_data, (row_index, col_index))
                num_places_checked += 1

    print(f"found {count} instances of 'XMAS' in the puzzle")
    print(f"Checked {num_places_checked} origins")
    
    count = 0
    num_places_checked = 0
    for row_index in range(len(padded_input_data)):
       for col_index in range(len(padded_input_data[row_index])):
            if padded_input_data[row_index][col_index] != ".":
                count += check_x_mas_directions(padded_input_data, (row_index, col_index))
                num_places_checked += 1
    
    for line in padded_input_data:
        print(line)  
    print(f"found {count} instances of 'X-MAS' in the puzzle")


def pad(input_data, amount=3):
    """Adds amount rows/columns around the text so that all valid 'XMAS' searches can be done using the edges as origin points"""
    padded_input_data = []
    length = 0
    for row in input_data:
        for a in range(amount):
            row = "." + row
            row = row + "."
        padded_input_data.append(row)
        length = len(row)

    for a in range(amount):
        padded_input_data.insert(0, "."*length)
        padded_input_data.append("."*length)
    return padded_input_data

def check_x_mas_directions(puzzle, origin: Tuple[int, int]) -> int:
    count = 0
    count += check_x_mas_down_right(puzzle, origin)
    # count += check_x_mas_down_left(puzzle, origin)
    return count


def check_x_mas_down_right(puzzle, origin: Tuple[int, int]) -> int:
    count = 0
    (row_index, col_index) = origin
    word = puzzle[row_index][col_index]
    word += puzzle[row_index+1][col_index+1]
    word += puzzle[row_index+2][col_index+2]
    word2 = puzzle[row_index][col_index+2]
    word2 += puzzle[row_index+1][col_index+1]
    word2 += puzzle[row_index+2][col_index]

    if word == 'MAS' or word == 'MAS'[::-1]:
        count += 1
    if word2 == 'MAS' or word2 == 'MAS'[::-1]:
        count += 1 
    # print(count)
    # print()
    # for i in range(3):
    #     print(puzzle[row_index+i][col_index:col_index+3])
    # print(word, word2)
    # print(word == 'MAS', word == 'MAS'[::-1], word2 == 'MAS', word2 == 'MAS'[::-1])
    # if False or True:
    #     print("test")

    if count == 2:
        # puzzle[row_index] = puzzle[row_index][:col_index] + "O" + puzzle[row_index][col_index + 1:]
        # print("YES!")
        return 1
    return 0
    

def check_x_mas_down_left(puzzle, origin: Tuple[int, int]) -> int:
    count = 0
    (row_index, col_index) = origin

    word = puzzle[row_index][col_index]
    word += puzzle[row_index+1][col_index-1]
    word += puzzle[row_index+2][col_index-2]
    word2 = puzzle[row_index][col_index+2]
    word2 += puzzle[row_index+1][col_index+1]
    word2 += puzzle[row_index+2][col_index]

    if word == 'MAS' or word == 'MAS'[::-1]:
        count += 1
    if word2 == 'MAS' or word == 'MAS'[::-1]:
        count += 1 

    if count == 2:
        return 1
    return 0


def check_directions(puzzle, origin: Tuple[int, int]) -> int:
    count = 0
    count += check_right(puzzle, origin)
    count += check_down(puzzle, origin)
    count += check_right_down_diagonal(puzzle, origin)
    count += check_left_down_diagonal(puzzle, origin)
    return count


def check_right(puzzle, origin: Tuple[int, int]) -> int:
    "Returns 0 or 1"
    count = 0
    (row_index, col_index) = origin
    if puzzle[row_index][col_index:col_index+4] == 'XMAS' or puzzle[row_index][col_index:col_index+4] == 'XMAS'[::-1]:
        count += 1

    return count


def check_down(puzzle, origin: Tuple[int, int]) -> int:
    "Returns 0 or 1"
    count = 0
    (row_index, col_index) = origin
    word = puzzle[row_index][col_index]
    word += puzzle[row_index+1][col_index]
    word += puzzle[row_index+2][col_index]
    word += puzzle[row_index+3][col_index]
    if word == 'XMAS' or word == 'XMAS'[::-1]:
        count += 1
    return count


def check_right_down_diagonal(puzzle, origin: Tuple[int, int]) -> int:
    "Returns 0 or 1"
    count = 0
    (row_index, col_index) = origin
    word = puzzle[row_index][col_index]
    word += puzzle[row_index+1][col_index+1]
    word += puzzle[row_index+2][col_index+2]
    word += puzzle[row_index+3][col_index+3]
    if word == 'XMAS' or word == 'XMAS'[::-1]:
        count += 1
    return count


def check_left_down_diagonal(puzzle, origin: Tuple[int, int]) -> int:
    "Returns 0 or 1"
    count = 0
    (row_index, col_index) = origin
    word = puzzle[row_index][col_index]
    word += puzzle[row_index+1][col_index-1]
    word += puzzle[row_index+2][col_index-2]
    word += puzzle[row_index+3][col_index-3]
    if word == 'XMAS' or word == 'XMAS'[::-1]:
        count += 1
    return count

 

if __name__ == "__main__":
    main()
    


# I could search each valid direction, doing a scan for valid answers


# o o o o

# o
# o
# o
# o

# o
#   o
#     o
#       o

#       o
#     o
#   o
# o

# These are the 4 valid shapes. If any of them contain "xmas" or "samx" then we can increment our valid word count.
# How do we make sure we don't double count? Don't look left or up!

# --

# okay, if my answer is too big, then I'm either double counting or counting things that shouldn't be counted