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

@dataclass
class Filesystem():
    type: str
    id: int
    size: int
    move_attempted: bool


def uncompress_fs_disk_map(disk_map: List[Filesystem]) -> List[any]:
    '''takes a typed disk map and uncompresses it'''
    uncompressed_disk_map = []
    for fs in disk_map:
        if fs.type == "file":
            uncompressed_disk_map.extend([fs.id for i in range(fs.size)])
        else:
            uncompressed_disk_map.extend(["." for i in range(fs.size)])

    return uncompressed_disk_map

def uncompress_disk_map(disk_map: List[int]) -> List[any]:
    '''takes a disk map and uncompresses it'''
    uncompressed_disk_map = []
    is_file = True # or free space
    cur_id = 0
    for num in disk_map:
        if is_file:
            uncompressed_disk_map.extend([cur_id for i in range(num)])
            cur_id +=1
        else:
            #is empty space
            uncompressed_disk_map.extend(["." for i in range(num)])
        is_file = not is_file

    return uncompressed_disk_map

def compact_filesystem(filesystem: List[any]) -> List[any]:
    '''moves the right most files into empty spaces'''

    # indexes of '.'
    # indexes of numbers
    # for i in range(len(empty_space)):
    #     swap empty spot with last indexes of numbers then pop the index off

    # or keep track of an pointer on each side
    # move left pointer to the right until you find a .
    # when you do, look at right pointer and if it is a number, then swap the elements and update both pointers.
    # if it's not a number, just decrement the right pointer.
    # stop when the left point reaches the right pointer

    left_pointer = 0
    right_pointer = len(filesystem) - 1

    while left_pointer < right_pointer:
        while filesystem[left_pointer] == '.':
            if filesystem[right_pointer] != '.':
                # tuple unpack in place
                filesystem[left_pointer], filesystem[right_pointer] = filesystem[right_pointer], filesystem[left_pointer]
                left_pointer += 1
                right_pointer -= 1
            else:
                right_pointer -= 1
            if right_pointer <= left_pointer:
                return filesystem
        left_pointer += 1
        
    return filesystem


def alt_compact_filesystem(disk_map_t: List[Filesystem]) -> List[Filesystem]:
    '''moves the right most files into empty spaces
    only moves whole files
    only attempts to move each file once'''
    left_pointer = 0
    right_pointer = len(disk_map_t) - 1

    # scan backwards for files we haven't attempted to move
    while right_pointer > 0:
        continue_outer = False

        # keep scanning until we find a file
        if disk_map_t[right_pointer].type != "file" and disk_map_t[right_pointer].move_attempted == False:
            right_pointer -= 1
            # debug_print(disk_map_t, left_pointer, right_pointer)
            continue

        # we have an unseen file, now scan left for spaces
        disk_map_t[right_pointer].move_attempted = True
        while (disk_map_t[left_pointer].type != 'space') or (disk_map_t[left_pointer].size < disk_map_t[right_pointer].size):
            left_pointer += 1
            # if the left pointer reached the end, we couldn't find a space big enough. Just break and continue moving the right pointer
            if left_pointer >= right_pointer:
                continue_outer = True
                left_pointer = 0
                right_pointer -= 1
                # debug_print(disk_map_t, left_pointer, right_pointer)
                break
        
        if continue_outer:
            continue

        # we found a space big enough for a swap
        assert disk_map_t[left_pointer].type == "space"

        # if there is too much space, first split the space.
        if disk_map_t[left_pointer].size > disk_map_t[right_pointer].size:
            size_diff = disk_map_t[left_pointer].size - disk_map_t[right_pointer].size
            disk_map_t.insert(left_pointer+1, Filesystem("space", -1, size_diff, False))
            right_pointer += 1 # adjust to account for the fact that we made the disk map bigger
            disk_map_t[left_pointer].size = disk_map_t[right_pointer].size

        disk_map_t[left_pointer], disk_map_t[right_pointer] = disk_map_t[right_pointer], disk_map_t[left_pointer]
        right_pointer -= 1
        # reset the left pointer so that we have a fresh sweep
        # debug_print(disk_map_t, left_pointer, right_pointer)
        left_pointer = 0
    
    return disk_map_t

def debug_print(disk_map_t: List[Filesystem], left_pointer: int, right_pointer: int) -> None:
    '''prints something like this:
    move_attempted: oo---ooo---x---xxx-xx-xxxx-xxxx-xxx-xxxxxx
    values:        00...111...2...333.44.5555.6666.777.888899
    pointers:         ^       ^                      '''
    
    # Build move_attempted string
    move_str = ""
    for fs in disk_map_t:
        if fs.type == "space":
            move_str += "." * fs.size
        else:
            if fs.move_attempted:
                move_str += "x" * fs.size
            else:
                move_str += "o" * fs.size
    
    # Build values string
    val_str = ""
    for fs in disk_map_t:
        if fs.type == "space":
            val_str += "." * fs.size
        else:
            val_str += str(fs.id) * fs.size
    
    # Build pointer string
    ptr_str = " " * len(val_str)
    
    # Calculate actual positions accounting for sizes
    left_actual = sum(fs.size for fs in disk_map_t[:left_pointer])
    right_actual = sum(fs.size for fs in disk_map_t[:right_pointer])
    
    if left_actual < len(ptr_str):
        ptr_str = ptr_str[:left_actual] + "^" + ptr_str[left_actual + 1:]
    if right_actual < len(ptr_str):
        ptr_str = ptr_str[:right_actual] + "^" + ptr_str[right_actual + 1:]
    
    # Print all strings
    print("move_attempted:", move_str)
    print("values:        ", val_str)
    print("pointers:      ", ptr_str)
    print(left_pointer, right_pointer)


def compute_checksum(compacted_filesystem):
    '''computes the checksum for a given filesystem'''
    checksum = 0
    for i in range(len(compacted_filesystem)):
        if compacted_filesystem[i] == '.':
            return checksum
        checksum += i * compacted_filesystem[i]
    return checksum

def compute_fs_checksum(fs):
    '''computes the checksum for a given filesystem'''
    checksum = 0
    for i in range(len(fs)):
        if fs[i] == '.':
            continue
        checksum += i * fs[i]
    return checksum

def solve_input(filename):
    input = parse_input(filename)
    disk_map = input[0]
    print(disk_map)
    filesystem = uncompress_disk_map(disk_map)
    compacted_filesystem = compact_filesystem(filesystem)
    # print(compacted_filesystem)
    part1_answer = compute_checksum(compacted_filesystem)
    print(f"Answer to part one, the resulting checksum: {part1_answer}")
    return part1_answer

def solve_input_part2(filename):
    input = parse_input(filename)
    disk_map = input[0]
    # print(disk_map)

    disk_map_t = []
    for i in range(len(disk_map)):
        if i % 2 == 0:
            disk_map_t.append(Filesystem("file", i//2, disk_map[i], False))
        else:
            disk_map_t.append(Filesystem("space", -1, disk_map[i], False))
    # for s in disk_map_t:
    #     print(s)
    # print()

    disk_map_t = alt_compact_filesystem(disk_map_t)
    
    # for s in disk_map_t:
    #     print(s)

    filesystem = uncompress_fs_disk_map(disk_map_t)
    # print(filesystem)
    part2_answer = compute_fs_checksum(filesystem)
    print(f"Answer to part two, the resulting checksum: {part2_answer}")
    return part2_answer

def main():
    # test_answer = solve_input("inputs/day9_input_test.txt")
    # assert test_answer == 1928
    # solve_input("inputs/day9_input.txt")
    
    test_answer = solve_input_part2("inputs/day9_input_test.txt")
    assert test_answer == 2858
    solve_input_part2("inputs/day9_input.txt")
    

if __name__ == "__main__":
    main()