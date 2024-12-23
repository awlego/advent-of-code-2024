from tqdm import tqdm
from collections import defaultdict
from typing import List, Tuple
import heapq

def parse_racetrack(filepath):
    """
    Parses a racetrack/maze file and returns:
    - A 2D grid representation
    - Start coordinates
    - End coordinates
    - Set of track coordinates (including start and end)
    
    Args:
        filepath (str): Path to the input file
        
    Returns:
        tuple: (grid, start_pos, end_pos, track_positions)
    """
    # Read the file
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    # Create the grid
    grid = [list(line) for line in lines]
    height = len(grid)
    width = len(grid[0])
    
    # Find start and end positions
    start_pos = None
    end_pos = None
    track_positions = set()
    
    for y in range(height):
        for x in range(width):
            if grid[y][x] == 'S':
                start_pos = (x, y)
                track_positions.add((x, y))
            elif grid[y][x] == 'E':
                end_pos = (x, y)
                track_positions.add((x, y))
            elif grid[y][x] == '.':
                track_positions.add((x, y))
    
    if start_pos is None or end_pos is None:
        raise ValueError("Start (S) or End (E) position not found in the grid")
    
    return grid, start_pos, end_pos, track_positions

def get_valid_neighbors(pos, track_positions, width, height):
    """
    Returns valid neighboring positions that are part of the track.
    
    Args:
        pos (tuple): Current (x, y) position
        track_positions (set): Set of valid track positions
        
    Returns:
        list: tuple of (list of valid neighboring (x, y) positions, list of all neighboring (x, y) positions)
    """
    assert type(pos) == tuple
    x, y = pos
    neighbors = [
        (x+1, y), (x-1, y),  # right, left
        (x, y+1), (x, y-1)   # down, up
    ]
    # check within bounds
    neighbors = [n for n in neighbors if 0 <= n[0] < width and 0 <= n[1] < height]
    return ([n for n in neighbors if n in track_positions], neighbors)


# def run_race(start, end, track):
#     """
#     Follows the track and returns the time it takes to finish the race using BFS.
    
#     Args:
#         start (tuple): Starting (x, y) position
#         end (tuple): Ending (x, y) position
#         track (set): Set of valid track positions
        
#     Returns:
#         int: Number of steps needed to reach the end, or -1 if no path exists
#     """
#     from collections import deque
    
#     queue = deque([(start, 0)])
#     visited = {start}
    
#     while queue:
#         current_pos, steps = queue.popleft()
        
#         if current_pos == end:
#             return steps
        
#         valid_neighbors, _ = get_valid_neighbors(current_pos, track)
#         for next_pos in valid_neighbors:
#             if next_pos not in visited:
#                 visited.add(next_pos)
#                 queue.append((next_pos, steps + 1))
    
#     return -1

def generate_cheats2(loc, track, width, height):
    '''returns a list of new tracks all possible cheats at location loc applied'''
    # assumes you get 2 cheat moves
    new_tracks = {}
    _, all_moves = get_valid_neighbors(loc, track, width, height)
    cheat_start = loc
    cheat_moves = [mov for mov in all_moves if mov not in valid_moves]
    for move in cheat_moves:
        temp_track = track.copy()
        if move not in temp_track:
            temp_track.add(move)
            _, second_moves = get_valid_neighbors(move, temp_track, width, height)
            for second_move in second_moves:
                cheat_end = second_move
                if second_move not in temp_track:
                    temp_track.add(second_move)
                    new_tracks[(cheat_start, cheat_end)] = temp_track

    return new_tracks

def generate_cheats(loc, track, width, height):
    '''returns a list of new tracks all possible cheats at location loc applied'''
    # assumes you get 1 cheat moves
    new_tracks = {}
    valid_moves, all_moves = get_valid_neighbors(loc, track, width, height)
    cheat_start = loc
    cheat_moves = [mov for mov in all_moves if mov not in valid_moves]
    one = loc
    for move in cheat_moves:
        temp_track = track.copy()
        if move not in temp_track:
            temp_track.add(move)
            # visualize_track(temp_track, (1, 3), (5, 7))
            two = move
            valid_second_moves, _ = get_valid_neighbors(move, temp_track, width, height)
            valid_second_moves = [m for m in valid_second_moves if m != cheat_start]
            # valid_second_moves = [m for m in valid_second_moves if m in track]
            for second_move in valid_second_moves:
                cheat_end = second_move
                three = second_move
                new_tracks[(cheat_start, cheat_end)] = temp_track

                # print(one, two, three)
    return new_tracks


def remove_loops(track: set[tuple[int, int]], start: tuple[int, int], end: tuple[int, int], width, height) -> set[tuple[int, int]]:
    """Removes loops from the track by finding the shortest path between start and end points.
    
    Args:
        track: Set of (x, y) coordinates representing the track
        start: Starting (x, y) coordinate
        end: Ending (x, y) coordinate
        
    Returns:
        Set of coordinates representing the shortest path
        
    Raises:
        ValueError: If no path exists between start and end
    """
    if not track:
        return set()
    
    # Create adjacency list
    graph = defaultdict(list)
    for pos in track:
        valid_neighbors, _ = get_valid_neighbors(pos, track, width, height)
        for neighbor in valid_neighbors:
            graph[pos].append((neighbor, 1))
    
    # Use Dijkstra's algorithm to find shortest path
    distances = {pos: float('inf') for pos in track}
    distances[start] = 0
    pq = [(0, start)]
    previous = {pos: None for pos in track}
    while pq:
        curr_dist, curr_pos = heapq.heappop(pq)

        if curr_pos == end:
            break
        if curr_dist > distances[curr_pos]:
            continue
            
        for next_pos, weight in graph[curr_pos]:
            distance = curr_dist + weight
            if distance < distances[next_pos]:
                distances[next_pos] = distance
                previous[next_pos] = curr_pos
                heapq.heappush(pq, (distance, next_pos))
    
    # Check if end is reachable
    if distances[end] == float('inf'):
        return set()
    
    # Reconstruct path
    path = set()
    curr = end
    while curr is not None:
        path.add(curr)
        curr = previous[curr]
    
    # Validate path contains both start and end
    if start not in path or end not in path:
        raise ValueError("Path reconstruction failed")
    
    return path


def fast_run(track, start, end):
    # print()
    # print("before removing loops")
    # visualize_track(track, (1, 3), (5, 7))
    track = remove_loops(track, start, end, width, height)
    # print("after removing loops")
    # visualize_track(track, (1, 3), (5, 7))
    return len(track) - 1


def cheat(track, start, end):
    times = {}
    
    for loc in tqdm(track, desc="Generating cheats"):
        cheat_tracks = generate_cheats(loc, track, width, height)        
        for endpoint, cheat_track in cheat_tracks.items():
            reversed_endpoint = (endpoint[1], endpoint[0])  # Create reversed tuple
            if endpoint in times or reversed_endpoint in times or endpoint == reversed_endpoint:
                continue
            times[endpoint] = fast_run(cheat_track, start, end)
    
    return times

def visualize_track(track, start, end):
    '''visualizes the track from a set of coordinates'''
    max_x = max(x for x, y in track)
    max_y = max(y for x, y in track)
    
    for y in range(max_y + 1):
        for x in range(max_x + 1):
            if (x, y) == start:
                print('S', end='')
            elif (x, y) == end:
                print('E', end='')
            elif (x, y) in track:
                print('O', end='')
            else:
                print(' ', end='')
        print()


def debug_2_picoseconds(cheat_times, normal_time):
    '''There are 42 locations that save 2 picoseconds, I want to find them'''
    # let's print the location of all of the locations that save 2 picoseconds
    i = 0
    for endpoint, time in cheat_times.items():
        if time == normal_time - 2:
            print(endpoint) 
            i += 1
    print(f"There are {i} locations that save 2 picoseconds")



if __name__ == "__main__":
    filepath = "inputs/day20_input_test.txt"
    grid, start, end, track = parse_racetrack(filepath)
    width = len(grid[0])
    height = len(grid)
    
    print(f"Grid dimensions: {width}x{height}")
    print(f"Start position: {start}")
    print(f"End position: {end}")
    print(f"Number of track positions: {len(track)}")
    
    valid_moves = get_valid_neighbors(start, track, width, height)
    print(f"Valid moves from start: {valid_moves}")

    normal_time = fast_run(track, start, end)

    print(f"Time to finish the race: {normal_time}")

    cheat_times = cheat(track, start, end)
    print(f"Minimum time to finish the race: {min(cheat_times.values())}")  

    cheat_n_times = defaultdict(lambda: 0)
    for time in sorted(cheat_times.values()):
        cheat_n_times[normal_time - time] += 1
    for key, value in sorted(cheat_n_times.items(), reverse=False):
        print(f'There are {value} locations that save {key} picoseconds')

    print(f"Number of cheat locations that save at least 100 picoseconds: {len([time for time in cheat_times.values() if time < normal_time - 100])}")

    # best_endpoints = [endpoint for endpoint, time in cheat_times.items() if time <= normal_time - 100]
    # for e in best_endpoints:
    #     print(e)

    # 1365 too low

