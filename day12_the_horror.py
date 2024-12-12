from typing import List, Tuple, Dict, Set
from functools import wraps
from collections import defaultdict
from dataclasses import dataclass
import textwrap

def parse_input(filename: str) -> str:
    input = []
    with open(filename, 'r') as openfile:
        for line in openfile:
            line = line.strip()
            input.append(line)
    return input

@dataclass
class Region():
    plant: str
    area: int
    perimeter: int
    sides: int  # New field for number of sides

@dataclass
class ExpectedRegion:
    area: int
    perimeter: int
    sides: int  # New field
    expected_price: int
    expected_discount_price: int  # New field for price calculated with sides

def get_neighbors(pos: Tuple[int, int], rows: int, cols: int) -> List[Tuple[int, int]]:
    """Return valid neighboring positions."""
    i, j = pos
    neighbors = []
    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        ni, nj = i + di, j + dj
        if 0 <= ni < rows and 0 <= nj < cols:
            neighbors.append((ni, nj))
    return neighbors


def flood_fill(garden_map: List[List[str]], start: Tuple[int, int], 
               visited: Set[Tuple[int, int]]) -> Tuple[Set[Tuple[int, int]], int]:
    """
    Flood fill from start position to find all connected cells of the same type.
    Returns the set of positions in the region, its perimeter, and number of sides.
    """
    rows, cols = len(garden_map), len(garden_map[0])
    plant = garden_map[start[0]][start[1]]
    region_cells = set()
    perimeter = 0
    stack = [start]
    
    while stack:
        current = stack.pop()
        if current in region_cells:
            continue
            
        region_cells.add(current)
        visited.add(current)
        
        # Check neighbors for same plant type and count perimeter
        for neighbor in get_neighbors(current, rows, cols):
            if garden_map[neighbor[0]][neighbor[1]] == plant:
                if neighbor not in region_cells:
                    stack.append(neighbor)
            else:
                perimeter += 1
                
        # Add perimeter for edges
        i, j = current
        if i == 0: perimeter += 1  # top edge
        if i == rows-1: perimeter += 1  # bottom edge
        if j == 0: perimeter += 1  # left edge
        if j == cols-1: perimeter += 1  # right edge
        
    return region_cells, perimeter

def calculate_regions(garden_map: List[List[str]]) -> Dict[str, Region]:
    """Calculate separate regions for each contiguous group of plants."""
    if not garden_map or not garden_map[0]:
        return {}
    
    rows, cols = len(garden_map), len(garden_map[0])
    regions = {}
    visited = set()
    region_count = defaultdict(int)
    
    # Find all distinct regions
    for i in range(rows):
        for j in range(cols):
            if (i, j) not in visited:
                plant = garden_map[i][j]
                region_cells, perimeter = flood_fill(garden_map, (i, j), visited)
                border_map = create_border_map(region_cells, rows, cols)
                sides = count_sides(border_map)
                # Create unique identifier for this region
                region_count[plant] += 1
                # For clarity in test comparisons, first region has no number, second has "1", etc
                region_id = f"{plant}{region_count[plant]-1}" if region_count[plant] > 1 else plant
                
                regions[region_id] = Region(
                    plant=plant,
                    area=len(region_cells),
                    perimeter=perimeter,
                    sides=sides
                )
    
    return regions

def calculate_total_score(regions: Dict[str, Region], use_sides: bool = False) -> int:
    if use_sides:
        return sum(region.area * region.sides for region in regions.values())
    return sum(region.area * region.perimeter for region in regions.values())


def create_border_map(region_cells: Set[Tuple[int, int]], rows: int, cols: int) -> List[List[str]]:
    """Create a border map with + - | characters around the region."""
    # Create a larger map to fit borders (2x + 1 size)
    border_map = [[' ' for _ in range(2 * cols + 1)] for _ in range(2 * rows + 1)]
    
    # First pass: Fill in the borders
    for i, j in region_cells:
        # Convert region coordinates to border map coordinates
        bi, bj = 2 * i + 1, 2 * j + 1
        
        # Mark cell position
        border_map[bi][bj] = '#'
        
        # Check and mark borders with neighbors
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for di, dj in neighbors:
            ni, nj = i + di, j + dj
            if (ni, nj) not in region_cells:
                # Add border elements
                border_i = bi + di
                border_j = bj + dj
                if di == 0:  # Vertical border
                    border_map[bi][border_j] = '|'
                else:  # Horizontal border
                    border_map[border_i][bj] = '-'
    
    for row in border_map:
        print(''.join(row))



    # Second pass: flood fill looking for unique "|" or "-" regions
    def flood_fill_border(start_i: int, start_j: int, visited: Set[Tuple[int, int]]) -> bool:
        """
        Flood fills a single border segment.
        Returns True if this is a new valid border segment.
        """
        if (start_i, start_j) in visited:
            return False
            
        char = border_map[start_i][start_j]
        if char not in '-|':
            return False
            
        # Remember this shape of border we're filling
        is_vertical = char == '|'
        
        # Flood fill this segment
        stack = [(start_i, start_j)]
        segment = set()
        
        while stack:
            i, j = stack.pop()
            if (i, j) in visited:
                continue
                
            current = border_map[i][j]
            if current != char:
                continue
                
            visited.add((i, j))
            segment.add((i, j))
            
            # Add neighbors in the same direction
            neighbors = [(0, 1), (0, -1)] if is_vertical else [(1, 0), (-1, 0)]
            for di, dj in neighbors:
                ni, nj = i + di, j + dj
                if 0 <= ni < len(border_map) and 0 <= nj < len(border_map[0]):
                    if border_map[ni][nj] == char:
                        stack.append((ni, nj))
        
        return True
    
    visited = set()
    unique_sides = 0
    
    for i in range(len(border_map)):
        for j in range(len(border_map[0])):
            print(visited)
            print(unique_sides)
            if flood_fill_border(i, j, visited):
                unique_sides += 1


    return border_map, unique_sides


def test_garden_regions():
    input_text = textwrap.dedent("""
    RRRRIICCFF
    RRRRIICCCF
    VVRRRCCFFF
    VVRCCCJFFF
    VVVVCJJCFE
    VVIVCCJJEE
    VVIIICJJEE
    MIIIIIJJEE
    MIIISIJEEE
    MMMISSJEEE
    """).strip()
    
    garden_map = [list(line) for line in input_text.split('\n')]
    actual_regions = calculate_regions(garden_map)
    
    # Updated expected data with sides and discount prices
    expected = {
        'R': ExpectedRegion(area=12, perimeter=18, sides=10, expected_price=216, expected_discount_price=120),
        'I': ExpectedRegion(area=4, perimeter=8, sides=4, expected_price=32, expected_discount_price=16),
        'I1': ExpectedRegion(area=14, perimeter=22, sides=16, expected_price=308, expected_discount_price=224),
        'C': ExpectedRegion(area=14, perimeter=28, sides=22, expected_price=392, expected_discount_price=308),
        'C1': ExpectedRegion(area=1, perimeter=4, sides=4, expected_price=4, expected_discount_price=4),
        'F': ExpectedRegion(area=10, perimeter=18, sides=12, expected_price=180, expected_discount_price=120),
        'V': ExpectedRegion(area=13, perimeter=20, sides=10, expected_price=260, expected_discount_price=130),
        'J': ExpectedRegion(area=11, perimeter=20, sides=12, expected_price=220, expected_discount_price=132),
        'E': ExpectedRegion(area=13, perimeter=18, sides=8, expected_price=234, expected_discount_price=104),
        'M': ExpectedRegion(area=5, perimeter=12, sides=6, expected_price=60, expected_discount_price=30),
        'S': ExpectedRegion(area=3, perimeter=8, sides=6, expected_price=24, expected_discount_price=18)
    }
    
    print("\nRegion Comparison:")
    print("-" * 100)
    print(f"{'Plant':^6} | {'Expected':^35} | {'Actual':^35} | {'Match?':^10}")
    print(f"{'':^6} | {'(area, perim, sides, price, disc)':^35} | {'(area, perim, sides, price, disc)':^35} |")
    print("-" * 100)
    
    total_actual = 0
    total_expected = 0
    total_actual_discount = 0
    total_expected_discount = 0
    
    for plant in sorted(set(list(expected.keys()) + list(actual_regions.keys()))):
        exp = expected.get(plant)
        act = actual_regions.get(plant)
        
        if exp and act:
            actual_price = act.area * act.perimeter
            actual_discount_price = act.area * act.sides
            matches = (exp.area == act.area and 
                      exp.perimeter == act.perimeter and 
                      exp.sides == act.sides and
                      exp.expected_price == actual_price and
                      exp.expected_discount_price == actual_discount_price)
            
            print(f"{plant:^6} | ({exp.area:2d}, {exp.perimeter:2d}, {exp.sides:2d}, {exp.expected_price:4d}, {exp.expected_discount_price:4d}) | "
                  f"({act.area:2d}, {act.perimeter:2d}, {act.sides:2d}, {actual_price:4d}, {actual_discount_price:4d}) | "
                  f"{'✓' if matches else '✗'}")
            
            total_expected += exp.expected_price
            total_actual += actual_price
            total_expected_discount += exp.expected_discount_price
            total_actual_discount += actual_discount_price
        else:
            print(f"{plant:^6} | {'missing' if not exp else '':<35} | "
                  f"{'missing' if not act else '':<35} | ✗")
    
    print("-" * 100)
    print(f"Regular Totals: Expected={total_expected}, Actual={total_actual}, "
          f"{'Match' if total_expected == total_actual else 'Differ'}")
    print(f"Discount Totals: Expected={total_expected_discount}, Actual={total_actual_discount}, "
          f"{'Match' if total_expected_discount == total_actual_discount else 'Differ'}")

    return (total_actual == total_expected and 
            total_actual_discount == total_expected_discount)


def count_sides(border_map):
    # for row in border_map:
    #     print(''.join(row))
    # print()

    def check_verticals(border_map):
        unique_walls = set()
        half_map = [row for i, row in enumerate(border_map) if i % 2 == 1]
        # for row in half_map:
        #     print(''.join(row))
        for row in range(len(half_map)):
            for col in range(len(half_map[0])):
                visited = set()
                if half_map[row][col] == "|":
                    region_cells, _ = flood_fill(half_map, (row, col), visited)
                    # print(region_cells)
                    unique_walls.add(frozenset(region_cells))
        # print(f"check_verticals: {len(unique_walls)}")
        return unique_walls
    
    def check_horizontals(border_map):
        unique_walls = set()
        half_map = [[col for j, col in enumerate(row) if j % 2 == 1] for row in border_map]
        # for row in half_map:
        #     print(''.join(row))
        for row in range(len(half_map)):
            for col in range(len(half_map[0])):
                visited = set()
                if half_map[row][col] == "-":
                    region_cells, _ = flood_fill(half_map, (row, col), visited)
                    unique_walls.add(frozenset(region_cells))
        # print(f"check_horizontals: {len(unique_walls)}")
        return unique_walls
    
    
    def check_crosses(border_map):
        additional = 0
        for row in range(len(border_map)):
            for col in range(len(border_map[0])):
                try:
                    if border_map[row+1][col] == "|" and \
                        border_map[row-1][col] == "|" and \
                        border_map[row][col+1] == "-" and \
                        border_map[row][col-1] == "-":
                        additional = 2
                except IndexError:
                    pass
        return additional

    vert_unique_walls = check_verticals(border_map)
    hoz_unique_walls = check_horizontals(border_map)
    additional = check_crosses(border_map)
    # print(vert_unique_walls)
    # print(hoz_unique_walls)
    # print(additional)
    return len(vert_unique_walls) + len(hoz_unique_walls) + additional


def create_border_map(region_cells: Set[Tuple[int, int]], rows: int, cols: int) -> List[List[str]]:
    """Create a border map with + - | characters around the region."""
    # Create a larger map to fit borders (2x + 1 size)
    border_map = [[' ' for _ in range(2 * cols + 1)] for _ in range(2 * rows + 1)]
    
    # First pass: Fill in the borders
    for i, j in region_cells:
        # Convert region coordinates to border map coordinates
        bi, bj = 2 * i + 1, 2 * j + 1
        
        # Mark cell position
        border_map[bi][bj] = '#'
        
        # Check and mark borders with neighbors
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for di, dj in neighbors:
            ni, nj = i + di, j + dj
            if (ni, nj) not in region_cells:
                # Add border elements
                border_i = bi + di
                border_j = bj + dj
                if di == 0:  # Vertical border
                    border_map[bi][border_j] = '|'
                else:  # Horizontal border
                    border_map[border_i][bj] = '-'
    
    return border_map


def test_garden_regions():
    """Test various garden configurations and their expected scores."""
    test_cases = [
        # Test case 1: 4x4 grid
        {
            "garden": """
AAAA
BBCD
BBCC
EEEC""",
            "expected": 80
        },
        
        # Test case 2: 5x5 grid with alternating pattern
        {
            "garden": """OOOOO
OXOXO
OOOOO
OXOXO
OOOOO""",
            "expected": 436
        },
        
        # Test case 3: 5x5 grid with E's and X's
        {
            "garden": """EEEEE
EXXXX
EEEEE
EXXXX
EEEEE""",
            "expected": 236
        },
        
        # Test case 4: 6x6 grid with A's and B's
        {
            "garden": """AAAAAA
AAABBA
AAABBA
ABBAAA
ABBAAA
AAAAAA""",
            "expected": 368
        },
        
        # Test case 5: 10x10 grid with multiple letters
        {
            "garden": """RRRRIICCFF
RRRRIICCCF
VVRRRCCFFF
VVRCCCJFFF
VVVVCJJCFE
VVIVCCJJEE
VVIIICJJEE
MIIIIIJJEE
MIIISIJEEE
MMMISSJEEE""",
            "expected": 1206
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        garden_map = test_case["garden"].strip().split('\n')
        for row in garden_map:
            print(''.join(row))
        regions = calculate_regions(garden_map)
        for region in regions.items():
            print(region)
        result = calculate_total_score(regions, use_sides=True)
        assert result == test_case["expected"], \
            f"Test case {i} failed: Expected {test_case['expected']}, got {result}"
        print(f"Test case {i} passed!")


def main():
    test_garden_regions()

    garden_map = parse_input("inputs/day12_input.txt")
    regular_price = calculate_total_score(calculate_regions(garden_map), use_sides=False)
    discount_price = calculate_total_score(calculate_regions(garden_map), use_sides=True)
    print(f"Regular price: {regular_price}")
    print(f"Discount price: {discount_price}")

    #  890206 discount price is too low


if __name__ == "__main__":
    main()