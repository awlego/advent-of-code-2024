import re
from typing import List, Tuple

def find_enable_positions(line: str) -> List[Tuple[int, bool]]:
    """Returns a sorted list of (position, enabled) tuples from do() and don't() matches"""
    positions = []
    
    for match in re.finditer(r"do\(\)", line):
        positions.append((match.start(), True))
    for match in re.finditer(r"don't\(\)", line):
        positions.append((match.start(), False))
    
    return sorted(positions, key=lambda x: x[0])

def is_enabled_at_position(position: int, enable_positions: List[Tuple[int, bool]], previous_state: bool) -> bool:
    """Determines if multiplication is enabled at the given position"""
    current_state = previous_state
    
    for pos, state in enable_positions:
        if pos > position:
            break
        current_state = state
    
    return current_state

def process_line(line: str, previous_state: bool) -> tuple[int, bool]:
    """
    Process a single line and return (sum, final_state)
    final_state will be used as the previous_state for the next line
    """
    sum_pattern = r"mul\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
    total = 0
    
    enable_positions = find_enable_positions(line)
    
    for match in re.finditer(sum_pattern, line):
        position = match.start()
        if is_enabled_at_position(position, enable_positions, previous_state):
            num1, num2 = map(int, match.groups())
            print(f"Enabled multiplication at position {position}: mul({num1}, {num2})")
            total += num1 * num2
        else:
            num1, num2 = map(int, match.groups())
            print(f"Skipping disabled multiplication at position {position}: mul({num1}, {num2})")
    
    final_state = previous_state
    if enable_positions:
        final_state = enable_positions[-1][1]
    
    return total, final_state

def main():
    """Process the input file"""
    total = 0
    current_state = True
    
    with open('inputs/day3_input.txt', 'r') as file:
        for line in file:
            line_sum, current_state = process_line(line.strip(), current_state)
            total += line_sum
    print(f"Final total: {total}")
    return total

def test_cases():
    """Run through various test cases to verify functionality"""
    test_inputs = [
        (["mul(2,3)"], 6, "Basic multiplication"),
        (["mul(2,3) mul(4,5)"], 26, "Multiple multiplications"),        

        (["mul(2,3)", "mul(4,5)"], 26, "Basic multi-line"),
        (["don't()", "mul(2,3)", "mul(4,5)"], 0, "State carries across lines"),
        (["do()", "mul(2,3)", "don't()", "mul(4,5)"], 6, "State changes across lines"),
        
        (["mul(2,3) don't()", "mul(4,5)", "do() mul(6,7)"], 48, "State changes mid-lines"),
        (["don't()", "mul(2,3)", "do()", "mul(4,5)"], 20, "Multiple lines with state changes"),
    ]
    
    passed = 0
    failed = 0
    
    for test_input, expected, description in test_inputs:
        total = 0
        current_state = True
        for line in test_input:
            line_sum, current_state = process_line(line, current_state)
            total += line_sum
        
        if total == expected:
            print(f"✓ PASS: {description}")
            print(f"  Input: {test_input}")
            print(f"  Expected: {expected}, Got: {total}")
            passed += 1
        else:
            print(f"✗ FAIL: {description}")
            print(f"  Input: {test_input}")
            print(f"  Expected: {expected}, Got: {total}")
            failed += 1
        print()
    
    print(f"Test Results: {passed} passed, {failed} failed")
    return passed, failed

if __name__ == "__main__":
    test_cases()
    main()


