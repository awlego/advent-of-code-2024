import pytest
from day5 import parse_input, create_ordering_map, order, check_manual

def test_simple_ordering():
    # Simple test case with clear ordering: 3 -> 2 -> 1
    test_rules = {
        2: [1],
        3: [2]
    }
    
    order_map = create_ordering_map(test_rules)
    print(order_map)
    test_manual = [3, 1, 2]
    ordered_manual = order(test_manual, order_map)
    
    assert ordered_manual == [3, 2, 1]
    assert check_manual(ordered_manual, test_rules)

def test_complex_ordering():
    # More complex case with multiple dependencies
    test_rules = {
        3: [1, 2],  # 3 must come after both 1 and 2
        4: [2],     # 4 must come after 2
        5: [3, 4]   # 5 must come after both 3 and 4
    }
    
    order_map = create_ordering_map(test_rules)
    test_manual = [5, 4, 3, 2, 1]
    ordered_manual = order(test_manual, order_map)
    
    assert check_manual(ordered_manual, test_rules)

def test_cyclic_dependencies():
    # Test handling of cyclic dependencies
    test_rules = {
        1: [2],
        2: [3],
        3: [1]
    }
    
    with pytest.raises(ValueError, match="Cycle detected"):
        order_map = create_ordering_map(test_rules)
        test_manual = [3, 2, 1]
        ordered_manual = order(test_manual, order_map)

def test_with_actual_input():
    # Test with real input file
    rules, manuals = parse_input('inputs/day5_input_test.txt')
    order_map = create_ordering_map(rules)
    
    # Test ordering of first invalid manual
    invalid_manuals = [m for m in manuals if not check_manual(m, rules)]
    if invalid_manuals:
        fixed_manual = order(invalid_manuals[0], order_map)
        assert check_manual(fixed_manual, rules) 
