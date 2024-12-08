import pytest
from day7 import parse_input, solve_operands, Equation

def test_2_add():
    e1 = Equation(29, [10, 19])
    assert solve_operands([e1]) == 29

def test_2_mul():
    e1 = Equation(190, [10, 19])
    assert solve_operands([e1]) == 190

def test_2_no_answer():
    e1 = Equation(191, [10, 19])
    assert solve_operands([e1]) == 0

def test_3_mixed_answer():
    e1 = Equation(192, [10, 19, 2])
    assert solve_operands([e1]) == 192

def test_3_mixed_answer2():
    e1 = Equation(31, [10, 19, 2])
    assert solve_operands([e1]) == 31

def test_3_no_answer():
    e1 = Equation(191, [10, 19, 2])
    assert solve_operands([e1]) == 0