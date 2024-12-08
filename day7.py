from dataclasses import dataclass
from collections import deque
from typing import List

@dataclass
class Equation:
    answer: int
    operands: List[int]

def parse_input(filepath: str) -> List[Equation]:
    equations = []
    with open(filepath, 'r') as openfile:
        for line in openfile:
            line = line.strip()
            answer = int(line.split(":")[0])
            operands = [int(o) for o in line.split(":")[1].split()]
            equations.append(Equation(answer=answer, operands=operands))
    return equations

class Node:
    def __init__(self, value: str, current_total: int, remaining_operands: List[int]):
        self.value = value  # '+' or '*' or '||'
        self.current_total = current_total
        self.remaining_operands = remaining_operands

def bfs_evaluate(equation: Equation):
    queue = deque([
        Node('+', equation.operands[0], equation.operands[1:]),
        Node('*', equation.operands[0], equation.operands[1:]),
        Node('||', equation.operands[0], equation.operands[1:])
    ])
    
    while queue:
        current = queue.popleft()
        
        if not current.remaining_operands:
            if current.current_total == equation.answer:
                return True
            continue
            
        next_operand = current.remaining_operands[0]
        remaining = current.remaining_operands[1:]
        
        queue.append(Node('+', 
            current.current_total + next_operand, 
            remaining))
        queue.append(Node('*', 
            current.current_total * next_operand, 
            remaining))
        queue.append(Node('||', 
            int(str(current.current_total) + str(next_operand)), 
            remaining))
    
    return False


def solve_operands(equations: List[Equation]) -> None:
    '''finds operands that work and prints a sum of those that do'''
    sum = 0
    for equation in equations:
        if bfs_evaluate(equation):
            sum += equation.answer

    print(sum)
    return sum


def main():
    equations = parse_input("inputs/day7_input.txt")
    solve_operands(equations)

def get_stats():
    equations = parse_input("inputs/day7_input.txt")
    n_lens = []
    for equation in equations:        
        n_lens.append(len(equation.operands))
    print(f"There are e={len(equations)} equations")
    print(f"Min n={min(n_lens)} operands")
    print(f"Max n={max(n_lens)} operands")


if __name__ == "__main__":
    get_stats()
    main()