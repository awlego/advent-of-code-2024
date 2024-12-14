from dataclasses import dataclass
from typing import List, Tuple, Union
from tqdm import tqdm
import re
import numpy as np
from math import gcd, ceil, floor

class NoSolutionError(Exception):
    """Exception raised when no solution exists"""
    pass

@dataclass(frozen=True)
class Coord():
    x: int
    y: int

    def __post_init__(self):
        object.__setattr__(self, 'x', int(self.x))
        object.__setattr__(self, 'y', int(self.y))

    def __add__(self, other: Union['Coord', int]) -> 'Coord':
        if isinstance(other, Coord):
            return Coord(x=self.x + other.x, y=self.y + other.y)
        elif isinstance(other, int):
            return Coord(x=self.x + other, y=self.y + other)

    def __sub__(self, other: 'Coord') -> 'Coord':
        return Coord(x=self.x - other.x, y=self.y - other.y)
    
@dataclass
class ClawMachine:
    a_diff: Coord
    b_diff: Coord
    prize_location: Coord
    a_cost = 3
    b_cost = 1
   

def parse_input(filepath: str) -> list[ClawMachine]:
    claw_machines = []
    with open(filepath, 'r') as openfile:
        while True:
            lines = [openfile.readline().strip() for _ in range(4)]
            if not any(lines):
                break
            nums = re.findall("(\d+)", lines[0])
            a_diff = Coord(x=nums[0], y=nums[1])
            nums = re.findall("(\d+)", lines[1])
            b_diff = Coord(x=nums[0], y=nums[1])
            nums = re.findall("(\d+)", lines[2])
            prize_location = Coord(x=nums[0], y=nums[1])

            claw_machines.append(ClawMachine(a_diff=a_diff, b_diff=b_diff, prize_location=prize_location))
    return claw_machines


def find_particular_solution(vector_a, vector_b, target):
    """
    Finds a particular solution to the system of equations:
    a₁x + b₁y = t₁
    a₂x + b₂y = t₂
    """
    a1, a2 = vector_a
    b1, b2 = vector_b
    t1, t2 = target
    tqdm.write(f"a1: {a1}, a2: {a2}")
    tqdm.write(f"b1: {b1}, b2: {b2}")

    det = a1 * b2 - a2 * b1
    if det == 0:
        raise NoSolutionError("Vectors are linearly dependent")
    
    tqdm.write(f"Determinant: {det}")

    # Check if solution would be integer
    if (t1 * b2 - t2 * b1) % det != 0 or (a1 * t2 - a2 * t1) % det != 0:
        raise NoSolutionError("No integer solution exists")
    
    # Compute particular solution
    x = (t1 * b2 - t2 * b1) // det
    y = (a1 * t2 - a2 * t1) // det
    
    return x, y

def solve_system(x1, y1, x2, y2, xp, yp):
    """
    Solves the system of equations:
    a⋅x1 + b⋅x2 = xp
    a⋅y1 + b⋅y2 = yp
    
    Returns (a, b) or raises NoSolutionError if no integer solution exists
    """
    denominator = x2 * y1 - y2 * x1
    
    if denominator == 0:
        raise NoSolutionError("System is linearly dependent")
        
    b_num = xp * y1 - yp * x1
    
    if b_num % denominator != 0:
        raise NoSolutionError("No integer solution exists")
        
    b = b_num // denominator
    
    if x1 == 0:
        if y1 == 0:
            raise NoSolutionError("Invalid vector A (0,0)")
        a = (yp - b * y2) // y1
    else:
        a_num = xp - b * x2
        if a_num % x1 != 0:
            raise NoSolutionError("No integer solution exists")
        a = a_num // x1
        
    if (a * x1 + b * x2 != xp) or (a * y1 + b * y2 != yp):
        raise NoSolutionError("Solution verification failed")
        
    return a, b

def find_kernel_vector(vector_a, vector_b):
    """
    Finds the kernel (homogeneous solution) vector of the system
    Returns (k₁, k₂) where all solutions are of form:
    (x₀ + t*k₁, y₀ + t*k₂) for some integer t
    
    The kernel vector must satisfy:
    a₁k₁ + b₁k₂ = 0
    a₂k₁ + b₂k₂ = 0
    """
    a1, a2 = vector_a
    b1, b2 = vector_b
    
    k1 = b2 - b1
    k2 = a1 - a2

    # Simplify to smallest integer components
    gcd_k = gcd(k1, k2)
    k1 //= gcd_k
    k2 //= gcd_k


    print(f"a1*k1 + b1*k2 = {a1}*{k1} + {b1}*{k2} = {a1*k1 + b1*k2}")
    assert a1*k1 + b1*k2 == 0
    print(f"a2*k1 + b2*k2 = {a2}*{k1} + {b2}*{k2} = {a2*k1 + b2*k2}")
    assert a2*k1 + b2*k2 == 0
    
    return k1, k2


def verify_solution(vector_a, vector_b, target, steps_a, steps_b):
    """Verify that a solution actually reaches the target"""
    final_pos = steps_a * vector_a + steps_b * vector_b
    return np.array_equal(final_pos, target)


def find_optimal_t(x0, y0, k1, k2, weight_a, weight_b):
    """
    Find optimal t that minimizes weight_a|x₀ + tk₁| + weight_b|y₀ + tk₂|
    """
    def compute_cost(t):
        return weight_a * abs(x0 + t * k1) + weight_b * abs(y0 + t * k2)

    critical_points = set()
    if k1 != 0:
        critical_points.add(-x0 // k1)
        critical_points.add(ceil(-x0 / k1))
    if k2 != 0:
        critical_points.add(-y0 // k2)
        critical_points.add(ceil(-y0 / k2))

    # Evaluate all critical points and neighbors
    candidates = sorted(critical_points)
    costs = [(compute_cost(t), t) for t in candidates]
    best_cost, best_t = min(costs)
    
    return best_t



def solve_vector_system_weighted(vector_a, vector_b, target, weight_a=1, weight_b=1):
    """
    Solves the system with weighted costs on both vectors
    Returns the exact optimal solution
    """
    # Find a particular solution
    x0, y0 = find_particular_solution(vector_a, vector_b, target)
    tqdm.write(f"Particular solution: x0: {x0}, y0: {y0}")
    
    # Find the kernel vector
    k1, k2 = find_kernel_vector(vector_a, vector_b)
    tqdm.write(f"k1: {k1}, k2: {k2}")
    
    # Find optimal t
    t = find_optimal_t(x0, y0, k1, k2, weight_a, weight_b)
    tqdm.write(f"t: {t}")
    
    # Compute final solution
    steps_a = x0 + t * k1
    steps_b = y0 + t * k2

    tqdm.write(f"optimized steps: {steps_a} {steps_b}")

    if not verify_solution(vector_a, vector_b, target, steps_a, steps_b):
        raise NoSolutionError("Final solution does not reach target")
    
    return int(steps_a), int(steps_b)


def part_a(claw_machines):
    win_all_prizes_cost = 0
    num_prizes = 0
    for claw in tqdm(claw_machines):
        # proof that we have a solution
        # steps_a = 80
        # steps_b = 40
        # vector_a = np.array([claw.a_diff.x, claw.a_diff.y])
        # vector_b = np.array([claw.b_diff.x, claw.b_diff.y])
        # target = np.array([claw.prize_location.x, claw.prize_location.y])
        # print(verify_solution(vector_a, vector_b, target, steps_a, steps_b))
        # total_weighted_cost = claw.a_cost * steps_a + claw.b_cost * steps_b
        # print(total_weighted_cost)

        # try our program's attempt at finding the solution
        tqdm.write("")
        try:
            vector_a = np.array([claw.a_diff.x, claw.a_diff.y])
            vector_b = np.array([claw.b_diff.x, claw.b_diff.y])
            target = np.array([claw.prize_location.x, claw.prize_location.y])
            # steps_a, steps_b = solve_vector_system_weighted(vector_a, vector_b, target, claw.a_cost, claw.b_cost)
            steps_a, steps_b = solve_system(vector_a[0], vector_a[1], vector_b[0], vector_b[1], target[0], target[1])
            final_pos = steps_a * vector_a + steps_b * vector_b
            total_weighted_cost = claw.a_cost * steps_a + claw.b_cost * steps_b
            
            tqdm.write(f"Solution found!")
            tqdm.write(f"Vector A: {vector_a}, Vector B: {vector_b}")
            tqdm.write(f"Steps of vector A: {steps_a}")
            tqdm.write(f"Steps of vector B: {steps_b}")
            tqdm.write(f"Final position: {final_pos}")
            tqdm.write(f"Target position: {target}")
            tqdm.write(f"Total weighted cost: {total_weighted_cost}")
            tqdm.write(f"Verification - position matches target: {np.array_equal(final_pos, target)}")

            win_all_prizes_cost += total_weighted_cost
            num_prizes += 1
        except NoSolutionError as e:
            tqdm.write(f"Error: {str(e)}")
            pass
    
    print(f"{num_prizes}/{len(claw_machines)} prizes found with a total cost of {win_all_prizes_cost}")
    assert win_all_prizes_cost == 36954

def part_b(claw_machines):
    for claw in claw_machines:
        claw.prize_location += 10000000000000
    
    part_a(claw_machines)


def main():
    claw_machines = parse_input("inputs/day13_input.txt")
    part_a(claw_machines)
    part_b(claw_machines)


if __name__ == "__main__":
    main()
    # 79352011449276
    # 27074863625559 # too low
    # 49383533824251
