import math
from dataclasses import dataclass
from itertools import islice
from multiprocessing import Pool, cpu_count
import copy

def parse_input(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    # Extract register values
    register_a = int(lines[0].split(': ')[1])
    register_b = int(lines[1].split(': ')[1])
    register_c = int(lines[2].split(': ')[1])
    
    # Extract program instructions
    program = list(map(int, lines[4].split(': ')[1].split(',')))
    
    return register_a, register_b, register_c, program


class ChronospatialComputer():

    def __init__(self, a, b, c, program):
        self.register_a = a
        self.register_b = b
        self.register_c = c
        self.program = program
        self.program_counter = 0
        self.inc_pc = True
        self.operations = {
            0: self.adv,
            1: self.bxl,
            2: self.bst,
            3: self.jnz,
            4: self.bxc,
            5: self.out,
            6: self.bdv,
            7: self.cdv
        }
        self.operands = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: lambda: self.register_a,
            5: lambda: self.register_b,
            6: lambda: self.register_c,

        }
        self.combo_operations = {
            0: True,
            1: False,
            2: True,
            3: False,
            4: False,
            5: True,
            6: True,
            7: True,
        }
        self.output = ""

    def compute(self):
        self.output = ""
        while self.program_counter < len(self.program):
            self.inc_pc = True
            opcode = self.program[self.program_counter]
            operand = self.program[self.program_counter + 1]
            if opcode in self.operations:
                if self.combo_operations[opcode] == True:
                    operand_value = self.operands[operand]() if callable(self.operands[operand]) else self.operands[operand]
                    self.operations[opcode](operand_value)
                else:
                    self.operations[opcode](operand)
            else:
                raise ValueError(f"Invalid opcode: {opcode}")
            if self.inc_pc == True:
                self.program_counter += 2
            
        return self.output.strip(",")
    
    def adv(self, combo):
        denom = 2**combo
        div = self.register_a / denom
        self.register_a = math.floor(div)
    
    def bxl(self, literal):
        self.register_b = self.register_b ^ literal
    
    def bst(self, combo):
        self.register_b = combo & 0b111

    def jnz(self, literal):
        if self.register_a == 0:
            return
        self.program_counter = literal
        self.inc_pc = False

    def bxc(self, operand):
        self.register_b = self.register_b ^ self.register_c

    def out(self, combo):
        self.output += str(combo % 8) + ","
    
    def bdv(self, combo):
        denom = 2**combo
        div = self.register_a / denom
        self.register_b = math.floor(div)

    def cdv(self, combo):
        denom = 2**combo
        div = self.register_a / denom
        self.register_c = math.floor(div)


def test_examples():
    cc = ChronospatialComputer(0, 0, 9, [2, 6])
    output = cc.compute()
    assert cc.register_b == 1, f"Assertion failed: register_b = {cc.register_b} (expected 1)"

    cc = ChronospatialComputer(10, 0, 0, [5,0])
    output = cc.compute()
    assert output == "0", f"Assertion failed: output = {output} (expected '0')"   

    cc = ChronospatialComputer(10, 0, 0, [5,0,5,1])
    output = cc.compute()
    assert output == "0,1", f"Assertion failed: output = {output} (expected '0,1')"   

    cc = ChronospatialComputer(10, 0, 0, [5,0,5,1,5,4])
    output = cc.compute()
    assert output == "0,1,2", f"Assertion failed: output = {output} (expected '0,1,2')"   

    cc = ChronospatialComputer(2024, 0, 0, [0,1])
    output = cc.compute()
    assert cc.register_a == 1012, f"Assertion failed: register_a = {cc.register_a} (expected 1012)"

    cc = ChronospatialComputer(2024, 3, 0, [0,5])
    output = cc.compute()
    assert cc.register_a == 253, f"Assertion failed: register_a = {cc.register_a} (expected 253)"

    cc = ChronospatialComputer(2024, 0, 0, [0,1,5,4])
    output = cc.compute()
    assert output == "4", f"Assertion failed: output = {output} (expected 4)"

    cc = ChronospatialComputer(2024, 0, 0, [0,1,5,4,3,0])
    output = cc.compute()
    assert output == "4,2,5,6,7,7,7,7,3,1,0", f"Assertion failed: output = {output} (expected '4,2,5,6,7,7,7,7,3,1,0')"   
    assert cc.register_a == 0, f"Assertion failed, register_a = {cc.register_a} (expected 0)"   

    cc = ChronospatialComputer(0, 29, 0, [1,7])
    output = cc.compute()
    assert cc.register_b == 26, f"Assertion failed: register_b = {cc.register_b} (expected 26)"

    cc = ChronospatialComputer(0, 2024, 43690, [4,0])
    output = cc.compute()
    assert cc.register_b == 44354, f"Assertion failed: register_b = {cc.register_b} (expected 44354)"

    cc = ChronospatialComputer(729, 0, 0, [0,1,5,4,3,0])
    output = cc.compute()
    assert output == "4,6,3,5,6,3,5,2,1,0", f"Assertion failed: output = {output} (expected '4,6,3,5,6,3,5,2,1,0')"   


def check_value(args):
    a, register_b, register_c, program = args
    cc = ChronospatialComputer(a, register_b, register_c, program)
    output = cc.compute()
    expected = ",".join(map(str, program))
    return a if output == expected else None


def check_value_with_output(args):
    a, reg_b, reg_c, prog, expected = args
    cc = ChronospatialComputer(a, reg_b, reg_c, prog)
    output = cc.compute()
    return a if output == expected else None

def find_solution_parallel(register_b, register_c, program, expected_output, chunk_size=1000, print_every=10):
    num_cores = cpu_count()
    chunks_processed = 0
    current_start = 0
    
    def generate_args():
        nonlocal current_start
        a = current_start
        while True:
            yield (a, register_b, register_c, program, expected_output)
            a += 1
    
    with Pool(num_cores) as pool:
        while True:
            # Take a chunk of numbers to process
            chunk = list(islice(generate_args(), chunk_size * num_cores))
            if not chunk:
                break
                
            chunks_processed += 1
            current_start += len(chunk)  # Update the starting point for next chunk
            
            if chunks_processed % print_every == 0:
                print(f"Searched up to a={current_start:,}")
                
            # Process the chunk across all cores
            results = pool.map(check_value_with_output, chunk)
            
            # Check if we found a solution
            for result in results:
                if result is not None:
                    return result
    
    return None


def find_solutions_recursive(program, a, results, level):
    """
    Recursively find solutions by working backwards from the last program value.
    
    Args:
        program: The program instructions
        a: Current register_a value to try
        results: List to store valid solutions
        level: Current recursion level (1 = last program value)
    """
    target_value = program[-level]  # Get the value we're looking for at this level
    
    # Try register_a values from a to a+7
    for i in range(8):
        current_a = a + i
        
        # Run the program with this register_a value
        cc = ChronospatialComputer(current_a, 0, 0, program)
        output = cc.compute()
        output_values = [int(x) for x in output.split(',')]
        
        # Check if first output matches our target
        if len(output_values) > 0 and output_values[0] == target_value:
            if level == len(program):
                # We've found a complete solution
                results.append(current_a)
                print(f"Found valid register_a: {current_a}")
            else:
                # Try next level, multiplying a by 8 to get next range
                find_solutions_recursive(program, current_a * 8, results, level + 1)


def solve_part2(program):
    """Find the smallest register_a value that produces program as output."""
    results = []
    find_solutions_recursive(program, 0, results, 1)
    
    if results:
        return min(results)
    return None



if __name__ == '__main__':
    test_examples()

    register_a, register_b, register_c, program = parse_input('inputs/day17_input.txt')

    cc = ChronospatialComputer(register_a, register_b, register_c, program)
    print(f"answer to part a: \n{cc.compute()}\n")
    
    # solution = find_solution_parallel(register_b, register_c, program, program)
    # print(f"answer to part b: \n{solution}\n")
    # checked up to 2,160,900,000 no solutions found
    # time to stop brute forcing...

    solution = solve_part2(program)
    print(f"Found solution: {solution}")
