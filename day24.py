from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Set
from pysat.solvers import Glucose3
from pysat.formula import CNF

import matplotlib.pyplot as plt
import networkx as nx


@dataclass
class Gate:
    op: callable
    input_a: str
    input_b: str
    output: str
    locked: bool = False

@dataclass
class Wire:
    name: str
    value: int | None # using int to represent 0 or 1

OPERATIONS = {
    "AND": int.__and__,
    "OR": int.__or__,  
    "XOR": int.__xor__,
}


class WireChange:
    def __init__(self, wire_name: str, current_value: int, required_value: int):
        self.wire_name = wire_name
        self.current_value = current_value
        self.required_value = required_value
    
    def __repr__(self):
        return f"WireChange({self.wire_name}: {self.current_value} -> {self.required_value})"

class LogicalConstraint:
    def __init__(self, wire_changes: List[WireChange], constraint_type: str):
        self.wire_changes = wire_changes
        self.constraint_type = constraint_type  # "ALL" or "ONE"
    
    def __repr__(self):
        if self.constraint_type == "ALL":
            return f"ALL of: {self.wire_changes}"
        else:
            return f"ONE of: {self.wire_changes}"

def analyze_gate(gate, desired_output: int, wire_states: dict) -> LogicalConstraint:
    """Returns the logical constraints needed to achieve the desired output"""
    input_a_val = wire_states[gate.input_a].value
    input_b_val = wire_states[gate.input_b].value
    current_output = wire_states[gate.output].value
    
    if gate.op == int.__and__:
        if desired_output == 1:
            # Both inputs must be 1
            changes = []
            if input_a_val == 0:
                changes.append(WireChange(gate.input_a, 0, 1))
            if input_b_val == 0:
                changes.append(WireChange(gate.input_b, 0, 1))
            return LogicalConstraint(changes, "ALL")
        else:  # desired_output == 0
            # At least one input must be 0
            changes = []
            if input_a_val == 1:
                changes.append(WireChange(gate.input_a, 1, 0))
            if input_b_val == 1:
                changes.append(WireChange(gate.input_b, 1, 0))
            return LogicalConstraint(changes, "ONE")
    
    elif gate.op == int.__or__:
        if desired_output == 1:
            # At least one input must be 1
            changes = []
            if input_a_val == 0:
                changes.append(WireChange(gate.input_a, 0, 1))
            if input_b_val == 0:
                changes.append(WireChange(gate.input_b, 0, 1))
            return LogicalConstraint(changes, "ONE")
        else:  # desired_output == 0
            # Both inputs must be 0
            changes = []
            if input_a_val == 1:
                changes.append(WireChange(gate.input_a, 1, 0))
            if input_b_val == 1:
                changes.append(WireChange(gate.input_b, 1, 0))
            return LogicalConstraint(changes, "ALL")
    
    elif gate.op == int.__xor__:
        # For XOR, inputs must be different for output 1, same for output 0
        changes = [
            WireChange(gate.input_a, input_a_val, 1 - input_a_val),
            WireChange(gate.input_b, input_b_val, 1 - input_b_val)
        ]
        return LogicalConstraint(changes, "ONE")

def find_minimal_wire_changes(gates, wire_states, bits_to_flip):
    all_constraints = []
    
    for bit in bits_to_flip:
        wire_name = f"z{bit}"
        desired_bit_state = int(not wire_states[wire_name].value)
        gate = find_gate_that_outputs_wire(wire_name)
        
        constraint = analyze_gate(gate, desired_bit_state, wire_states)
        all_constraints.append(constraint)
    
    return all_constraints

# Helper function from your code
def find_gate_that_outputs_wire(wire_name):
    for g in gates:
        if g.output == wire_name:
            return g
    return None

def parse_input(filename):
    initial_wire_states = defaultdict()
    gates = []
    parsing_gates = False
    with open(filename, 'r') as file:
        for line in file.readlines():
            if line == "\n":
                parsing_gates = True
            elif parsing_gates:
                line_info = line.strip().split(" ")
                g = Gate(OPERATIONS[line_info[1]], line_info[0], line_info[2], line_info[-1])
                gates.append(g)
            else:
                line_info = line.strip().split(":")
                initial_wire_states[line_info[0]] = Wire(name=line_info[0], value=int(line_info[1]))

    return initial_wire_states, gates

def add_gates_for_processing(wire_states, gates, gate_queue):
    for g in gates:
        if g.locked:
            continue
    # if both inputs are known, we can flag the gate for evaluation
        try:
            if wire_states[g.input_a].value is not None and wire_states[g.input_b].value is not None:
                gate_queue.append(g)
        except:
            pass

def visualize_circuit(gates):
    """Creates a visualization of the circuit using gates as nodes and wires as edges"""
    
    # Create a new directed graph
    G = nx.DiGraph()
    
    # Add gate nodes
    for gate in gates:
        gate_name = f"{gate.op.__name__}_{gate.output}"
        G.add_node(gate_name, node_type='gate')
    
    # Add edges (wires) between gates
    for gate in gates:
        output_wire = gate.output
        # Find gates that use this output as an input
        for target_gate in gates:
            if target_gate.input_a == output_wire or target_gate.input_b == output_wire:
                source = f"{gate.op.__name__}_{gate.output}"
                target = f"{target_gate.op.__name__}_{target_gate.output}"
                G.add_edge(source, target, wire_name=output_wire)
    
    # Set up the layout
    pos = nx.spring_layout(G)
    
    # Draw the nodes (gates)
    nx.draw_networkx_nodes(G, pos, 
                          node_color='lightcoral',
                          node_shape='s', 
                          node_size=1000)
    
    # Draw edges (wires)
    nx.draw_networkx_edges(G, pos)
    
    # Add labels for gates
    nx.draw_networkx_labels(G, pos)
    
    # Add labels for wires (edges)
    edge_labels = nx.get_edge_attributes(G, 'wire_name')
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    
    plt.title("Circuit Diagram (Gates as Nodes)")
    plt.axis('off')
    plt.show()


def get_wire_value(wire_prefix):
    wire_values = [w.value for w in sorted([w for w in wire_states.values() if w.name.startswith(wire_prefix)], key=lambda x: x.name, reverse=True)]
    binary_string = ''.join(str(x) for x in wire_values)
    result = int(binary_string, 2)
    return result, binary_string

def find_gate_that_outputs_wire(wire_name):
    for g in gates:
        if g.output == wire_name:
            return g
    return None

def convert_wire_to_var_id(wire_name: str) -> int:
    """Convert wire names to positive integers for SAT solver"""
    # Remove any non-alphanumeric characters and convert to a unique integer
    return abs(hash(wire_name)) % (10**9) + 1

def solve_wire_changes(gates, wire_states, bits_to_flip):
    # Initialize SAT solver
    solver = Glucose3()
    cnf = CNF()
    
    # Get all constraints
    constraints = find_minimal_wire_changes(gates, wire_states, bits_to_flip)
    
    # Convert constraints to CNF clauses
    for constraint in constraints:
        if constraint.constraint_type == "ALL":
            # For AND constraints, add each change as a separate clause
            for change in constraint.wire_changes:
                var_id = convert_wire_to_var_id(change.wire_name)
                # If we need to change to 1, add positive literal, else negative
                literal = var_id if change.required_value == 1 else -var_id
                cnf.append([literal])
        
        elif constraint.constraint_type == "ONE":
            # For OR constraints, add all changes as literals in a single clause
            clause = []
            for change in constraint.wire_changes:
                var_id = convert_wire_to_var_id(change.wire_name)
                # If we need to change to 1, add positive literal, else negative
                literal = var_id if change.required_value == 1 else -var_id
                clause.append(literal)
            if clause:  # Only add non-empty clauses
                cnf.append(clause)
    
    # Add clauses to solver
    solver.append_formula(cnf.clauses)
    
    # Solve
    if solver.solve():
        model = solver.get_model()
        # Convert solution back to wire changes
        changes = []
        for var in model:
            wire_name = None
            # Find the wire name that corresponds to this variable
            for change in sum((c.wire_changes for c in constraints), []):
                if convert_wire_to_var_id(change.wire_name) == abs(var):
                    wire_name = change.wire_name
                    break
            if wire_name:
                current_value = wire_states[wire_name].value
                new_value = 1 if var > 0 else 0
                if current_value != new_value:
                    changes.append(WireChange(wire_name, current_value, new_value))
        return changes
    
    return None

wire_states, gates = parse_input("inputs/day24_input.txt")
gate_queue = []

for g in gates:
    if g.output not in wire_states:
        wire_states[g.output] = Wire(name=g.output, value=None)

unprocessed_wires = any([w.value for w in wire_states.values()])
add_gates_for_processing(wire_states, gates, gate_queue)

while unprocessed_wires:
    while gate_queue:
        gate = gate_queue.pop()
        out = gate.op(wire_states[gate.input_a].value, wire_states[gate.input_b].value)
        gate.locked = True
        wire_states[gate.output] = Wire(name=gate.output, value=out)
    
    add_gates_for_processing(wire_states, gates, gate_queue)
    unprocessed_wires = any([w.value == None for w in wire_states.values()])


z_wire_values = get_wire_value("z")
print(f"binary_string: {z_wire_values[1]}")
print(f"Answer to part a: {z_wire_values[0]}")

# part b
# wire_states, gates = parse_input("inputs/day24_input.txt")

x_wire_values = get_wire_value("x")
y_wire_values = get_wire_value("y")

print(x_wire_values)
print(y_wire_values)

# currently:
#  24427219795661
# +17974857205785
#  ==============
#  42410633905894

# true answer to addition:
# 42402077001446

# so I want to change the wiring so that the values on the z wires will become 42402077001446

print(bin(42410633905894))
print(bin(42402077001446))

# bits that need to flip:
def find_flipped_bits(a, b):
    a_bin = bin(a)[2:]
    b_bin = bin(b)[2:]
    # Reverse the index by subtracting from length-1 such that we get the index of the bit from the lowest to the highest
    return [len(a_bin)-1 - i for i in range(len(a_bin)) if a_bin[i] != b_bin[i]]

# of note these are from the highest bit to the lowest bit
bits_that_need_to_flip = find_flipped_bits(42410633905894, 42402077001446)
print(bits_that_need_to_flip)

# so in order to get bit 33 to flip...
# find the gate that outputs wire z33
# hgj XOR cqm -> z33

# either hgj or cqm needs to be flipped.

for bit in bits_that_need_to_flip:
    print(f"bit {bit} needs to be flipped")

    # find the gate that outputs the wire bit and print it's operation and inputs
    for g in gates:
        if g.output == f"z{bit}":
            print(f"gate {g.op.__name__} {g.input_a} {g.input_b} -> {g.output}")

def print_gate_info(gate, desired_bit_state):
    '''prints a nicely formatted gate name, operation, input names and output names along with the values.'''
    op_name = gate.op.__name__
    input_a_val = wire_states[gate.input_a].value
    input_b_val = wire_states[gate.input_b].value
    output_val = wire_states[gate.output].value
    
    print(f"Gate: {op_name}")
    print(f"├── Input A: {gate.input_a} = {input_a_val}")
    print(f"├── Input B: {gate.input_b} = {input_b_val}")
    print(f"└── Output: {gate.output} = {output_val}")
    print(f"desired_bit_state: {desired_bit_state}")
    
# rules:
for bit in bits_that_need_to_flip:
    wire_name = f"z{bit}"
    desired_bit_state = int(not wire_states[wire_name].value)
    gate = find_gate_that_outputs_wire(wire_name)
    print_gate_info(gate, desired_bit_state)
    changes = find_minimal_wire_changes(gates, wire_states, [bit])
    print(changes)
    print()
    #update the wires that need to be changed -- finding changes may have added to the bits that need to flip.
    # we probably can't for loop this because we're changing the bits that need to flip.

    # I can go backwards computing more potential changes... I want to reduce the number of changes until I can find
    # exactly 4 wires that can be swapped to make the circuit work.

# Use the solver
bits_that_need_to_flip = find_flipped_bits(42410633905894, 42402077001446)
# solution = solve_wire_changes(gates, wire_states, bits_that_need_to_flip)

# if solution:
#     print("Solution found! Wire changes needed:")
#     for change in solution:
#         print(f"  {change}")
# else:
#     print("No solution found!")

