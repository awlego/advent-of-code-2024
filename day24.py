from dataclasses import dataclass
from collections import defaultdict, deque

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

wire_states, gates = parse_input("inputs/day24_input.txt")
gate_queue = []

for g in gates:
    if g.output not in wire_states:
        wire_states[g.output] = Wire(name=g.output, value=None)

unprocessed_wires = any([w.value for w in wire_states.values()])
add_gates_for_processing(wire_states, gates, gate_queue)
print(f"Initial gate queue length: {len(gate_queue)}")

while unprocessed_wires:
    # print(f"unprocessed_wires: {unprocessed_wires}")
    while gate_queue:
        print(f"gate_queue length: {len(gate_queue)}")
        gate = gate_queue.pop()
        out = gate.op(wire_states[gate.input_a].value, wire_states[gate.input_b].value)
        gate.locked = True
        # print(wire_states)
        wire_states[gate.output] = Wire(name=gate.output, value=out)
    
    add_gates_for_processing(wire_states, gates, gate_queue)
    unprocessed_wires = any([w.value == None for w in wire_states.values()])

for w in wire_states.values():
    print(w)
z_gates_values = [w.value for w in sorted([w for w in wire_states.values() if w.name.startswith("z")], key=lambda x: x.name, reverse=True)]
print(''.join(str(x) for x in z_gates_values))
binary_string = ''.join(str(x) for x in z_gates_values)
print(f"binary_string: {binary_string}")
result = int(binary_string, 2)
print(f"Answer to part a: {result}")
