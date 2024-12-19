from collections import deque

def parse_input(input="inputs/day19_input.txt"):
    with open(input, 'r') as openfile:
        avail_patterns = openfile.readline()
        avail_patterns = avail_patterns.strip().split(',')
        avail_patterns = [p.strip() for p in avail_patterns]

        openfile.readline() # should be blank

        designs = openfile.readlines()
        designs = [d.strip() for d in designs]
    return avail_patterns, designs


def is_design_possible_km(avail_patterns, design, patterns_used=[], level=0):
    '''
    1 if true
    0 if false

    k = len(avail_patterns)
    m = len(design)
    O(k^m)
    '''
    if design == "":
        return 1

    for pattern in avail_patterns:
        n = len(pattern)
        if design[0:n] == pattern:
            patterns_used.append(pattern)
            pos = is_design_possible(avail_patterns, design[n:], patterns_used, level=level+1)
            if pos:
                return 1

    return 0

class TrieNode:
    def __init__(self):
        self.children = {}  # character -> TrieNode
        self.fail = None
        self.output = set() # patterns that end at this node
        self.is_pattern = False

def build_trie(patterns):
    """
    First phase: Build the basic trie structure from the patterns
    Returns: root node of trie
    """
    root = TrieNode()

    def insert(root, pattern):
        current = root
        for char in pattern:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        # mark end of word
        current.is_pattern = True
        current.output.add(pattern)

    for pattern in patterns:
        insert(root, pattern)

    return root

def print_trie(root, prefix="", is_last=True, char=""):
    # Print current node
    branch = "└── " if is_last else "├── "
    print(prefix + branch + str(char) + (" *" if root.is_pattern else ""))
    
    # Prepare prefix for children
    prefix += "    " if is_last else "│   "
    
    # Get and sort children for consistent output
    children = sorted(root.children.items())
    
    # Print all children except last
    for char, child in children[:-1]:
        print_trie(child, prefix, False, char)
    
    # Print last child
    if children:
        print_trie(children[-1][1], prefix, True, children[-1][0])

def build_failure_links(root):
    queue = deque()
    
    for child in root.children.values():
        child.fail = root
        queue.append(child)
    
    while queue:
        current = queue.popleft()
        for char, child in current.children.items():
            queue.append(child)
            fail_state = current.fail
            while fail_state is not None and char not in fail_state.children:
                fail_state = fail_state.fail
            child.fail = fail_state.children[char] if fail_state else root
            child.output.update(child.fail.output)

def is_design_possible(patterns, design):
    """
    Check if a design can be constructed using the available patterns.
    Patterns can be reused multiple times.
    Returns the count of possible ways to construct the design.
    """
    def count_constructions(pos):
        if pos >= len(design):
            return 1
            
        if pos in memo:
            return memo[pos]
            
        total_count = 0
        for pattern in patterns:
            if pos + len(pattern) <= len(design) and design[pos:pos+len(pattern)] == pattern:
                total_count += count_constructions(pos + len(pattern))
                    
        memo[pos] = total_count
        return total_count

    memo = {}
    
    return count_constructions(0)

def num_possible_designs(avail_patterns, designs, find_all=True):
    num = 0
    for design in designs:
        counts = is_design_possible(avail_patterns, design)
        num += counts if find_all else min(counts, 1)
            
    return num


avail_patterns, designs = parse_input('inputs/day19_input_example.txt')

print(num_possible_designs(avail_patterns, designs, False))
print(num_possible_designs(avail_patterns, designs, True))

print()
avail_patterns, designs = parse_input('inputs/day19_input.txt')
print(num_possible_designs(avail_patterns, designs, False))
print(num_possible_designs(avail_patterns, designs, True))