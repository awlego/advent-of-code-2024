from collections import defaultdict
from typing import List, Tuple, Dict, Set, DefaultDict

def parse_input(file_path: str) -> Tuple[DefaultDict[int, List[int]], List[List[int]]]:
    parsing_rules = True
    page_ordering_rules = defaultdict(list)
    safety_manuals = []
    with open(file_path, 'r') as file:
        for line in file:
            # line break in input indicates moving from rules to manuals
            if line == ('\n'):
                parsing_rules = False
                continue
            line = line.strip()
            if parsing_rules:
                line = line.split("|")
                page = int(line[0])
                rule = int(line[1])
                page_ordering_rules[page].append(rule)
            else:
                line = line.split(",")
                safety_manuals.append([int(x) for x in line])



    # print(page_ordering_rules)
    # print(safety_manuals)
    return (page_ordering_rules, safety_manuals)


def check_manual(manual: List[int], page_ordering_rules: DefaultDict[int, List[int]]) -> bool:
    '''checks a given manual to see if it follows all ordering rules'''

    for page in page_ordering_rules:
        for rule in page_ordering_rules[page]:
            if page in manual and rule in manual:
                page_index = manual.index(page)
                rule_index = manual.index(rule)
                if page_index > rule_index:
                    return False
                
    return True


def build_graph(page_ordering_rules: DefaultDict[int, List[int]]) -> DefaultDict[int, Set[int]]:
    graph = defaultdict(set)
    
    for page, must_come_before in page_ordering_rules.items():            
        for before_page in must_come_before:
            graph[page].add(before_page)
                
    return graph

def topological_sort_dfs(dag: DefaultDict[int, Set[int]]) -> Dict[int, int]:
    '''uses depth-first search to create an ordering map from a DAG'''    
    visited = set()        # Nodes we're completely done with
    temp_visited = set()     # Nodes in our current exploration path
    order_list = []
    order_map = {}           # Our sorted result

    def explore_node(node):
        if node in temp_visited: 
            raise ValueError(f"Cycle detected at node {node}. Current visited: {visited}, current temp_visited: {temp_visited}")
        if node in visited:
            return
        
        temp_visited.add(node)

        for next_node in dag[node]:
            explore_node(next_node)
        
        temp_visited.remove(node)
        visited.add(node)
        order_list.append(node)


    all_nodes = set(dag.keys()) | {node for nodes in dag.values() for node in nodes}
    for node in all_nodes:
        if node not in visited:
            explore_node(node)

    for position, page in enumerate(reversed(order_list)):
        order_map[page] = position

    return order_map


def create_ordering_map(page_ordering_rules: DefaultDict[int, List[int]]) -> Dict[int, int]:
    '''Creates a map of page to its position in the ordering based on rules
    
    page_ordering_rules: a defaultdict with a list of integers. 
    For each key, all elements in the list must come before the key.
    
    e.g. The first rule, {47: [53, 44]}, means that if an update includes both 
    page number 47 and page number 53, then page number 47 must be printed 
    at some point before page number 53. (47 doesn't necessarily need to be 
    immediately before 53; other pages are allowed to be between them. 
    The same is true for 47 and 44, 47 must come before 44.
    '''

    # We can make a DAG out of the page ordering rules.
    dag = build_graph(page_ordering_rules)

    # Then we can do a topological sort.
    return topological_sort_dfs(dag)


def filter_ordering_rules(manual: List[int], page_ordering_rules: DefaultDict[int, List[int]]) -> DefaultDict[int, List[int]]:
    '''return only ordering rules that pertain to pages in the manual'''
    manual_pages = set(manual)
    filtered_rules = defaultdict(list)
    
    for page, rules in page_ordering_rules.items():
        if page in manual_pages:
            # Only keep rules for pages that exist in the manual
            filtered_rules[page] = [r for r in rules if r in manual_pages]
            
    return filtered_rules

def order(manual: List[int], page_ordering_rules: DefaultDict[int, List[int]]) -> List[int]:
    '''takes an invalid manual and correctly orders it according to the order map'''
    
    # we need to filter the ordering rules to just include things in the manual
    # because there are cycles in the ordering rules
    filtered_ordering_rules = filter_ordering_rules(manual, page_ordering_rules)
    order_map = create_ordering_map(filtered_ordering_rules)


    return sorted(manual, key=lambda x: order_map.get(x, float('-inf')))


def main() -> None:
    (page_ordering_rules, safety_manuals) = parse_input('inputs/day5_input.txt')

    valid_manuals = [manual for manual in safety_manuals if check_manual(manual, page_ordering_rules)]
    print(f"Found {len(valid_manuals)}/{len(safety_manuals)} valid manuals.")
    sum = 0
    for manual in valid_manuals:
        sum += manual[len(manual)//2]

    print(f"Sum of middle numbers of valid manuals: {sum}")
    
    # PART 2
    # note you can't just create one order map since the page ordering rules have cycles
    # order_map = create_ordering_map(page_ordering_rules)

    invalid_manuals = [manual for manual in safety_manuals if not check_manual(manual, page_ordering_rules)]
    print(f"Found {len(invalid_manuals)}/{len(safety_manuals)} invalid manuals.")
    fixed_manuals = []
    for manual in invalid_manuals:
        fixed_manuals.append(order(manual, page_ordering_rules))

    sum = 0
    for manual in fixed_manuals:
        sum += manual[len(manual)//2]
    print(f"Sum of middle numbers of only fixed manuals: {sum}")
if __name__ == "__main__":
    main()



