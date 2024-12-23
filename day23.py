from itertools import combinations
import networkx as nx

def parse_input(filename):
    connections = {}
    
    with open(filename, 'r') as file:
        for line in file:
            node1, node2 = line.strip().split('-')

            if node1 not in connections:
                connections[node1] = []
            if node2 not in connections:
                connections[node2] = []
                
            connections[node1].append(node2)
            connections[node2].append(node1)
    
    return connections

def sets_of_connected_computers(network, num_computers=3):
    G = nx.Graph(network)
    
    connected_sets = [set(clique) for clique in nx.enumerate_all_cliques(G) 
                     if len(clique) == num_computers]
    
    return connected_sets

def largest_connected_set(network):
    G = nx.Graph(network)
    largest_set = max(nx.find_cliques(G), key=len)
    return largest_set


def names_starts_with(interconnected_computers, letter):
    matching_computer_sets = []
    for computer_set in interconnected_computers:
        for computer in computer_set:
            if computer.startswith(letter):
                matching_computer_sets.append(computer_set)
                break
    return matching_computer_sets

def main():
    network = parse_input('inputs/day23_input.txt')
    
    connected_sets = sets_of_connected_computers(network, 3)

    print(len(names_starts_with(connected_sets, 't')))

    print(sorted(largest_connected_set(network)))

if __name__ == "__main__":
    main()

