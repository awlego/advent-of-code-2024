# https://adventofcode.com/2024/day/1
import time

from typing import List
from collections import Counter

input_list_path = "inputs/day1_input.txt"


def get_lists(input_path) -> List[int]:
    '''
    With a list like such:
    aaaa bbbb
    a    b
    aaa  bbb

    it will return two lists, each containing the values from the appropriate column
    '''
    with open(input_list_path, 'r') as open_file:
        list_A = []
        list_B = []
        for line in open_file:
            a, b = line.split()
            list_A.append(int(a))
            list_B.append(int(b))
        
        open_file.read

    return (list_A, list_B)


def get_total_difference_between_lists(list_A, list_B):
    '''
    Computes the difference between element n in each list and sums the differences
    '''
    return sum([abs(a - b) for a, b in zip(list_A, list_B)])


def get_similarity_score(list_A, list_B):
    '''
    Similarity score:
    adding up each number in the left list after multiplying it by the number of
    times that number appears in the right list

    this version is O(n^2) where n is the length of list A and B
    '''        
    return sum([number * list_B.count(number) for number in list_A])

def get_similarity_score_optimized(list_A, list_B):
    '''
    Similarity score:
    adding up each number in the left list after multiplying it by the number of
    times that number appears in the right list

    this version is O(n) where n is the length of list A and B but also uses O(n) in space
    '''        
    frequencies = Counter(list_B)
    return sum([number * frequencies[number] for number in list_A])

def print_lists_together(list_A, list_B):
    for a, b in zip(list_A, list_B):
        print(a, b)


if __name__ == "__main__":
    list_A, list_B = get_lists(input_list_path)
    list_A.sort()
    list_B.sort()
    # print_lists_together(list_A, list_B)
    answer = get_total_difference_between_lists(list_A, list_B)
    print(f"Distances between lists: {answer}")
    print(f"Similarity score: {get_similarity_score(list_A, list_B)}")

    # Test original version
    start = time.time()
    original_result = get_similarity_score(list_A, list_B)
    original_time = time.time() - start

    # Test optimized version
    start = time.time()
    optimized_result = get_similarity_score_optimized(list_A, list_B)
    optimized_time = time.time() - start

    print(f"Original version time: {original_time:.4f} seconds")
    print(f"Optimized version time: {optimized_time:.4f} seconds")