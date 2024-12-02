# https://adventofcode.com/2024/day/2
import time

from typing import List
from collections import Counter

input_list_path = "inputs/day2_input.txt"

class Report:
    def __init__(self, levels: List[int]):
        self.levels = levels

    def is_monotonic(self):
        is_decreasing = True
        is_increasing = True
        prev_level = self.levels[0]
        for level in self.levels[1:]:
            if level >= prev_level:
                is_decreasing = False
                break
            prev_level = level

        prev_level = self.levels[0]
        for level in self.levels[1:]:
            if level <= prev_level:
                is_increasing = False
                break
            prev_level = level

        return is_increasing != is_decreasing

    def has_no_big_level_gaps(self):
        prev_level = self.levels[0]
        for level in self.levels[1:]:
            if abs(level - prev_level) > 3:
                return False 
            prev_level = level
        return True
        
    def is_safe(self):
        # print(f"{self.levels} is {self.is_monotonic()} and {self.has_no_big_level_gaps()}")
        return self.is_monotonic() and self.has_no_big_level_gaps()
    
    def is_safe_with_dampener(self):
        if self.is_safe():
            return True
        
        # print(f"{self.levels} is {self.is_monotonic()} and {self.has_no_big_level_gaps()}")
        for i in range(len(self.levels)):
            modified_levels = self.levels[:i] + self.levels[i+1:]
            if Report(modified_levels).is_safe():
                # print(f"{modified_levels} produced a safe report from {self.levels}!")
                return True
            
        return False
    

def read_unusual_data(file_path: str) -> int:
    safe_report_count = 0
    damped_safe_report_count = 0
    with open(file_path, 'r') as f:
        for report in f:
            parsed_report = Report([int(x) for x in report.split()])
            if parsed_report.is_safe():
                safe_report_count += 1
            if parsed_report.is_safe_with_dampener():
                damped_safe_report_count += 1
    return safe_report_count, damped_safe_report_count

if __name__ == "__main__":
    safe_report_count, damped_safe_report_count = read_unusual_data(input_list_path)
    print(f"Safe reports: {safe_report_count}")
    print(f"Safe reports with the Problem Dampener: {damped_safe_report_count}")
