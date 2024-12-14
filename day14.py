import re
from dataclasses import dataclass
from math import prod
from typing import List, Tuple, Union

import numpy as np
import pygame
from tqdm import tqdm


@dataclass
class Robot():
    # 0, 0 top left
    # x, y not r, c
    position: tuple[int, int]
    # tiles per second
    velocity: tuple[int, int]

    def step(self, width: int, height: int) -> None:
        """Updates position based on velocity, wrapping robot
        position around the bounds"""
        x, y = self.position
        dx, dy = self.velocity
        
        new_x = x + dx
        new_y = y + dy
        
        new_x = new_x % width
        new_y = new_y % height
        
        self.position = (new_x, new_y)


def parse_input(filepath: str):
    robots = []
    with open(filepath, 'r') as openfile:
        for line in openfile:
            r = re.findall("([-\d]+)", line)
            robots.append(Robot((int(r[0]), int(r[1])), (int(r[2]), int(r[3]))))
    return robots


def step_robot_positions(robots, w, h):
    for robot in robots:
        robot.step(w, h)


def run_simulation(robots, w, h, seconds: int):
    for i in range(seconds):
        step_robot_positions(robots, w, h)


def count_quadrants(grid):
    '''count robots in each quadrant
    robots on quadrant boundaries (middle row and
    middle column do not count
    '''
    height, width = grid.shape
    mid_h, mid_w = height//2, width//2
    
    q1 = grid[:mid_h, :mid_w]        # top left
    q2 = grid[:mid_h, mid_w+1:]        # top right  
    q3 = grid[mid_h+1:, :mid_w]        # bottom left
    q4 = grid[mid_h+1:, mid_w+1:]        # bottom right

    # make sure I did the quadrant dividing correctly
    assert(q1.shape[0] == mid_h)
    assert(q1.shape[1] == mid_w)
    assert(q2.shape[0] == mid_h)
    assert(q2.shape[1] == mid_w)
    assert(q3.shape[0] == mid_h)
    assert(q3.shape[1] == mid_w)
    assert(q4.shape[0] == mid_h)
    assert(q4.shape[1] == mid_w)

    counts = []
    for quadrant in [q1, q2, q3, q4]:
        count = sum(len(robots) for robots in quadrant.flat)
        counts.append(count)
    
    # making sure I didn't screw up the counts by getting fancy
    counts2 = []
    for quadrant in [q1, q2, q3, q4]:
        count = 0
        for row in quadrant:
            for robots in row:
                count += len(robots)
        counts2.append(count)
    assert counts == counts2
    return counts


def make_grid(robots, w, h):
    grid = np.empty((h, w), dtype=object)
    for r in range(h):
        for c in range(w):
            grid[r,c] = []

    for robot in robots:
        grid[robot.position[1], robot.position[0]].append(robot)
    return grid


def simulation_step(robots, grid):
    h, w = grid.shape
    run_simulation(robots, w=w, h=h, seconds=1)
    grid = make_grid(robots, w=w, h=h)
    return grid


def visualize_grid(robots, width, height, cell_size=12):
    pygame.init()
    
    # Set up the display
    screen = pygame.display.set_mode((width * cell_size, height * cell_size))
    pygame.display.set_caption("Robot Grid")
    
    # Colors
    BACKGROUND = (240, 240, 240)  # Very light gray
    GRID_COLOR = (200, 200, 200)  # Light gray
    NUMBER_COLOR = (255, 64, 0)   # Bright orange-red
    
    # Initialize grid
    grid = make_grid(robots, width, height)
    
    running = True
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Update grid with new robot positions
        # run_simulation(robots, w=width, h=height, seconds=1)
        # grid = make_grid(robots, w=width, h=height)
                
        # Clear screen
        screen.fill(BACKGROUND)
        
        # Draw grid lines and cells
        for r in range(height):
            for c in range(width):
                # Draw cell border
                rect = pygame.Rect(c * cell_size, r * cell_size, cell_size, cell_size)
                pygame.draw.rect(screen, GRID_COLOR, rect, 1)
                
                # Draw number of robots if present
                if len(grid[r,c]) > 0:
                    center = (c * cell_size + cell_size//2, r * cell_size + cell_size//2)
                    font = pygame.font.SysFont(None, 28, bold=True)  # Bold system font
                    text = font.render(str(len(grid[r,c])), True, NUMBER_COLOR)
                    text_rect = text.get_rect(center=center)
                    screen.blit(text, text_rect)
        
        pygame.display.flip()
        clock.tick(2)
        
    pygame.quit()


def test():
    robots = []
    robots.append(Robot((0,4),(3,-3)))
    robots.append(Robot((6,3),(-1,-3)))
    robots.append(Robot((10,3),(-1,2)))
    robots.append(Robot((2,0),(2,-1)))
    robots.append(Robot((0,0),(1,3)))
    robots.append(Robot((3,0),(-2,-2)))
    robots.append(Robot((7,6),(-1,-3)))
    robots.append(Robot((3,0),(-1,-2)))
    robots.append(Robot((9,3),(2,3)))
    robots.append(Robot((7,3),(-1,2)))
    robots.append(Robot((2,4),(2,-3)))
    robots.append(Robot((9,5),(-3,-3)))
    
    width = 11
    height = 7

    run_simulation(robots, w=width, h=height, seconds=100)
    grid = make_grid(robots, w=width, h=height)
    quadrant_counts = count_quadrants(grid)
    print(f"quadrant_counts: {quadrant_counts}")
    safety_factor = prod(quadrant_counts)
    print(safety_factor)
    assert safety_factor == 12
    # visualize_grid(robots, width, height)


def main():
    test()

    robots = parse_input("inputs/day14_input.txt")
    # for robot in robots:
    #     print(robot)
    # robots = [robots[123]]
    # robots = [Robot((0,0),(2,1))]
    width = 101
    height = 103
    # Start visualization with the simulation step function
    # visualize_grid(robots, width, height)


    run_simulation(robots, w=width, h=height, seconds=100)
    grid = make_grid(robots, w=width, h=height)
    quadrant_counts = count_quadrants(grid)
    print(f"quadrant_counts: {quadrant_counts}")
    safety_factor = prod(quadrant_counts)
    print(safety_factor)
    # visualize_grid(robots, width, height)



if __name__ == "__main__":
    main()

