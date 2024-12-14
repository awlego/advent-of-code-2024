import csv
import os
import re
from collections import deque
from dataclasses import dataclass
from math import prod
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pygame
from matplotlib.animation import FuncAnimation
from PIL import Image, ImageDraw, ImageFont
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


def calculate_sparsity(arr):
    # Count empty lists
    empty_count = sum(1 for elem in arr.flatten() if len(elem) == 0)
    
    # Calculate total elements
    total_elements = arr.size
    
    # Calculate sparsity ratio
    sparsity = empty_count / total_elements
    
    return sparsity

from scipy.spatial.distance import pdist, squareform


def get_clustering_metrics(arr):
    # Get coordinates of non-empty lists
    non_empty_coords = np.array([(i, j) for i in range(arr.shape[0]) 
                                for j in range(arr.shape[1]) 
                                if len(arr[i,j]) > 0])
    
    if len(non_empty_coords) <= 1:
        return {
            'avg_nearest_neighbor_dist': 0,
            'clustering_coefficient': 0
        }
    
    # Calculate pairwise distances between all non-empty elements
    distances = pdist(non_empty_coords)
    dist_matrix = squareform(distances)
    
    # Average distance to nearest neighbor
    # (excluding self-distance which is 0)
    nearest_neighbor_dists = np.array([np.min(dists[dists > 0]) 
                                     for dists in dist_matrix])
    avg_nn_dist = np.mean(nearest_neighbor_dists)
    
    # Calculate a clustering coefficient
    # (ratio of actual nearby neighbors to potential nearby neighbors)
    threshold = 2  # Consider points within 2 units as "clustered"
    neighbor_counts = (dist_matrix < threshold).sum(axis=1) - 1  # subtract self
    max_possible_neighbors = len(non_empty_coords) - 1
    clustering_coeff = np.mean(neighbor_counts / max_possible_neighbors)
    
    return {
        'avg_nearest_neighbor_dist': avg_nn_dist,
        'clustering_coefficient': clustering_coeff
    }


def simulation_step(robots, grid):
    h, w = grid.shape
    run_simulation(robots, w=w, h=h, seconds=1)
    grid = make_grid(robots, w=w, h=h)
    return grid


class ClusteringMonitor:
    def __init__(self, window_size=10, threshold_stds=2.0):
        self.window_size = window_size
        self.threshold_stds = threshold_stds
        self.history = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.metrics = []
        self.max_metric = 0
        self.min_metric = 1000000000
        self.max_metric_timestamp = 0
        self.min_metric_timestamp = 0
        
    def update(self, grid, simulation_time: int) -> Dict:
        """Updates monitoring stats with new grid state
        
        Args:
            grid: The current grid state
            simulation_time: Current simulation time in seconds
        """
        new_min_max = False
        metrics = get_clustering_metrics(grid)
        metric = metrics['avg_nearest_neighbor_dist']
        
        self.history.append(metric)
        self.timestamps.append(simulation_time)

        if metric > self.max_metric:
            self.max_metric = metric
            self.max_metric_timestamp = simulation_time
            new_min_max = True

        if metric < self.min_metric:
            self.min_metric = metric
            self.min_metric_timestamp = simulation_time
            new_min_max = True
        
        return {
            'current_metric': metric,
            'sparsity': calculate_sparsity(grid),
            'min_metric': self.min_metric,
            'min_metric_timestamp': self.min_metric_timestamp,
            'max_metric': self.max_metric,
            'max_metric_timestamp': self.max_metric_timestamp,
            'save': new_min_max,
            **metrics
        } 
    

class MetricsPlotter:
    def __init__(self, max_points=100):
        self.max_points = max_points
        
        # Setup the figure and axis
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.fig.canvas.manager.set_window_title('Clustering Metrics')
        
        # Initialize empty data
        self.times = np.array([])
        self.metrics = np.array([])
        
        # Create initial empty line
        self.line, = self.ax.plot([], [], 'b-', label='Avg Nearest Neighbor Distance')
        
        # Setup axis labels and title
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Distance')
        self.ax.set_title('Clustering Metrics Over Time')
        self.ax.grid(True)
        self.ax.legend()
        
        # Set initial axis limits
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 10)
        
        # Show the plot window
        plt.show(block=False)
        
    def update_data(self, time, metric):
        # Append new data
        self.times = np.append(self.times, time)
        self.metrics = np.append(self.metrics, metric)
        
        # Keep only the last max_points
        if len(self.times) > self.max_points:
            self.times = self.times[-self.max_points:]
            self.metrics = self.metrics[-self.max_points:]
            
        # Update the line data
        self.line.set_data(self.times, self.metrics)
        
        # Update axis limits if needed
        if len(self.times) > 0:
            xmin, xmax = self.times[0], self.times[-1]
            self.ax.set_xlim(max(0, xmax - self.max_points), xmax + 5)
            
            ymin, ymax = np.min(self.metrics), np.max(self.metrics)
            margin = max((ymax - ymin) * 0.1, 0.1)
            self.ax.set_ylim(ymin - margin, ymax + margin)
        
        # Redraw the figure
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def close(self):
        plt.close(self.fig)


def save_picture(grid, timestamp=None, output_dir="robot_snapshots"):
    """
    Saves an image of the current robot grid state
    
    Args:
        grid: numpy array containing robot positions
        timestamp: Current simulation time in seconds
        output_dir: Directory to save images to
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Image settings
    cell_size = 20  # pixels per cell
    padding = 40    # extra space for timestamp
    grid_height, grid_width = grid.shape
    
    # Create image with white background
    img_width = grid_width * cell_size
    img_height = grid_height * cell_size + padding
    image = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Draw grid cells
    for r in range(grid_height):
        for c in range(grid_width):
            x1 = c * cell_size
            y1 = r * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            
            # Draw cell border
            draw.rectangle([x1, y1, x2, y2], outline='lightgray')
            
            # Fill cells containing robots
            if len(grid[r,c]) > 0:
                draw.rectangle([x1+1, y1+1, x2-1, y2-1], fill='orange')
    
    # Add timestamp if provided
    if timestamp is not None:
        try:
            # Try to use a nice font if available
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            # Fall back to default font
            font = ImageFont.load_default()
            
        timestamp_text = f"Time: {timestamp}s"
        text_width = draw.textlength(timestamp_text, font=font)
        text_x = (img_width - text_width) // 2
        text_y = grid_height * cell_size + (padding // 4)
        draw.text((text_x, text_y), timestamp_text, fill='black', font=font)
    
    # Save the image with timestamp in filename
    timestamp_str = f"_{timestamp}s" if timestamp is not None else ""
    filename = f"robot_grid{timestamp_str}.png"
    filepath = os.path.join(output_dir, filename)
    image.save(filepath)
    print(f"Saved grid snapshot to {filepath}")


def visualize_grid(robots, width, height, cell_size=10, show_num=False, start_time=0):
    pygame.init()
    
    window_width = width * cell_size
    window_height = height * cell_size + 40  # Extra space for UI
    
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Robot Grid")
    
    BACKGROUND = (240, 240, 240)  # Very light gray
    GRID_COLOR = (200, 200, 200)  # Light gray
    NUMBER_COLOR = (255, 64, 0)   # Bright orange-red
    BUTTON_COLOR = (100, 100, 100)  # Dark gray
    UI_TEXT_COLOR = (0, 0, 0)     # Black
    
    grid = make_grid(robots, width, height)
    paused = True
    seconds_elapsed = start_time
    
    button_width = 80
    button_height = 30
    button_rect = pygame.Rect((window_width - button_width) // 2, 
                             height * cell_size + 5,
                             button_width, 
                             button_height)
    
    running = True
    clock = pygame.time.Clock()
    
    monitor = ClusteringMonitor(window_size=10, threshold_stds=2.0)
    metrics_plotter = MetricsPlotter(max_points=100000)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    paused = not paused

        if not paused:
            run_simulation(robots, w=width, h=height, seconds=1)
            grid = make_grid(robots, w=width, h=height)
            
            metrics = monitor.update(grid, simulation_time=seconds_elapsed)

            metrics_plotter.update_data(
                seconds_elapsed, 
                metrics['avg_nearest_neighbor_dist']
            )
            
            if metrics:
                print(f"Current avg_nearest_neighbor_dist value: {metrics['current_metric']:.2f}")
                print(f"    Max avg_nearest_neighbor_dist of {metrics['max_metric']} at t={metrics['max_metric_timestamp']}")
            
            seconds_elapsed += 1
            
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
                    if show_num:
                        font = pygame.font.SysFont(None, 28, bold=True) 
                        text = font.render(str(len(grid[r,c])), True, NUMBER_COLOR)
                        text_rect = text.get_rect(center=center)
                        screen.blit(text, text_rect)
                    else:
                        cell_rect = pygame.Rect(c * cell_size, r * cell_size, cell_size, cell_size)
                        pygame.draw.rect(screen, NUMBER_COLOR, cell_rect)
        
        # Draw pause button
        pygame.draw.rect(screen, BUTTON_COLOR, button_rect)
        font = pygame.font.SysFont(None, 24)
        button_text = font.render("PAUSE" if not paused else "RESUME", True, (255, 255, 255))
        text_rect = button_text.get_rect(center=button_rect.center)
        screen.blit(button_text, text_rect)
        
        # Draw seconds counter
        time_font = pygame.font.SysFont(None, 24)
        time_text = time_font.render(f"Time: {seconds_elapsed}s", True, UI_TEXT_COLOR)
        screen.blit(time_text, (10, height * cell_size + 15))
        
        pygame.display.flip()
        clock.tick(60)
        
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
    # test()
    robots = parse_input("inputs/day14_input.txt")

    width = 101
    height = 103
    # visualize_grid(robots, width, height)

    # test_time = 4255
    # run_simulation(robots, w=width, h=height, seconds=test_time)
    # grid = make_grid(robots, w=width, h=height)
    # quadrant_counts = count_quadrants(grid)
    # print(f"quadrant_counts: {quadrant_counts}")
    # safety_factor = prod(quadrant_counts)
    # print(safety_factor)
    # visualize_grid(robots, width, height, start_time=test_time)

    
    monitor = ClusteringMonitor(window_size=10, threshold_stds=2.0)
    metrics_plotter = MetricsPlotter(max_points=100000)

    for t in range(100000000):
        run_simulation(robots, w=width, h=height, seconds=1)
        grid = make_grid(robots, w=width, h=height)
        metrics = monitor.update(grid, simulation_time=t+1)

        metrics_plotter.update_data(
            t, 
            metrics['avg_nearest_neighbor_dist']
        )
        if metrics:
            print(f"Current avg_nearest_neighbor_dist value: {metrics['current_metric']:.2f}")
            print(f"    Min avg_nearest_neighbor_dist of {metrics['min_metric']} at t={metrics['min_metric_timestamp']}")
            print(f"    Max avg_nearest_neighbor_dist of {metrics['max_metric']} at t={metrics['max_metric_timestamp']}")

        if metrics['save']:
            save_picture(grid, timestamp=t+1)
        
if __name__ == "__main__":
    main()

