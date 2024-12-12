from numpy import array, count_nonzero, unique
from scipy.ndimage import label
from scipy.signal import convolve2d

garden_layout = array([list(line.strip()) for line in open('inputs/day12_input.txt')])
results_by_plant = {}
plant_types = unique(garden_layout)

for plant_type in plant_types:
    results_by_plant[plant_type] = array([0, 0])
    labeled_regions, num_regions = label(garden_layout == plant_type)
    
    for region_index in range(num_regions):
        current_region = (labeled_regions == region_index + 1)

        horizontal_edges = count_nonzero(convolve2d(current_region, [[1, -1]]))
        vertical_edges = count_nonzero(convolve2d(current_region, [[1], [-1]]))
        diagonal_patterns = abs(convolve2d(current_region, [[-1, 1], [1, -1]]))

        results_by_plant[plant_type] += current_region.sum() * array([
            horizontal_edges + vertical_edges,
            diagonal_patterns.sum()
        ])

total = array([0, 0])
for plant_type, counts in results_by_plant.items():
    print(f"Plant type '{plant_type}': {counts}")
    total += counts

print(f"\nTotal: {total}")