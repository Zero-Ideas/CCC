#!/usr/bin/env python3
"""
Simple test script to demonstrate sector boundary visualization
"""

import random
import matplotlib.pyplot as plt
import numpy as np
from TSP import calculate_optimal_subdivision, subdivide_points_spatially

def visualize_sector_boundaries(n_points=500, area_size=25):
    """
    Create a simple visualization showing how points are divided into sectors
    """
    print(f"Testing sector visualization with {n_points} points in {area_size}x{area_size} area")

    # Generate random points
    random.seed(42)
    points = [(random.uniform(0, area_size), random.uniform(0, area_size)) for _ in range(n_points)]

    # Calculate optimal subdivision
    sectors_x, sectors_y, sector_width, sector_height, bounds = calculate_optimal_subdivision(
        points, max_nodes_per_sector=250
    )

    print(f"Optimal subdivision: {sectors_x}x{sectors_y} sectors")
    print(f"Sector size: {sector_width:.2f} x {sector_height:.2f}")

    # Subdivide points into sectors
    spatial_sectors = subdivide_points_spatially(
        points, sectors_x, sectors_y, sector_width, sector_height, bounds
    )

    print(f"Created {len(spatial_sectors)} sectors")
    for i, sector in enumerate(spatial_sectors):
        print(f"  Sector {i}: {len(sector)} points")

    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect('equal')

    # Draw sector boundaries
    min_x, max_x, min_y, max_y = bounds

    # Draw vertical grid lines
    for i in range(sectors_x + 1):
        x = min_x + i * sector_width
        ax.axvline(x=x, color='red', linestyle='--', linewidth=2, alpha=0.8)

    # Draw horizontal grid lines
    for j in range(sectors_y + 1):
        y = min_y + j * sector_height
        ax.axhline(y=y, color='red', linestyle='--', linewidth=2, alpha=0.8)

    # Color points by sector
    colors = plt.cm.Set3(np.linspace(0, 1, len(spatial_sectors)))

    for sector_idx, sector in enumerate(spatial_sectors):
        sector_points = [points[i] for i in sector]
        if sector_points:
            xs = [p[0] for p in sector_points]
            ys = [p[1] for p in sector_points]
            ax.scatter(xs, ys, c=[colors[sector_idx]], label=f'Sector {sector_idx} ({len(sector)} pts)',
                      s=30, alpha=0.7)

    # Add sector labels
    for i in range(sectors_x):
        for j in range(sectors_y):
            sector_center_x = min_x + (i + 0.5) * sector_width
            sector_center_y = min_y + (j + 0.5) * sector_height
            sector_num = i * sectors_y + j
            ax.text(sector_center_x, sector_center_y,
                   f'S{sector_num}',
                   fontsize=14, ha='center', va='center', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))

    ax.set_title(f'Spatial Subdivision: {n_points} points → {sectors_x}×{sectors_y} sectors\n'
                f'Max 250 points per sector, Actual max: {max(len(s) for s in spatial_sectors)}',
                fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    # Save to file instead of showing
    output_file = "sector_visualization.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    plt.close()

    return sectors_x, sectors_y, spatial_sectors

if __name__ == "__main__":
    # Test with different sizes
    print("=== Sector Visualization Test ===")

    # Example 1: 500 points
    visualize_sector_boundaries(500, 25)

    print("\n" + "="*50)

    # Example 2: 1000 points (like in user's request)
    visualize_sector_boundaries(1000, 25)

    print("\nVisualization complete! Check sector_visualization.png to see the results.")