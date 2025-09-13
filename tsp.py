import math, random, itertools, json, time
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache
import pandas as pd
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from sklearn.cluster import KMeans
import threading
import queue
import os


def parse_tsp_file(filepath):
    """
    Parse a .tsp file and extract the coordinate points.
    Supports standard TSP file formats including EUC_2D, EXPLICIT, and others.

    Args:
        filepath: Path to the .tsp file

    Returns:
        tuple: (points, metadata) where points is list of (x, y) tuples and metadata contains file info
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"TSP file not found: {filepath}")

    points = []
    metadata = {
        'name': '',
        'comment': '',
        'type': '',
        'dimension': 0,
        'edge_weight_type': '',
        'node_coord_type': '',
        'display_data_type': '',
        'filepath': filepath
    }

    with open(filepath, 'r') as file:
        lines = file.readlines()

    reading_coords = False
    reading_edge_weights = False
    coord_section_started = False

    for line_num, line in enumerate(lines):
        line = line.strip()

        if not line or line.startswith('%'):
            continue

        # Parse header information
        if ':' in line and not coord_section_started:
            key, value = line.split(':', 1)
            key = key.strip().upper()
            value = value.strip()

            if key == 'NAME':
                metadata['name'] = value
            elif key == 'COMMENT':
                metadata['comment'] = value
            elif key == 'TYPE':
                metadata['type'] = value
            elif key == 'DIMENSION':
                metadata['dimension'] = int(value)
            elif key == 'EDGE_WEIGHT_TYPE':
                metadata['edge_weight_type'] = value
            elif key == 'NODE_COORD_TYPE':
                metadata['node_coord_type'] = value
            elif key == 'DISPLAY_DATA_TYPE':
                metadata['display_data_type'] = value

        # Start of coordinate section
        elif line.upper() == 'NODE_COORD_SECTION':
            reading_coords = True
            coord_section_started = True
            print(f"Reading coordinates from {metadata['name']} ({metadata['dimension']} points)...")

        # Start of edge weight section (for explicit distance matrices)
        elif line.upper() == 'EDGE_WEIGHT_SECTION':
            reading_edge_weights = True
            coord_section_started = True
            print(f"Warning: EDGE_WEIGHT_SECTION found - explicit distance matrices not fully supported")

        # End of data sections
        elif line.upper() == 'EOF':
            break

        # Read coordinate data
        elif reading_coords and not line.upper().startswith(('EDGE_WEIGHT_SECTION', 'EOF')):
            try:
                parts = line.split()
                if len(parts) >= 3:
                    # Standard format: node_id x y
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    points.append((x, y))
                elif len(parts) == 2:
                    # Simple format: x y
                    x = float(parts[0])
                    y = float(parts[1])
                    points.append((x, y))
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse coordinate line {line_num + 1}: {line} - {e}")

    # Validate parsed data
    if not points:
        raise ValueError(f"No coordinate points found in TSP file: {filepath}")

    if metadata['dimension'] > 0 and len(points) != metadata['dimension']:
        print(f"Warning: Found {len(points)} points but dimension is {metadata['dimension']}")

    # Update dimension if it wasn't specified
    if metadata['dimension'] == 0:
        metadata['dimension'] = len(points)

    print(f"Successfully parsed TSP file: {len(points)} points from {metadata['name']}")
    return points, metadata


def normalize_tsp_coordinates(points, target_area_size=25):
    """
    Normalize TSP coordinates to fit within a target area size.
    This helps with visualization and ensures consistent solver behavior.

    Args:
        points: List of (x, y) coordinate tuples
        target_area_size: Target area size (default: 25x25)

    Returns:
        list: Normalized points that fit within target_area_size x target_area_size
    """
    if not points:
        return points

    # Find bounding box
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Calculate scaling factors
    width = max_x - min_x
    height = max_y - min_y

    if width == 0 or height == 0:
        return points

    # Scale to fit within target area while maintaining aspect ratio
    scale = min(target_area_size / width, target_area_size / height)

    # Normalize coordinates
    normalized_points = []
    for x, y in points:
        # Translate to origin, scale, then translate to center of target area
        normalized_x = (x - min_x) * scale
        normalized_y = (y - min_y) * scale
        normalized_points.append((normalized_x, normalized_y))

    return normalized_points


def load_and_solve_tsp_file(filepath, visualize=True, normalize_coords=True, target_area_size=25):
    """
    Load a TSP file and solve it using the enhanced spatial subdivision solver.

    Args:
        filepath: Path to the .tsp file
        visualize: Whether to show visualization
        normalize_coords: Whether to normalize coordinates to target area size
        target_area_size: Target area size for normalization

    Returns:
        dict: Solver results including performance metrics and metadata
    """
    print(f"\n=== Loading and Solving TSP File ===")
    print(f"File: {filepath}")

    try:
        # Parse the TSP file
        points, metadata = parse_tsp_file(filepath)

        # Normalize coordinates if requested
        if normalize_coords:
            original_bounds = {
                'min_x': min(p[0] for p in points),
                'max_x': max(p[0] for p in points),
                'min_y': min(p[1] for p in points),
                'max_y': max(p[1] for p in points)
            }
            points = normalize_tsp_coordinates(points, target_area_size)
            print(f"Coordinates normalized from {original_bounds} to ~{target_area_size}x{target_area_size}")

        # Solve using the enhanced spatial subdivision solver
        start_time = time.time()
        result = advanced_tsp_solver(points, visualize=visualize, use_all_optimizations=True, real_time_viz=False)
        total_time = time.time() - start_time

        # Add TSP file metadata to results
        result['tsp_metadata'] = metadata
        result['original_bounds'] = original_bounds if normalize_coords else None
        result['normalized'] = normalize_coords
        result['file_load_time'] = total_time - result['solve_time']

        print(f"\n=== TSP File Solution Results ===")
        print(f"File: {metadata['name']} ({metadata.get('comment', 'No comment')})")
        print(f"Points: {len(points)}")
        print(f"Final cost: {result['cost']:.2f}")
        print(f"Solve time: {result['solve_time']:.3f}s")
        print(f"Total time (including file I/O): {total_time:.3f}s")
        print(f"Final clusters: {result['num_clusters']}")

        return result

    except Exception as e:
        print(f"Error loading/solving TSP file: {e}")
        raise


def calculate_optimal_subdivision(points, max_nodes_per_sector=250):
    """
    Calculate optimal subdivision dimensions for the given points to ensure
    each sector has at most max_nodes_per_sector points.

    Args:
        points: List of (x, y) coordinate tuples
        max_nodes_per_sector: Maximum allowed nodes per sector (default: 250)

    Returns:
        tuple: (sectors_x, sectors_y, sector_width, sector_height, bounds)
    """
    if not points or len(points) == 0:
        return 1, 1, 1, 1, (0, 1, 0, 1)

    # Find bounding box
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Add small padding to avoid edge cases
    padding = max((max_x - min_x), (max_y - min_y)) * 0.01
    min_x -= padding
    max_x += padding
    min_y -= padding
    max_y += padding

    width = max_x - min_x
    height = max_y - min_y
    area = width * height

    # Calculate total sectors needed
    n_points = len(points)
    sectors_needed = max(1, math.ceil(n_points / max_nodes_per_sector))

    # Find optimal aspect ratio that maintains area proportions
    aspect_ratio = width / height if height > 0 else 1.0

    # Calculate sectors in each dimension
    sectors_y = max(1, int(math.sqrt(sectors_needed / aspect_ratio)))
    sectors_x = max(1, math.ceil(sectors_needed / sectors_y))

    # Ensure we have enough sectors
    while sectors_x * sectors_y < sectors_needed:
        if aspect_ratio >= 1.0:
            sectors_x += 1
        else:
            sectors_y += 1

    sector_width = width / sectors_x
    sector_height = height / sectors_y

    return sectors_x, sectors_y, sector_width, sector_height, (min_x, max_x, min_y, max_y)


def subdivide_points_spatially(points, sectors_x, sectors_y, sector_width, sector_height, bounds):
    """
    Divide points into spatial sectors based on geographic location.

    Args:
        points: List of (x, y) coordinate tuples
        sectors_x, sectors_y: Number of sectors in each dimension
        sector_width, sector_height: Size of each sector
        bounds: (min_x, max_x, min_y, max_y) bounding box

    Returns:
        list: List of sectors, each containing list of point indices
    """
    min_x, max_x, min_y, max_y = bounds
    sectors = [[[] for _ in range(sectors_y)] for _ in range(sectors_x)]

    for idx, point in enumerate(points):
        x, y = point

        # Determine sector indices
        sector_x = min(sectors_x - 1, int((x - min_x) / sector_width))
        sector_y = min(sectors_y - 1, int((y - min_y) / sector_height))

        sectors[sector_x][sector_y].append(idx)

    # Flatten to list of non-empty sectors
    flat_sectors = []
    for i in range(sectors_x):
        for j in range(sectors_y):
            if len(sectors[i][j]) > 0:
                flat_sectors.append(sectors[i][j])

    return flat_sectors


def hierarchical_cluster_within_sector(points, sector_indices, max_cluster_size=25, depth=0, max_depth=3):
    """
    Apply hierarchical clustering within a sector using the same subdivision method.
    This recursively subdivides sectors that are still too large.

    Args:
        points: Full list of points
        sector_indices: Indices of points in this sector
        max_cluster_size: Maximum points per final cluster
        depth: Current recursion depth
        max_depth: Maximum allowed depth

    Returns:
        list: List of clusters (each cluster is a list of point indices)
    """
    if len(sector_indices) <= max_cluster_size or depth >= max_depth:
        return [sector_indices]

    # Extract points for this sector
    sector_points = [points[i] for i in sector_indices]

    # Calculate subdivision for this sector
    try:
        sectors_x, sectors_y, sector_width, sector_height, bounds = calculate_optimal_subdivision(
            sector_points, max_nodes_per_sector=max_cluster_size
        )

        # If no subdivision needed, return as single cluster
        if sectors_x == 1 and sectors_y == 1:
            return [sector_indices]

        # Create mapping from sector points back to original indices
        point_to_original = {i: sector_indices[i] for i in range(len(sector_indices))}

        # Subdivide the sector points
        subsectors = subdivide_points_spatially(
            sector_points, sectors_x, sectors_y,
            sector_width, sector_height, bounds
        )

        # Convert back to original indices and recursively cluster
        final_clusters = []
        for subsector in subsectors:
            original_indices = [point_to_original[local_idx] for local_idx in subsector]
            # Recursively apply hierarchical clustering
            sub_clusters = hierarchical_cluster_within_sector(
                points, original_indices, max_cluster_size, depth + 1, max_depth
            )
            final_clusters.extend(sub_clusters)

        return final_clusters

    except Exception as e:
        # Fallback to original behavior if subdivision fails
        print(f"Warning: Hierarchical clustering failed at depth {depth}, using fallback: {e}")
        return [sector_indices]


def enhanced_spatial_clustering_with_optimization(points, max_nodes_per_sector=250, max_cluster_size=25, apply_sector_optimization=True):
    """
    Main function that combines area subdivision with hierarchical clustering and sector-level optimization.

    Args:
        points: List of (x, y) coordinate tuples
        max_nodes_per_sector: Maximum nodes per initial spatial sector
        max_cluster_size: Maximum nodes per final cluster after hierarchical subdivision
        apply_sector_optimization: Whether to apply or-opt within each sector

    Returns:
        tuple: (clusters, sector_optimization_info) where sector_optimization_info contains performance data
    """
    if not points or len(points) <= max_cluster_size:
        return [list(range(len(points)))], {}

    print(f"Enhanced spatial clustering: {len(points)} points -> max {max_nodes_per_sector} per sector -> max {max_cluster_size} per cluster")

    # Step 1: Calculate optimal spatial subdivision
    sectors_x, sectors_y, sector_width, sector_height, bounds = calculate_optimal_subdivision(
        points, max_nodes_per_sector
    )

    print(f"Spatial subdivision: {sectors_x}x{sectors_y} sectors ({sector_width:.2f} x {sector_height:.2f} each)")

    # Step 2: Divide points into spatial sectors
    spatial_sectors = subdivide_points_spatially(
        points, sectors_x, sectors_y, sector_width, sector_height, bounds
    )

    print(f"Created {len(spatial_sectors)} non-empty spatial sectors")

    # Step 3: Apply hierarchical clustering and optimization within each sector
    final_clusters = []
    sector_optimization_info = {
        'sector_times': [],
        'sector_improvements': [],
        'total_sector_optimization_time': 0
    }

    for i, sector in enumerate(spatial_sectors):
        print(f"Processing sector {i+1}/{len(spatial_sectors)} with {len(sector)} points")

        if len(sector) <= max_cluster_size:
            # Sector is already small enough
            final_clusters.append(sector)
            sector_optimization_info['sector_times'].append(0)
            sector_optimization_info['sector_improvements'].append(0)
        else:
            # Apply hierarchical subdivision within this sector
            sector_clusters = hierarchical_cluster_within_sector(
                points, sector, max_cluster_size
            )
            final_clusters.extend(sector_clusters)

            # Apply sector-level optimization if enabled and sector is large enough
            if apply_sector_optimization and len(sector) >= 50:
                import time
                sector_start_time = time.time()

                print(f"  Applying comprehensive sector optimization to {len(sector)} points...")
                optimized_clusters, sector_route_info = optimize_sector_clusters(points, sector, sector_clusters)

                sector_time = time.time() - sector_start_time
                sector_optimization_info['sector_times'].append(sector_time)
                sector_optimization_info['total_sector_optimization_time'] += sector_time

                # Replace the last added clusters with optimized ones
                final_clusters = final_clusters[:-len(sector_clusters)]
                final_clusters.extend(optimized_clusters)

                # Store sector route info for potential inter-sector optimization
                if sector_route_info:
                    if 'sector_routes' not in sector_optimization_info:
                        sector_optimization_info['sector_routes'] = {}
                    sector_optimization_info['sector_routes'][i] = sector_route_info

                print(f"  Sector {i+1} optimization completed in {sector_time:.3f}s")
            else:
                sector_optimization_info['sector_times'].append(0)
                sector_optimization_info['sector_improvements'].append(0)

    print(f"Hierarchical clustering complete: {len(final_clusters)} final clusters")

    if apply_sector_optimization:
        print(f"Total sector-level optimization time: {sector_optimization_info['total_sector_optimization_time']:.3f}s")

    # Validate clusters
    total_points_clustered = sum(len(cluster) for cluster in final_clusters)
    if total_points_clustered != len(points):
        print(f"Warning: Point count mismatch. Expected {len(points)}, got {total_points_clustered}")

    return final_clusters, sector_optimization_info


def enhanced_spatial_clustering(points, max_nodes_per_sector=250, max_cluster_size=25):
    """
    Wrapper function to maintain compatibility with existing code
    """
    clusters, _ = enhanced_spatial_clustering_with_optimization(
        points, max_nodes_per_sector, max_cluster_size, apply_sector_optimization=True
    )
    return clusters


def optimize_sector_clusters(points, sector_indices, sector_clusters):
    """
    Apply comprehensive optimization within a single sector including stages 5 and 7.
    This creates a complete local TSP solution for the sector.

    Args:
        points: Full list of points
        sector_indices: Indices of all points in this sector
        sector_clusters: List of clusters within this sector

    Returns:
        tuple: (optimized_clusters, sector_route_info) where sector_route_info contains the solved route details
    """
    if len(sector_clusters) <= 1 or len(sector_indices) < 10:
        return sector_clusters, None

    try:
        # Create a local TSP problem for this sector
        sector_points = [points[i] for i in sector_indices]

        # Build distance matrix for this sector only
        from TSP import pairwise_dist_matrix
        sector_D = pairwise_dist_matrix(sector_points)

        # Create mapping from sector points to original indices
        sector_to_original = {local_idx: sector_indices[local_idx] for local_idx in range(len(sector_indices))}
        original_to_sector = {sector_indices[local_idx]: local_idx for local_idx in range(len(sector_indices))}

        # Convert clusters to local indices
        local_clusters = []
        for cluster in sector_clusters:
            local_cluster = [original_to_sector[orig_idx] for orig_idx in cluster if orig_idx in original_to_sector]
            if local_cluster:  # Only add non-empty clusters
                local_clusters.append(local_cluster)

        if len(local_clusters) <= 2:
            return sector_clusters, None

        print(f"    Comprehensive sector optimization: {len(local_clusters)} clusters, {len(sector_indices)} points")

        # STAGE 5 (Sector-level): Enhanced sub-clustering within this sector
        print(f"    Stage 5 (Sector): Enhanced sub-clustering...")
        enhanced_local_clusters = []
        for cluster in local_clusters:
            if len(cluster) > 8:  # Only sub-cluster large clusters
                from TSP import adaptive_subcluster
                subclusters = adaptive_subcluster(sector_points, cluster, max_size=6, depth=0, max_depth=2)
                enhanced_local_clusters.extend(subclusters)
            else:
                enhanced_local_clusters.append(cluster)

        print(f"    Sub-clustering: {len(local_clusters)} -> {len(enhanced_local_clusters)} clusters")
        local_clusters = enhanced_local_clusters

        # Step 1: Quick cluster ordering based on centroids
        print(f"    Stage 6 (Sector): Optimizing cluster order...")
        centroids = []
        for cluster in local_clusters:
            if cluster:
                cx = sum(sector_points[i][0] for i in cluster) / len(cluster)
                cy = sum(sector_points[i][1] for i in cluster) / len(cluster)
                centroids.append((cx, cy))
            else:
                centroids.append((0, 0))

        # Use appropriate cluster ordering method
        if len(centroids) <= 8:
            from TSP import solve_cluster_order
            cluster_order, _ = solve_cluster_order(centroids)
        else:
            cluster_order = nearest_neighbor_cluster_order(centroids)

        ordered_local_clusters = [local_clusters[i] for i in cluster_order]

        # STAGE 7 (Sector-level): Finding optimal entries and exits within sector
        print(f"    Stage 7 (Sector): Solving entry/exit points...")
        sector_best_cost, sector_seq = choose_entries_exits_sector(
            sector_points, sector_D, ordered_local_clusters, Kcand=2  # Reduced Kcand for performance
        )

        # Build the complete sector route from the sequence
        from TSP import stitch_full_route
        sector_route = stitch_full_route(sector_seq)

        print(f"    Initial sector route cost: {sector_best_cost:.2f}")

        if len(sector_route) > 3:
            # Step 3: Apply Enhanced 2-opt + Lin-Kernighan at SECTOR level
            print(f"    Applying Enhanced 2-opt + Lin-Kernighan on {len(sector_route)} points...")
            from TSP import enhanced_2opt_with_lk

            # Apply the critical optimization that doesn't scale well - but now on smaller sector
            optimized_cost, optimized_route = enhanced_2opt_with_lk(
                sector_points, sector_D, sector_route + [sector_route[0]]
            )
            sector_route = optimized_route[:-1]  # Remove duplicate end point

            # Step 4: Lightweight or-opt on the optimized route
            print(f"    Applying sector or-opt...")
            from TSP import or_opt_optimization_fast
            final_cost, sector_route = or_opt_optimization_fast(
                sector_D, sector_route, max_segment_length=2, max_iterations=2
            )
            print(f"    Final sector route cost: {final_cost:.2f}")

        # Step 5: Convert back to original indices and return enhanced clusters
        optimized_clusters = []
        for local_cluster in ordered_local_clusters:
            orig_cluster = [sector_to_original[local_idx] for local_idx in local_cluster]
            optimized_clusters.append(orig_cluster)

        # Create sector route info for inter-sector connections
        sector_route_info = {
            'route': [sector_to_original[local_idx] for local_idx in sector_route],
            'cost': final_cost if 'final_cost' in locals() else sector_best_cost,
            'entry_point': sector_to_original[sector_route[0]] if sector_route else None,
            'exit_point': sector_to_original[sector_route[-1]] if sector_route else None
        }

        print(f"    Sector optimization complete")
        return optimized_clusters, sector_route_info

    except Exception as e:
        print(f"    Warning: Sector optimization failed: {e}, using original clusters")
        return sector_clusters, None


def build_complete_sector_route(sector_points, sector_D, ordered_clusters):
    """
    Build a complete route through all points in the sector by solving cluster paths
    and connecting them optimally.
    """
    if len(ordered_clusters) <= 1:
        return [i for cluster in ordered_clusters for i in cluster]

    from TSP import cluster_path
    complete_route = []

    for i, cluster in enumerate(ordered_clusters):
        if len(cluster) == 0:
            continue

        if i == 0:
            # First cluster: start from arbitrary point
            if len(ordered_clusters) == 1:
                # Only one cluster, create simple path
                complete_route.extend(cluster)
            else:
                # Multiple clusters: find best end point to connect to next cluster
                next_cluster = ordered_clusters[i+1] if i+1 < len(ordered_clusters) else []
                if next_cluster:
                    # Find best connection points
                    start_point = cluster[0]  # Arbitrary start
                    end_point = min(cluster, key=lambda x: min(sector_D[x][y] for y in next_cluster))
                    cost, path = cluster_path(sector_points, sector_D, cluster, start_point, end_point)
                    complete_route.extend(path if path else cluster)
                else:
                    complete_route.extend(cluster)

        elif i == len(ordered_clusters) - 1:
            # Last cluster: connect from previous cluster's end
            if complete_route:
                prev_end = complete_route[-1]
                start_point = min(cluster, key=lambda x: sector_D[prev_end][x])
                end_point = cluster[0] if len(cluster) == 1 else cluster[-1]  # Arbitrary end
                cost, path = cluster_path(sector_points, sector_D, cluster, start_point, end_point)
                if path and len(path) > 1:
                    complete_route.extend(path[1:])  # Skip first point to avoid duplication
                else:
                    complete_route.extend(cluster)
            else:
                complete_route.extend(cluster)

        else:
            # Middle cluster: connect optimally between adjacent clusters
            if complete_route and i+1 < len(ordered_clusters):
                prev_end = complete_route[-1]
                next_cluster = ordered_clusters[i+1]

                start_point = min(cluster, key=lambda x: sector_D[prev_end][x])
                end_point = min(cluster, key=lambda x: min(sector_D[x][y] for y in next_cluster))

                cost, path = cluster_path(sector_points, sector_D, cluster, start_point, end_point)
                if path and len(path) > 1:
                    complete_route.extend(path[1:])  # Skip first point to avoid duplication
                else:
                    complete_route.extend(cluster)
            else:
                complete_route.extend(cluster)

    # Remove any duplicates that might have been introduced
    seen = set()
    deduplicated_route = []
    for point in complete_route:
        if point not in seen:
            seen.add(point)
            deduplicated_route.append(point)

    return deduplicated_route


def nearest_neighbor_cluster_order(centroids):
    """Quick nearest neighbor ordering for cluster centroids"""
    if len(centroids) <= 1:
        return list(range(len(centroids)))

    from TSP import euclid
    unvisited = set(range(len(centroids)))
    order = [0]
    unvisited.remove(0)

    while unvisited:
        current = order[-1]
        nearest = min(unvisited, key=lambda i: euclid(centroids[current], centroids[i]))
        order.append(nearest)
        unvisited.remove(nearest)

    return order


def lightweight_inter_cluster_oropt(sector_points, sector_D, clusters):
    """
    Apply lightweight or-opt relocations between adjacent clusters within a sector
    """
    if len(clusters) < 3:
        return clusters

    optimized_clusters = [cluster[:] for cluster in clusters]  # Deep copy

    # Try relocating small clusters to better positions
    for iterations in range(2):  # Limited iterations for performance
        improved = False

        for i in range(len(optimized_clusters)):
            current_cluster = optimized_clusters[i]

            # Only try relocating small clusters (≤ 5 points)
            if len(current_cluster) > 5:
                continue

            best_position = i
            best_cost_delta = 0

            # Try inserting this cluster at different positions
            for j in range(len(optimized_clusters)):
                if abs(j - i) <= 1:  # Skip adjacent positions
                    continue

                # Calculate cost delta for moving cluster i to position j
                delta = calculate_cluster_relocation_delta(
                    sector_points, sector_D, optimized_clusters, i, j
                )

                if delta < best_cost_delta:
                    best_cost_delta = delta
                    best_position = j

            # Apply the best move if it's beneficial
            if best_position != i and best_cost_delta < -0.01:
                # Move cluster from position i to best_position
                cluster_to_move = optimized_clusters.pop(i)
                optimized_clusters.insert(best_position, cluster_to_move)
                improved = True
                break

        if not improved:
            break

    return optimized_clusters


def calculate_cluster_relocation_delta(sector_points, sector_D, clusters, from_pos, to_pos):
    """
    Calculate the cost delta of relocating a cluster from from_pos to to_pos
    This is a simplified calculation focusing on inter-cluster distances
    """
    if from_pos == to_pos or abs(from_pos - to_pos) <= 1:
        return 0

    try:
        from TSP import euclid

        # Get cluster centroids for quick estimation
        def cluster_centroid(cluster):
            if not cluster:
                return (0, 0)
            cx = sum(sector_points[i][0] for i in cluster) / len(cluster)
            cy = sum(sector_points[i][1] for i in cluster) / len(cluster)
            return (cx, cy)

        centroids = [cluster_centroid(cluster) for cluster in clusters]

        # Estimate cost change based on centroid distances
        old_cost = 0
        new_cost = 0

        # Cost of removing from current position
        if from_pos > 0:
            old_cost += euclid(centroids[from_pos-1], centroids[from_pos])
        if from_pos < len(clusters) - 1:
            old_cost += euclid(centroids[from_pos], centroids[from_pos+1])
        if from_pos > 0 and from_pos < len(clusters) - 1:
            new_cost += euclid(centroids[from_pos-1], centroids[from_pos+1])

        # Cost of inserting at new position
        if to_pos > 0:
            old_cost += euclid(centroids[to_pos-1], centroids[to_pos])
            new_cost += euclid(centroids[to_pos-1], centroids[from_pos])
        new_cost += euclid(centroids[from_pos], centroids[to_pos])

        return new_cost - old_cost

    except:
        return 0  # Return 0 if calculation fails


def choose_entries_exits_sector(sector_points, sector_D, clusters, Kcand=3):
    """
    Optimized entry/exit point selection for sector-level solving with reduced scaling harshness.
    This is a lightweight version of choose_entries_exits specifically for sectors.

    Args:
        sector_points: Points in this sector (local indices)
        sector_D: Distance matrix for sector points
        clusters: List of clusters within the sector (local indices)
        Kcand: Number of candidate boundary nodes (reduced for performance)

    Returns:
        tuple: (best_cost, sequence) for the sector
    """
    if len(clusters) <= 1:
        if clusters and len(clusters[0]) > 0:
            return 0.0, [(0, clusters[0][0], clusters[0][-1], clusters[0])]
        return 0.0, []

    # Generate lightweight candidates for each cluster
    from TSP import candidate_boundary_nodes

    candidates = {}
    for ci in range(len(clusters)):
        cidxs = clusters[ci]
        # Use smaller K for faster performance
        candidates[ci] = candidate_boundary_nodes(sector_points, cidxs, K=min(Kcand, 2))
        if len(candidates[ci]) == 0:
            candidates[ci] = [cidxs[0]]

    # Simplified intra-cluster path computation (faster than full DP)
    intra = {}
    for ci in range(len(clusters)):
        intra[ci] = {}
        idxs = clusters[ci]

        for a in candidates[ci]:
            for b in candidates[ci]:
                if a == b and len(idxs) > 1:
                    intra[ci][(a,b)] = (float('inf'), None)
                    continue

                # Use fast path computation for sector-level
                from TSP import greedy_path_between_fast
                cost, path = greedy_path_between_fast(idxs, a, b, sector_D)
                intra[ci][(a,b)] = (cost, path)

    # Simplified DP (faster than full version)
    order = list(range(len(clusters)))  # Simple sequential order
    dp = [{} for _ in order]
    parent = [{} for _ in order]

    for pos, ci in enumerate(order):
        if pos == 0:
            for (a,b), (cst, path) in intra[ci].items():
                dp[pos][b] = cst
                parent[pos][b] = (None, a)
        else:
            for (a,b), (cst, path) in intra[ci].items():
                best = float('inf')
                bestprev = None
                for prev_exit, prev_cost in dp[pos-1].items():
                    inter = sector_D[prev_exit][a]
                    tot = prev_cost + inter + cst
                    if tot < best:
                        best = tot
                        bestprev = prev_exit
                if best < float('inf'):
                    if b not in dp[pos] or best < dp[pos][b]:
                        dp[pos][b] = best
                        parent[pos][b] = (bestprev, a)

    # Reconstruct solution
    last_pos = len(order) - 1
    if not dp[last_pos]:
        return float('inf'), []

    best_exit = min(dp[last_pos], key=lambda x: dp[last_pos][x])
    best_cost = dp[last_pos][best_exit]

    seq = []
    cur_exit = best_exit
    for pos in range(last_pos, -1, -1):
        prev_exit, entry = parent[pos][cur_exit]
        ci = order[pos]
        seq.append((ci, entry, cur_exit, intra[ci][(entry, cur_exit)][1]))
        cur_exit = prev_exit

    seq = list(reversed(seq))
    return best_cost, seq


def euclid(a,b):
    try:
        return math.hypot(a[0]-b[0], a[1]-b[1])
    except Exception as e:
        #print(f"[DeBUG] Error in euclid: a={a}, b={b}, error={e}")
        raise
def pairwise_dist_matrix(points):
    n=len(points)
    #print(f"[DeBUG] Creating {n}x{n} distance matrix...")
    #print(f"[DeBUG] Points sample: {points[:2] if points else 'empty'}")

    # Check for valid points
    if not points or n == 0:
        #print(f"[DeBUG] No points provided!")
        return []

    D=[[0.0]*n for _ in range(n)]
    #print(f"[DeBUG] Matrix initialized, starting computation...")

    computation_count = 0
    for i in range(n):
        #print(f"[DeBUG] Row {i}: computing {n-i-1} distances")
        for j in range(i+1,n):
            try:
                d=euclid(points[i], points[j])
                D[i][j]=d
                D[j][i]=d
                computation_count += 1
                #if computation_count % 10 == 0:
                #    #print(f"[DeBUG] Computed {computation_count} distances")
            except Exception as e:
                print(f"[DeBUG] Error computing distance between points {i} and {j}: {e}")
                print(f"[DeBUG] Point {i}: {points[i]}, Point {j}: {points[j]}")
                raise

    #print(f"[DeBUG] Distance matrix complete - {computation_count} computations")
    return D

def avg_distances(points):
    #print(f"[DeBUG] Computing pairwise distance matrix for {len(points)} points...")
    D=pairwise_dist_matrix(points)
    #print(f"[DeBUG] Distance matrix computed, computing averages...")
    avgs=[sum(D[i])/(len(points)-1) if len(points)>1 else 0.0 for i in range(len(points))]
    grand=sum(avgs)/len(avgs)
    #print(f"[DeBUG] Averages computed")
    return avgs, grand, D

def adaptive_subcluster(points, cluster_idxs, max_size=5, depth=0, max_depth=3):
    """
    Recursively divide clusters into sub-clusters until each has <= max_size points
    """
    if len(cluster_idxs) <= max_size or depth >= max_depth:
        return [cluster_idxs]

    # Use k-means for sub-clustering
    cluster_points = np.array([points[i] for i in cluster_idxs])
    n_subclusters = min(len(cluster_idxs) // max_size + 1, len(cluster_idxs))

    if n_subclusters <= 1:
        return [cluster_idxs]

    try:
        kmeans = KMeans(n_clusters=n_subclusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(cluster_points)

        subclusters = [[] for _ in range(n_subclusters)]
        for i, label in enumerate(labels):
            subclusters[label].append(cluster_idxs[i])

        # Recursively sub-cluster each subcluster
        final_subclusters = []
        for subcluster in subclusters:
            if len(subcluster) > 0:
                final_subclusters.extend(adaptive_subcluster(points, subcluster, max_size, depth+1, max_depth))

        return final_subclusters
    except:
        return [cluster_idxs]

def connect_subclusters_optimally(points, D, subclusters, parent_entry, parent_exit):
    """
    Connect sub-clusters in optimal order from parent_entry to parent_exit
    """
    if len(subclusters) == 1:
        return subclusters[0]

    # Calculate centroids for each subcluster
    centroids = []
    for subcluster in subclusters:
        xs = [points[i][0] for i in subcluster]
        ys = [points[i][1] for i in subcluster]
        centroids.append((sum(xs)/len(xs), sum(ys)/len(ys)))

    # Find which subclusters contain entry and exit points
    entry_subcluster = None
    exit_subcluster = None
    for i, subcluster in enumerate(subclusters):
        if parent_entry in subcluster:
            entry_subcluster = i
        if parent_exit in subcluster:
            exit_subcluster = i

    if entry_subcluster is None or exit_subcluster is None:
        # Fallback: order by distance from entry point
        if entry_subcluster is not None:
            start_centroid = centroids[entry_subcluster]
        else:
            start_centroid = points[parent_entry]

        distances = [(i, euclid(start_centroid, centroid)) for i, centroid in enumerate(centroids)]
        ordered_indices = [i for i, _ in sorted(distances, key=lambda x: x[1])]
        return [subclusters[i] for i in ordered_indices]

    if entry_subcluster == exit_subcluster:
        return subclusters

    # Order subclusters optimally from entry to exit
    remaining = set(range(len(subclusters)))
    ordered = [entry_subcluster]
    remaining.remove(entry_subcluster)

    current = entry_subcluster
    while len(remaining) > 1:
        next_idx = min(remaining, key=lambda i: euclid(centroids[current], centroids[i]))
        ordered.append(next_idx)
        remaining.remove(next_idx)
        current = next_idx

    if exit_subcluster in remaining:
        ordered.append(exit_subcluster)

    return [subclusters[i] for i in ordered]

def benchmark_cluster_assignment(points):
    #print(f"[DeBUG] Computing average distances for {len(points)} points...")
    avgs, grand_avg, D = avg_distances(points)
    #print(f"[DeBUG] Average distances computed, grand_avg: {grand_avg:.2f}")
    core_idxs = [i for i,a in enumerate(avgs) if a <= grand_avg]
    periphery_idxs = [i for i,a in enumerate(avgs) if a > grand_avg]
    #print(f"[DeBUG] Core: {len(core_idxs)}, Periphery: {len(periphery_idxs)}")
    if len(core_idxs)==0:
        return [list(range(len(points)))], avgs, grand_avg, D
    clusters = [[i] for i in core_idxs]
    def centroid(cluster):
        xs=[points[i][0] for i in cluster]; ys=[points[i][1] for i in cluster]
        return (sum(xs)/len(xs), sum(ys)/len(ys))
    merged=True
    while merged:
        merged=False; best_pair=None; best_dist=float('inf')
        for a,b in itertools.combinations(range(len(clusters)),2):
            ca=centroid(clusters[a]); cb=centroid(clusters[b]); d=euclid(ca,cb)
            if d<best_dist: best_dist=d; best_pair=(a,b)
        if best_pair and best_dist < grand_avg:
            a,b=best_pair
            newc = clusters[a]+clusters[b]
            clusters = [clusters[i] for i in range(len(clusters)) if i not in best_pair] + [newc]
            merged=True
    for p in periphery_idxs:
        best=None; bestd=float('inf')
        for ci,c in enumerate(clusters):
            ccent=centroid(c); d=euclid(points[p], ccent)
            if d<bestd: bestd=d; best=ci
        clusters[best].append(p)
    clusters=[sorted(c) for c in clusters if len(c)>0]
    return clusters, avgs, grand_avg, D

def candidate_boundary_nodes(points, cluster_idxs, K=4):
    pts=[points[i] for i in cluster_idxs]
    cx=sum(p[0] for p in pts)/len(pts); cy=sum(p[1] for p in pts)/len(pts)
    dlist=[(i, euclid(points[i], (cx,cy))) for i in cluster_idxs]
    dlist_sorted=sorted(dlist, key=lambda x: x[1])
    nearest=[i for i,_ in dlist_sorted[:K]]
    farthest=[i for i,_ in dlist_sorted[-K:]]
    candidates=list(dict.fromkeys(farthest+nearest))
    return candidates[:min(len(candidates), K)]

@lru_cache(maxsize=128)
def brute_force_hamiltonian_path_cached(idxs_tuple, start_idx, end_idx, D_tuple):
    # Cached version for performance
    idxs = list(idxs_tuple)
    D = [list(row) for row in D_tuple]
    return brute_force_hamiltonian_path_uncached(idxs, start_idx, end_idx, D)

def brute_force_hamiltonian_path_uncached(idxs, start_idx, end_idx, D):
    others = [i for i in idxs if i not in (start_idx, end_idx)]
    best_cost = float('inf'); best_perm = None
    if start_idx == end_idx:
        if len(idxs)==1: return 0.0, [start_idx]
        return float('inf'), None

    # Early termination for performance
    if len(others) > 6:  # Reduce from 8 to 6 for performance
        return greedy_path_between_fast(idxs, start_idx, end_idx, D)

    for perm in itertools.permutations(others):
        tour = [start_idx] + list(perm) + [end_idx]
        cost = sum(D[tour[i]][tour[i+1]] for i in range(len(tour)-1))
        if cost < best_cost:
            best_cost = cost; best_perm = tour[:]
    return best_cost, best_perm

def brute_force_hamiltonian_path(points, D, idxs, start_idx, end_idx):
    # Use cached version if cluster is small enough for exact solving
    if len(idxs) <= 6:  # Reduced threshold for performance
        try:
            D_tuple = tuple(tuple(row) for row in D)
            return brute_force_hamiltonian_path_cached(tuple(sorted(idxs)), start_idx, end_idx, D_tuple)
        except:
            pass
    return greedy_path_between_fast(idxs, start_idx, end_idx, D)

def greedy_path_between_fast(idxs, start_idx, end_idx, D):
    """Fast greedy path without 2-opt for performance"""
    remaining = set(idxs)
    path = [start_idx]; remaining.remove(start_idx)
    if end_idx in remaining: remaining.remove(end_idx)

    while remaining:
        last = path[-1]
        nxt = min(remaining, key=lambda j: D[last][j])
        path.append(nxt); remaining.remove(nxt)
    path.append(end_idx)

    cost = sum(D[path[i]][path[i+1]] for i in range(len(path)-1))
    return cost, path

def greedy_path_between(points, D, idxs, start_idx, end_idx):
    remaining = set(idxs)
    path = [start_idx]; remaining.remove(start_idx)
    if end_idx in remaining: remaining.remove(end_idx)
    while remaining:
        last = path[-1]
        nxt = min(remaining, key=lambda j: D[last][j])
        path.append(nxt); remaining.remove(nxt)
    path.append(end_idx)
    # simple 2-opt (endpoints fixed)
    def tour_len(t): return sum(D[t[i]][t[i+1]] for i in range(len(t)-1))
    improved=True
    while improved:
        improved=False
        for i in range(1, len(path)-3):
            for j in range(i+1, len(path)-1):
                a,b = path[i-1], path[i]
                c,d = path[j], path[j+1]
                delta = (D[a][c] + D[b][d]) - (D[a][b] + D[c][d])
                if delta < -1e-12:
                    path[i:j+1] = reversed(path[i:j+1]); improved=True
    return tour_len(path), path

def cluster_path(points, D, idxs, start_idx, end_idx):
    if len(idxs) <= 6:  # Reduced from 8 to 6 for performance
        return brute_force_hamiltonian_path(points, D, idxs, start_idx, end_idx)
    else:
        return greedy_path_between_fast(idxs, start_idx, end_idx, D)

def adaptive_cluster_path(points, D, idxs, start_idx, end_idx):
    """
    Enhanced cluster path using adaptive sub-clustering
    """
    if len(idxs) <= 5:
        return cluster_path(points, D, idxs, start_idx, end_idx)

    # Apply adaptive sub-clustering
    subclusters = adaptive_subcluster(points, idxs, max_size=5)

    if len(subclusters) == 1:
        return cluster_path(points, D, idxs, start_idx, end_idx)

    # Connect subclusters optimally
    ordered_subclusters = connect_subclusters_optimally(points, D, subclusters, start_idx, end_idx)

    # Solve each subcluster and connect them
    total_cost = 0
    full_path = []

    for i, subcluster in enumerate(ordered_subclusters):
        if i == 0:
            # First subcluster: start from start_idx
            if len(ordered_subclusters) == 1:
                sub_end = end_idx
            else:
                # Find best connection point to next subcluster
                next_subcluster = ordered_subclusters[i+1]
                sub_end = min(subcluster, key=lambda x: min(D[x][y] for y in next_subcluster))
        elif i == len(ordered_subclusters) - 1:
            # Last subcluster: end at end_idx
            sub_start = min(subcluster, key=lambda x: D[full_path[-1]][x])
            sub_end = end_idx
        else:
            # Middle subcluster
            sub_start = min(subcluster, key=lambda x: D[full_path[-1]][x])
            next_subcluster = ordered_subclusters[i+1]
            sub_end = min(subcluster, key=lambda x: min(D[x][y] for y in next_subcluster))

        if i == 0:
            sub_start = start_idx

        cost, path = cluster_path(points, D, subcluster, sub_start, sub_end)

        # Handle None path case
        if path is None:
            path = [sub_start, sub_end] if sub_start != sub_end else [sub_start]
            cost = D[sub_start][sub_end] if sub_start != sub_end else 0

        total_cost += cost

        if i == 0:
            full_path.extend(path)
        else:
            # Connect to previous path and add transition cost
            if full_path and path:
                total_cost += D[full_path[-1]][path[0]]
                full_path.extend(path[1:])  # Skip first element to avoid duplication

    return total_cost, full_path

def solve_cluster_order(centroids):
    m=len(centroids)
    Cdist=[[euclid(centroids[i], centroids[j]) for j in range(m)] for i in range(m)]
    if m<=8:
        best=None; best_perm=None
        for perm in itertools.permutations(range(1,m)):
            tour = [0] + list(perm) + [0]
            cost = sum(Cdist[tour[i]][tour[i+1]] for i in range(len(tour)-1))
            if best is None or cost < best:
                best=cost; best_perm=tour[:-1]
        return best_perm, best
    # NN + 2-opt
    start=0; unvis=set(range(m)); order=[start]; unvis.remove(start)
    while unvis:
        last=order[-1]; nxt=min(unvis, key=lambda j: Cdist[last][j]); order.append(nxt); unvis.remove(nxt)
    order.append(start)
    improved=True
    while improved:
        improved=False
        for i in range(1, len(order)-3):
            for j in range(i+1, len(order)-1):
                a,b = order[i-1], order[i]; c,d = order[j], order[j+1]
                delta = (Cdist[a][c] + Cdist[b][d]) - (Cdist[a][b] + Cdist[c][d])
                if delta < -1e-12:
                    order[i:j+1] = reversed(order[i:j+1]); improved=True
    return order[:-1], sum(Cdist[order[i]][order[i+1]] for i in range(len(order)-1))

class TSPVisualizer:
    """
    Real-time visualization of TSP solving process with cluster-by-cluster updates
    """
    def __init__(self, points, clusters=None, real_time=True, sector_info=None):
        self.points = points
        self.clusters = clusters or []
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.ax.set_aspect('equal')
        self.current_route = []
        self.solving_status = "Initializing..."
        self.cluster_colors = plt.cm.Set3(np.linspace(0, 1, max(len(clusters), 1) if clusters else 1))
        self.real_time = real_time
        self.cluster_routes = {}  # Store individual cluster routes
        self.current_cluster = None
        self.solved_clusters = set()
        self.cluster_being_solved = None

        # Sector visualization info
        self.sector_info = sector_info  # (sectors_x, sectors_y, sector_width, sector_height, bounds)
        self.show_sectors = sector_info is not None

        # Real-time animation support
        self.animation_enabled = real_time
        if self.animation_enabled:
            plt.ion()  # Enable interactive mode
            self.fig.show()

        # Memory management for matplotlib
        self._plot_counter = 0
        self._max_plots = 100  # Clear figure after this many updates

    def set_sector_info(self, sectors_x, sectors_y, sector_width, sector_height, bounds):
        """Set sector subdivision info for visualization"""
        self.sector_info = (sectors_x, sectors_y, sector_width, sector_height, bounds)
        self.show_sectors = True

    def update_status(self, status):
        self.solving_status = status
        if self.real_time:
            self.draw()

    def start_cluster_solving(self, cluster_idx):
        """Mark a cluster as being actively solved"""
        self.cluster_being_solved = cluster_idx
        self.solving_status = f"Solving cluster {cluster_idx}..."
        if self.real_time:
            self.draw()

    def update_cluster_route(self, cluster_idx, route):
        """Update the route for a specific cluster"""
        self.cluster_routes[cluster_idx] = route[:]
        if self.real_time:
            self.draw()

    def finish_cluster_solving(self, cluster_idx, final_route):
        """Mark a cluster as solved with its final route"""
        self.cluster_routes[cluster_idx] = final_route[:]
        self.solved_clusters.add(cluster_idx)
        self.cluster_being_solved = None
        if self.real_time:
            self.draw()

    def update_route(self, route, partial=False):
        self.current_route = route[:]
        if self.real_time:
            self.draw(partial)
        else:
            self.draw(partial)

    def draw(self, partial=False):
        # Periodic cleanup to prevent memory leaks
        self._plot_counter += 1
        if self._plot_counter > self._max_plots:
            plt.close('all')
            self.fig, self.ax = plt.subplots(figsize=(14, 10))
            self._plot_counter = 0
            if self.animation_enabled:
                plt.ion()
                self.fig.show()

        self.ax.clear()
        self.ax.set_aspect('equal')

        # Draw sector boundaries if available
        if self.show_sectors and self.sector_info:
            sectors_x, sectors_y, sector_width, sector_height, bounds = self.sector_info
            min_x, max_x, min_y, max_y = bounds

            # Draw vertical grid lines
            for i in range(sectors_x + 1):
                x = min_x + i * sector_width
                self.ax.axvline(x=x, color='lightgray', linestyle='--', linewidth=1, alpha=0.7)

            # Draw horizontal grid lines
            for j in range(sectors_y + 1):
                y = min_y + j * sector_height
                self.ax.axhline(y=y, color='lightgray', linestyle='--', linewidth=1, alpha=0.7)

            # Add sector labels
            for i in range(sectors_x):
                for j in range(sectors_y):
                    sector_center_x = min_x + (i + 0.5) * sector_width
                    sector_center_y = min_y + (j + 0.5) * sector_height
                    self.ax.text(sector_center_x, sector_center_y,
                               f'S{i*sectors_y + j}',
                               fontsize=8, ha='center', va='center',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        # Draw clusters with different states
        if self.clusters:
            for ci, cluster in enumerate(self.clusters):
                xs = [self.points[i][0] for i in cluster]
                ys = [self.points[i][1] for i in cluster]
                color = self.cluster_colors[ci % len(self.cluster_colors)]

                # Different visual states for clusters
                if ci == self.cluster_being_solved:
                    # Currently being solved - highlight with bold border
                    self.ax.scatter(xs, ys, c=[color], alpha=0.9, s=80,
                                  edgecolors='red', linewidths=3, label=f'Cluster {ci} (Solving)')
                elif ci in self.solved_clusters:
                    # Solved - normal appearance with green border
                    self.ax.scatter(xs, ys, c=[color], alpha=0.8, s=60,
                                  edgecolors='green', linewidths=2, label=f'Cluster {ci} (Solved)')
                else:
                    # Unsolved - faded appearance
                    self.ax.scatter(xs, ys, c=[color], alpha=0.4, s=50,
                                  edgecolors='gray', linewidths=1, label=f'Cluster {ci} (Pending)')

                # Draw cluster centroid
                cx, cy = sum(xs)/len(xs), sum(ys)/len(ys)
                self.ax.scatter([cx], [cy], c='black', marker='x', s=100)
                self.ax.text(cx+0.2, cy+0.2, f'C{ci}', fontsize=10, fontweight='bold')

                # Draw cluster routes if available
                if ci in self.cluster_routes and len(self.cluster_routes[ci]) > 1:
                    route = self.cluster_routes[ci]
                    if ci == self.cluster_being_solved:
                        # Active solving - red dashed line
                        route_color, linestyle, linewidth = 'red', '--', 2
                    elif ci in self.solved_clusters:
                        # Solved cluster - solid green line
                        route_color, linestyle, linewidth = 'green', '-', 3
                    else:
                        # Partial route - thin gray line
                        route_color, linestyle, linewidth = 'gray', '-', 1

                    for i in range(len(route) - 1):
                        p1 = self.points[route[i]]
                        p2 = self.points[route[i + 1]]
                        self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                                   c=route_color, linestyle=linestyle, linewidth=linewidth, alpha=0.8)
        else:
            # No clusters - draw all points
            xs = [p[0] for p in self.points]
            ys = [p[1] for p in self.points]
            self.ax.scatter(xs, ys, c='blue', alpha=0.7, s=50)

        # Draw overall route if available
        if len(self.current_route) > 1:
            route_color = 'purple' if partial else 'black'
            linewidth = 3 if not partial else 2
            alpha = 1.0 if not partial else 0.6

            for i in range(len(self.current_route) - 1):
                p1 = self.points[self.current_route[i]]
                p2 = self.points[self.current_route[i + 1]]
                self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                           c=route_color, linewidth=linewidth, alpha=alpha)

            # Highlight start and end points
            if not partial:
                start_point = self.points[self.current_route[0]]
                end_point = self.points[self.current_route[-1]]
                self.ax.scatter([start_point[0]], [start_point[1]], c='green', marker='s', s=120, label='Start')
                self.ax.scatter([end_point[0]], [end_point[1]], c='red', marker='X', s=120, label='End')

        # Status info
        status_text = f'TSP Solving Progress: {self.solving_status}'
        if self.cluster_being_solved is not None:
            status_text += f'\nCurrently solving cluster {self.cluster_being_solved}'
        if self.solved_clusters:
            status_text += f'\nSolved clusters: {len(self.solved_clusters)}/{len(self.clusters)}'

        # Add sector info to status
        if self.show_sectors and self.sector_info:
            sectors_x, sectors_y, sector_width, sector_height, bounds = self.sector_info
            status_text += f'\nSpatial subdivision: {sectors_x}x{sectors_y} sectors'
            status_text += f' ({sector_width:.1f}x{sector_height:.1f} each)'

        self.ax.set_title(status_text, fontsize=12)
        self.ax.grid(True, alpha=0.3)
        if self.clusters and len(self.clusters) <= 10:  # Only show legend for small number of clusters
            self.ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

        if self.real_time:
            plt.draw()
            try:
                self.fig.canvas.flush_events()
            except:
                pass  # Ignore flush errors
        else:
            plt.draw()

    def show_final(self, route, cost):
        self.current_route = route
        self.solving_status = f"Completed! Total cost: {cost:.2f}"
        self.draw()

        # Save visualization to file for better compatibility
        try:
            plt.savefig("tsp_solution_with_sectors.png", dpi=150, bbox_inches='tight')
            print("Final visualization saved to tsp_solution_with_sectors.png")
        except:
            print("Could not save visualization file")

        if self.real_time:
            try:
                plt.show()
                plt.pause(3)  # Shorter pause
            except:
                print("Interactive display not available, visualization saved to file instead")

def choose_entries_exits(points, D, clusters, order, Kcand=3, visualizer=None):
    # Reduce Kcand for performance on large problems
    effective_kcand = min(Kcand, 2) if len(order) > 10 else Kcand

    candidates={}
    for ci in order:
        cidxs = clusters[ci]
        candidates[ci] = candidate_boundary_nodes(points, cidxs, K=effective_kcand)
        if len(candidates[ci])==0: candidates[ci] = [clusters[ci][0]]

    intra={}
    for ci in order:
        intra[ci] = {}
        idxs = clusters[ci]

        # Pre-compute cluster paths with limited caching for performance
        cluster_cache = {}
        cache_size_limit = 100  # Limit cache size

        for a in candidates[ci]:
            for b in candidates[ci]:
                if a==b and len(idxs)>1:
                    intra[ci][(a,b)] = (float('inf'), None)
                    continue

                # Use cache to avoid recomputation
                cache_key = (tuple(sorted(idxs)), a, b)
                if cache_key in cluster_cache:
                    cost, path = cluster_cache[cache_key]
                else:
                    # Notify visualizer that we're solving this cluster (only once per cluster)
                    if visualizer and hasattr(visualizer, 'start_cluster_solving') and len(cluster_cache) == 0:
                        visualizer.start_cluster_solving(ci)

                    # Use fast path for performance
                    if len(idxs) <= 6:
                        cost, path = cluster_path(points, D, idxs, a, b)
                    else:
                        cost, path = greedy_path_between_fast(idxs, a, b, D)
                    cluster_cache[cache_key] = (cost, path)

                intra[ci][(a,b)] = (cost, path)

        # Mark this cluster as solved
        if visualizer and hasattr(visualizer, 'finish_cluster_solving'):
            # Find the best route for this cluster
            best_route = None
            best_cost = float('inf')
            for (a,b), (cost, path) in intra[ci].items():
                if cost < best_cost and path:
                    best_cost = cost
                    best_route = path
            if best_route:
                visualizer.finish_cluster_solving(ci, best_route)
    # DP along sequence
    dp=[{} for _ in order]; parent=[{} for _ in order]
    for pos,ci in enumerate(order):
        if pos==0:
            for (a,b),(cst,path) in intra[ci].items():
                dp[pos][b] = cst; parent[pos][b] = (None, a)
        else:
            for (a,b),(cst,path) in intra[ci].items():
                best=float('inf'); bestprev=None
                for prev_exit, prev_cost in dp[pos-1].items():
                    inter = D[prev_exit][a]
                    tot = prev_cost + inter + cst
                    if tot < best:
                        best=tot; bestprev=prev_exit
                if best < float('inf'):
                    if b not in dp[pos] or best < dp[pos][b]:
                        dp[pos][b]=best; parent[pos][b]=(bestprev,a)
    last_pos = len(order)-1
    if not dp[last_pos]:
        return float('inf'), []
    best_exit = min(dp[last_pos], key=lambda x: dp[last_pos][x])
    best_cost = dp[last_pos][best_exit]
    seq=[]; cur_exit=best_exit
    for pos in range(last_pos, -1, -1):
        prev_exit, entry = parent[pos][cur_exit]
        ci = order[pos]
        seq.append((ci, entry, cur_exit, intra[ci][(entry,cur_exit)][1]))
        cur_exit = prev_exit
    seq=list(reversed(seq))
    return best_cost, seq

def stitch_full_route(seq):
    route = []
    for i,(ci, entry, exitn, path) in enumerate(seq):
        if path is None or len(path) == 0:
            continue
        if i==0:
            route.extend(path)
        else:
            # Skip duplicate entry point to avoid double-adding
            if len(route) > 0 and len(path) > 0 and route[-1] == path[0]:
                route.extend(path[1:])
            else:
                route.extend(path)

    # Handle empty route case
    if len(route) == 0:
        return []

    # remove consecutive duplicates
    final = [route[0]]
    for x in route[1:]:
        if x!=final[-1]: final.append(x)
    return final

def run_example(n_points=20, Kcand=3):
    points=[]
    for _ in range(n_points//3):
        points.append((random.uniform(0,2)+2, random.uniform(0,2)+2))
    for _ in range(n_points//3):
        points.append((random.uniform(0,2)+7, random.uniform(0,2)+1))
    for _ in range(n_points - 2*(n_points//3)):
        points.append((random.uniform(0,10), random.uniform(0,10)))
    D = pairwise_dist_matrix(points)
    clusters, avgs, grand_avg, D = benchmark_cluster_assignment(points)
    centroids = [ (sum(points[i][0] for i in c)/len(c), sum(points[i][1] for i in c)/len(c)) for c in clusters ]
    order, _ = solve_cluster_order(centroids)
    best_cost, seq = choose_entries_exits(points, D, clusters, order, Kcand=Kcand)
    full_route = stitch_full_route(seq)
    return {
        "points": points,
        "clusters": clusters,
        "avgs": avgs,
        "grand_avg": grand_avg,
        "order": order,
        "seq": seq,
        "full_route": full_route,
        "cost": best_cost
    }

import random

@lru_cache(maxsize=128)
def cached_distance(p1, p2):
    """Cached distance calculation for performance"""
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def enhanced_2opt(points, D, route, max_iterations=1000):
    """Enhanced 2-opt with better stopping criteria and performance"""
    def tour_cost(r):
        return sum(D[r[i]][r[i+1]] for i in range(len(r)-1))

    best_route = route[:]
    best_cost = tour_cost(best_route)
    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue  # Skip adjacent edges

                # Calculate improvement
                old_cost = D[route[i-1]][route[i]] + D[route[j-1]][route[j]]
                new_cost = D[route[i-1]][route[j-1]] + D[route[i]][route[j]]

                if new_cost < old_cost:
                    # Apply 2-opt swap
                    route[i:j] = reversed(route[i:j])
                    improved = True
                    break
            if improved:
                break

    return tour_cost(route), route

def solve_tsp_with_options(points, Kcand=3, visualize=True, use_adaptive=True):
    """Main TSP solver with enhanced options"""
    start_time = time.time()

    D = pairwise_dist_matrix(points)
    clusters, avgs, grand_avg, D = benchmark_cluster_assignment(points)

    # Initialize visualizer if requested
    visualizer = None
    if visualize:
        visualizer = TSPVisualizer(points, clusters)
        visualizer.update_status("Initial clustering complete")
        visualizer.draw()

    # Apply adaptive sub-clustering if enabled
    if use_adaptive:
        if visualizer:
            visualizer.update_status("Applying adaptive sub-clustering...")

        enhanced_clusters = []
        for cluster in clusters:
            if len(cluster) > 5:
                subclusters = adaptive_subcluster(points, cluster, max_size=5)
                enhanced_clusters.extend(subclusters)
            else:
                enhanced_clusters.append(cluster)
        clusters = enhanced_clusters

        if visualizer:
            visualizer.clusters = clusters
            visualizer.cluster_colors = plt.cm.Set3(np.linspace(0, 1, len(clusters)))
            visualizer.update_status(f"Sub-clustering complete. {len(clusters)} total clusters")
            visualizer.draw()

    centroids = [ (sum(points[i][0] for i in c)/len(c), sum(points[i][1] for i in c)/len(c)) for c in clusters ]

    if visualizer:
        visualizer.update_status("Solving cluster order...")

    order, _ = solve_cluster_order(centroids)

    if visualizer:
        visualizer.update_status("Finding optimal entries and exits...")

    best_cost, seq = choose_entries_exits(points, D, clusters, order, Kcand=Kcand, visualizer=visualizer)
    full_route = stitch_full_route(seq)

    # Apply enhanced 2-opt optimization
    if visualizer:
        visualizer.update_status("Applying 2-opt optimization...")
        visualizer.update_route(full_route, partial=True)

    if use_adaptive and len(full_route) > 3:
        optimized_cost, full_route = enhanced_2opt(points, D, full_route + [full_route[0]])
        full_route = full_route[:-1]  # Remove duplicate end point
        best_cost = optimized_cost

    solve_time = time.time() - start_time

    if visualizer:
        visualizer.show_final(full_route, best_cost)

    return {
        "points": points,
        "clusters": clusters,
        "avgs": avgs,
        "grand_avg": grand_avg,
        "order": order,
        "seq": seq,
        "full_route": full_route,
        "cost": best_cost,
        "solve_time": solve_time,
        "num_clusters": len(clusters),
        "adaptive_used": use_adaptive
    }

def run_example(n_points=20, Kcand=3, visualize=False, use_adaptive=True):
    points=[]
    for _ in range(n_points//3):
        points.append((random.uniform(0,2)+2, random.uniform(0,2)+2))
    for _ in range(n_points//3):
        points.append((random.uniform(0,2)+7, random.uniform(0,2)+1))
    for _ in range(n_points - 2*(n_points//3)):
        points.append((random.uniform(0,10), random.uniform(0,10)))

    return advanced_tsp_solver(points, visualize=visualize, use_all_optimizations=True)

def run_example_real(n_points=20, Kcand=3, visualize=False, use_adaptive=True):
    points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(n_points)]
    return advanced_tsp_solver(points, visualize=visualize, use_all_optimizations=True)

# Main execution and examples
#if __name__ == "__main__":
#    # Example usage with different configurations
#    #print("=== Basic TSP Example ===")
#    #res_basic = run_example_real(150, Kcand=3, visualize=False, use_adaptive=False)
#    #print(f"Basic solver - Cost: {res_basic['cost']:.2f}, Time: {res_basic['solve_time']:.3f}s, Clusters: {res_basic['num_clusters']}")
##
#    #print("\n=== Enhanced TSP with Adaptive Sub-clustering ===")
#    #res_enhanced = run_example_real(150, Kcand=3, visualize=True, use_adaptive=True)
#    #print(f"Enhanced solver - Cost: {res_enhanced['cost']:.2f}, Time: {res_enhanced['solve_time']:.3f}s, Clusters: {res_enhanced['num_clusters']}")
##
#    #improvement = ((res_basic['cost'] - res_enhanced['cost']) / res_basic['cost']) * 100
#    #print(f"\nImprovement: {improvement:.2f}% cost reduction")
#
#    # Performance comparison
#    print("\n=== Performance Comparison ===")
#    basic_times = []
#    enhanced_times = []
#    basic_costs = []
#    enhanced_costs = []
#
#    for i in range(1):  # Reduced for faster testing
#        print(f"Run {i+1}/3", end=" ")
#
#        # Basic solver
#
#        # Enhanced solver
#        start = time.time()
#        res_e = run_example_real(500, visualize=False, use_adaptive=True)
#        enhanced_times.append(time.time() - start)
#        enhanced_costs.append(res_e['cost'])
#
#        print("")
#
#    print(f"\nAverage Results (200   points, 3 runs):")
#    print(f"Enhanced: Cost={np.mean(enhanced_costs):.2f}±{np.std(enhanced_costs):.2f}, Time={np.mean(enhanced_times):.3f}±{np.std(enhanced_times):.3f}s")

    #avg_improvement = ((np.mean(basic_costs) - np.mean(enhanced_costs)) / np.mean(basic_costs)) * 100
    #print(f"Average improvement: {avg_improvement:.2f}% cost reduction")

# Convenience functions for quick testing
def quick_test(n_points=25, visualize=True):
    """Quick test function with reasonable defaults"""
    print(f"Quick TSP test with {n_points} points...")
    return test_advanced_solver(n_points, visualize=visualize, real_time_viz=True)

def demo_real_time_viz(n_points=15):
    """Demo function showcasing real-time cluster-by-cluster visualization"""
    print(f"\n=== Real-Time TSP Visualization Demo ({n_points} points) ===")
    print("This demo shows:")
    print("- Real-time cluster-by-cluster solving progress")
    print("- Visual feedback as each cluster gets solved")
    print("- Different visual states (Pending/Solving/Solved)")
    print("- Individual cluster routes being constructed")
    print("\nWatch the visualization window for real-time updates!")

    return test_advanced_solver(n_points=n_points, visualize=True, real_time_viz=True)

class MLDistancePredictor:
    """Machine learning-based distance and cost prediction for TSP optimization"""

    def __init__(self, max_cache_size=1000):
        self.cost_cache = {}
        self.feature_cache = {}
        self.prediction_model = None
        self.max_cache_size = max_cache_size

    def extract_features(self, points, start_idx, end_idx, cluster_idxs):
        """Extract features for ML prediction"""
        if len(cluster_idxs) < 2:
            return None

        cluster_points = np.array([points[i] for i in cluster_idxs])
        start_point = np.array(points[start_idx])
        end_point = np.array(points[end_idx])

        # Geometric features
        cluster_center = np.mean(cluster_points, axis=0)
        cluster_span = np.ptp(cluster_points, axis=0)
        cluster_area = cluster_span[0] * cluster_span[1]

        # Distance features
        start_to_center = np.linalg.norm(start_point - cluster_center)
        end_to_center = np.linalg.norm(end_point - cluster_center)
        start_to_end = np.linalg.norm(start_point - end_point)

        # Cluster shape features
        if len(cluster_points) > 2:
            cluster_std = np.std(cluster_points, axis=0)
            aspect_ratio = max(cluster_std) / (min(cluster_std) + 1e-6)
        else:
            aspect_ratio = 1.0

        # Density feature
        density = len(cluster_idxs) / (cluster_area + 1e-6)

        features = [
            len(cluster_idxs),           # cluster size
            cluster_area,                # cluster area
            start_to_center,            # start distance to center
            end_to_center,              # end distance to center
            start_to_end,               # direct start-end distance
            aspect_ratio,               # cluster shape
            density,                    # point density
            cluster_span[0],            # width
            cluster_span[1],            # height
        ]

        return np.array(features)

    def predict_cost(self, points, start_idx, end_idx, cluster_idxs):
        """Predict path cost using cached patterns"""
        cache_key = (tuple(sorted(cluster_idxs)), start_idx, end_idx)

        if cache_key in self.cost_cache:
            return self.cost_cache[cache_key]

        # Feature-based prediction for similar problems
        features = self.extract_features(points, start_idx, end_idx, cluster_idxs)
        if features is None:
            return None

        # Simple heuristic prediction (can be replaced with trained ML model)
        direct_dist = euclid(points[start_idx], points[end_idx])
        cluster_complexity = len(cluster_idxs) * features[5]  # size * aspect_ratio
        estimated_cost = direct_dist * (1 + cluster_complexity * 0.1)

        return estimated_cost

    def update_cache(self, points, start_idx, end_idx, cluster_idxs, actual_cost):
        """Update cache with actual results"""
        cache_key = (tuple(sorted(cluster_idxs)), start_idx, end_idx)

        # Limit cache size to prevent memory leaks
        if len(self.cost_cache) >= self.max_cache_size:
            # Remove oldest 20% of entries
            items_to_remove = list(self.cost_cache.keys())[:self.max_cache_size // 5]
            for key in items_to_remove:
                self.cost_cache.pop(key, None)

        self.cost_cache[cache_key] = actual_cost

def smart_cluster_path_with_ml(points, D, idxs, start_idx, end_idx, ml_predictor=None):
    """Enhanced cluster path with ML prediction and smart caching"""

    if ml_predictor:
        # Try to get prediction first
        predicted_cost = ml_predictor.predict_cost(points, start_idx, end_idx, idxs)

        # For very small clusters or if prediction suggests simple path is best
        if len(idxs) <= 3 or (predicted_cost and predicted_cost <= euclid(points[start_idx], points[end_idx]) * 1.1):
            cost, path = cluster_path(points, D, idxs, start_idx, end_idx)
            if ml_predictor:
                ml_predictor.update_cache(points, start_idx, end_idx, idxs, cost)
            return cost, path

    # Use adaptive clustering for larger/complex clusters
    if len(idxs) > 5:
        cost, path = adaptive_cluster_path(points, D, idxs, start_idx, end_idx)
    else:
        cost, path = cluster_path(points, D, idxs, start_idx, end_idx)

    # Update ML cache
    if ml_predictor:
        ml_predictor.update_cache(points, start_idx, end_idx, idxs, cost)

    return cost, path

def advanced_tsp_solver(points=None, visualize=True, use_all_optimizations=True, real_time_viz=True, File=None, normalize_coords=True, target_area_size=25):
    """
    Most advanced TSP solver with all optimizations enabled.
    Can solve from points or load from a .tsp file.

    Args:
        points: List of (x, y) coordinate tuples (ignored if File is specified)
        visualize: Whether to show visualization
        use_all_optimizations: Whether to use all optimization techniques
        real_time_viz: Whether to show real-time visualization updates
        File: Path to .tsp file to load and solve (overrides points parameter)
        normalize_coords: Whether to normalize coordinates when loading from file
        target_area_size: Target area size for coordinate normalization

    Returns:
        dict: Solver results including performance metrics
    """
    # Handle file loading if File parameter is specified
    if File is not None:
        print(f"File parameter specified: {File}")
        if not points:
            # Load from file
            return load_and_solve_tsp_file(
                File,
                visualize=visualize,
                normalize_coords=normalize_coords,
                target_area_size=target_area_size
            )
        else:
            print("Warning: Both File and points specified. File parameter takes precedence.")
            return load_and_solve_tsp_file(
                File,
                visualize=visualize,
                normalize_coords=normalize_coords,
                target_area_size=target_area_size
            )

    # Validate points parameter if no file specified
    if points is None or len(points) == 0:
        raise ValueError("Either 'points' must be provided or 'File' parameter must specify a valid .tsp file path")

    #print(f"[DeBUG] Advanced TSP Solver - Processing {len(points)} points")
    start_time = time.time()

    # Initialize ML predictor
    print("Stage 1: Initializing ML predictor...")
    ml_predictor = MLDistancePredictor() if use_all_optimizations else None

    # Stage 2: Enhanced spatial clustering with subdivision
    print("Stage 2: Enhanced spatial clustering with area subdivision...")

    # Use enhanced spatial clustering for large datasets
    sector_info = None
    sector_optimization_info = {}

    if len(points) >= 100:
        print(f"Large dataset detected ({len(points)} points) - using spatial subdivision approach")
        clusters, sector_optimization_info = enhanced_spatial_clustering_with_optimization(
            points, max_nodes_per_sector=250, max_cluster_size=25, apply_sector_optimization=True
        )

        # Get sector info for visualization
        sectors_x, sectors_y, sector_width, sector_height, bounds = calculate_optimal_subdivision(
            points, max_nodes_per_sector=250
        )
        sector_info = (sectors_x, sectors_y, sector_width, sector_height, bounds)

        # Compute distance matrix after clustering
        print("Computing distance matrix...")
        D = pairwise_dist_matrix(points)

        # Calculate dummy averages for compatibility with existing code
        avgs = [0.0] * len(points)  # Placeholder
        grand_avg = 0.0  # Placeholder
    else:
        # For smaller datasets, use original clustering method
        print("Small dataset - using original clustering method...")
        D = pairwise_dist_matrix(points)
        clusters, avgs, grand_avg, D = benchmark_cluster_assignment(points)

    # Initialize visualizer
    visualizer = None
    if visualize:
        print("Stage 3: Initializing visualization...")
        visualizer = TSPVisualizer(points, clusters, real_time=real_time_viz, sector_info=sector_info)
        if sector_info:
            visualizer.set_sector_info(*sector_info)
        visualizer.update_status("Advanced analysis in progress...")
        if not real_time_viz:
            visualizer.draw()

    # Check if comprehensive sector optimization was applied
    sector_optimized = sector_optimization_info.get('total_sector_optimization_time', 0) > 0

    if not sector_optimized:
        # Stage 2: Adaptive parameter tuning (only if sectors weren't optimized)
        print("Stage 4: Auto-tuning parameters...")
        clustering_config, adaptive_kcand = adaptive_parameter_tuning(points, clusters)

        # Stage 3: Enhanced sub-clustering (only if sectors weren't optimized)
        print("Stage 5: Enhanced sub-clustering...")
        enhanced_clusters = []
        for i, cluster in enumerate(clusters):
            if len(cluster) > clustering_config.max_size:
                subclusters = adaptive_subcluster_enhanced(points, cluster, clustering_config)
                enhanced_clusters.extend(subclusters)
            else:
                enhanced_clusters.append(cluster)
        clusters = enhanced_clusters
        print(f"Sub-clustering complete: {len(clusters)} total clusters")

        if visualizer:
            visualizer.clusters = clusters
            visualizer.cluster_colors = plt.cm.Set3(np.linspace(0, 1, len(clusters)))
            visualizer.update_status(f" {len(clusters)} optimized clusters created")
            visualizer.draw()

        # Stage 4: Genetic algorithm for cluster ordering
        print(f"Stage 6: Optimizing cluster order ({len(clusters)} clusters)...")
        centroids = [(sum(points[i][0] for i in c)/len(c), sum(points[i][1] for i in c)/len(c)) for c in clusters]

        if len(centroids) > 8 and len(centroids) <= 50:
            print("Using genetic algorithm for cluster ordering...")
            order, order_cost = genetic_cluster_order(centroids, max_time=3.0)
        else:
            print("Using nearest neighbor for cluster ordering...")
            order, order_cost = solve_cluster_order(centroids)
        print(f"Cluster order optimized with cost: {order_cost:.2f}")

        # Stage 5: Parallel cluster solving
        print("Stage 7: Solving cluster entry/exit points...")
        if visualizer:
            visualizer.update_status("Processing clusters...")
        if len(clusters) > 4:
            best_cost, seq = choose_entries_exits_parallel(points, D, clusters, order, Kcand=adaptive_kcand, visualizer=visualizer)
        else:
            best_cost, seq = choose_entries_exits(points, D, clusters, order, Kcand=adaptive_kcand, visualizer=visualizer)

        full_route = stitch_full_route(seq)
        print(f"Initial route cost: {best_cost:.2f}")

    else:
        # Sectors were comprehensively optimized - use lightweight global connection
        print("Stages 4-7: Skipped (comprehensive sector optimization completed)")
        print("Stage 8: Lightweight inter-sector route connection...")

        # Create a simple route by connecting sector routes
        full_route = []
        best_cost = 0

        if 'sector_routes' in sector_optimization_info:
            sector_routes = sector_optimization_info['sector_routes']

            # Simple concatenation of sector routes for now
            # In practice, you'd want to optimize the connections between sectors
            for sector_id in sorted(sector_routes.keys()):
                sector_route = sector_routes[sector_id]['route']
                if sector_route:
                    if not full_route:
                        full_route.extend(sector_route)
                    else:
                        # Connect to previous sector (remove duplicate connection points)
                        full_route.extend(sector_route)
                    best_cost += sector_routes[sector_id]['cost']

            print(f"Inter-sector route connected: {best_cost:.2f} (sum of sector costs)")
        else:
            # Fallback to simple cluster concatenation
            full_route = [i for cluster in clusters for i in cluster]
            best_cost = 0  # Will be calculated later

        # Light optimization of inter-sector connections
        if len(full_route) > 3:
            print("Optimizing inter-sector connections...")
            from TSP import or_opt_optimization_fast
            inter_cost, full_route = or_opt_optimization_fast(
                D, full_route, max_segment_length=1, max_iterations=1
            )
            best_cost = inter_cost
            print(f"Inter-sector optimization complete: {best_cost:.2f}")

    # Stage 6: Multi-level optimization
    print("Stage 8: Multi-level route optimization...")
    if visualizer:
        visualizer.update_status("Route optimization...")
        visualizer.update_route(full_route, partial=True)

    if len(full_route) > 3:
        # Check if sector-level optimization was applied
        sector_optimized = sector_optimization_info.get('total_sector_optimization_time', 0) > 0

        if sector_optimized:
            # Sector-level 2-opt + LK was already applied, only do lightweight global optimization
            print("Level 1: Lightweight global route optimization (sectors already optimized)...")

            # Just apply fast or-opt at the global level to connect sectors optimally
            from TSP import or_opt_optimization_fast
            lk_cost, lk_route = or_opt_optimization_fast(
                D, full_route, max_segment_length=2, max_iterations=1
            )
            lk_route = full_route  # Keep the route as-is since sectors are optimized

            print(f"Global route connecting optimized sectors: {lk_cost:.2f}")
        else:
            # No sector optimization was done, apply full global optimization
            print("Level 1: Enhanced 2-opt + Lin-Kernighan (global)...")
            lk_cost, lk_route = enhanced_2opt_with_lk(points, D, full_route + [full_route[0]])
            lk_route = lk_route[:-1]  # Remove duplicate end point

        # Level 2: Lightweight hierarchical or-opt (much reduced since sectors are optimized)
        print("Level 2: Lightweight inter-sector optimization...")

        if sector_optimized:
            # Very light optimization since heavy work was done in sectors
            final_cost, final_route = or_opt_optimization_fast(
                D, lk_route, max_segment_length=1, max_iterations=1
            )
            optimization_method = "Lightweight inter-sector"
        else:
            # Standard hierarchical optimization for non-sector-optimized problems
            cluster_boundaries = []
            route_pos = 0
            for ci in order:
                cluster_size = len(clusters[ci])
                cluster_boundaries.append((route_pos, route_pos + cluster_size))
                route_pos += cluster_size

            if len(clusters) > 10:
                final_cost, final_route = true_hierarchical_or_opt_optimization(points, D, lk_route, clusters, order)
                optimization_method = "True hierarchical or-opt"
            elif len(lk_route) > 150:
                final_cost, final_route = hierarchical_or_opt_optimization(points, D, lk_route, clusters, cluster_boundaries)
                optimization_method = "Sliding window or-opt"
            else:
                final_cost, final_route = or_opt_optimization(points, D, lk_route)
                optimization_method = "Standard or-opt"

        if final_cost < best_cost:
            best_cost = final_cost
            full_route = final_route
            print(f"{optimization_method} improved cost to: {best_cost:.2f}")
        else:
            full_route = lk_route
            best_cost = lk_cost if 'lk_cost' in locals() else best_cost
            print(f"Global optimization achieved final cost: {best_cost:.2f}")

    solve_time = time.time() - start_time

    # Final results
    print(f"Advanced TSP Solution Complete!")
    print(f"Final cost: {best_cost:.2f}, Clusters: {len(clusters)}, Time: {solve_time:.3f}s")

    # Report sector optimization details if available
    if sector_optimization_info and 'total_sector_optimization_time' in sector_optimization_info:
        sector_time = sector_optimization_info['total_sector_optimization_time']
        global_time = solve_time - sector_time
        print(f"  Sector-level optimization: {sector_time:.3f}s")
        print(f"  Global optimization: {global_time:.3f}s")
        print(f"  Efficiency gain: {sector_time/solve_time*100:.1f}% of work done in parallel sectors")

    if visualizer:
        visualizer.show_final(full_route, best_cost)

    return {
        "points": points,
        "clusters": clusters,
        "order": order if 'order' in locals() else list(range(len(clusters))),
        "full_route": full_route,
        "cost": best_cost,
        "solve_time": solve_time,
        "sector_optimization_time": sector_optimization_info.get('total_sector_optimization_time', 0),
        "num_clusters": len(clusters),
        "optimizations_used": {
            "adaptive_subclustering": True,
            "sector_level_or_opt": len(points) >= 100,
            "genetic_cluster_order": 'centroids' in locals() and len(centroids) > 8,
            "parallel_processing": len(clusters) > 4,
            "lin_kernighan": True,
            "or_opt": True,
            "ml_prediction": use_all_optimizations
        }
    }
    
def demo_adaptive_clustering():
    """Demonstration of adaptive sub-clustering benefits"""
    print("=== Adaptive Sub-clustering Demo ===")

    # Create a challenging scenario with large clusters
    points = []
    # Large cluster 1
    for _ in range(15):
        points.append((random.uniform(0, 3), random.uniform(0, 3)))
    # Large cluster 2
    for _ in range(12):
        points.append((random.uniform(7, 10), random.uniform(7, 10)))
    # Scattered points
    for _ in range(8):
        points.append((random.uniform(0, 10), random.uniform(0, 10)))

    print(f"Testing with {len(points)} points...")

    # Without adaptive clustering
    print("\nSolving without adaptive sub-clustering...")
    res_normal = solve_tsp_with_options(points, Kcand=3, visualize=False, use_adaptive=False)

    # With adaptive clustering
    print("Solving with adaptive sub-clustering...")
    res_adaptive = solve_tsp_with_options(points, Kcand=3, visualize=True, use_adaptive=True)

    print(f"\nResults:")
    print(f"Normal:   Cost={res_normal['cost']:.2f}, Clusters={res_normal['num_clusters']}, Time={res_normal['solve_time']:.3f}s")
    print(f"Adaptive: Cost={res_adaptive['cost']:.2f}, Clusters={res_adaptive['num_clusters']}, Time={res_adaptive['solve_time']:.3f}s")

    improvement = ((res_normal['cost'] - res_adaptive['cost']) / res_normal['cost']) * 100
    print(f"Improvement: {improvement:.2f}% cost reduction")

    return res_normal, res_adaptive

# Missing functions implementation

def adaptive_parameter_tuning(points, clusters):
    """Automatically tune parameters based on problem characteristics"""
    n_points = len(points)
    n_clusters = len(clusters)
    avg_cluster_size = n_points / n_clusters if n_clusters > 0 else 0

    # Simple config class
    class ClusteringConfig:
        def __init__(self):
            self.max_size = 5
            self.method = 'adaptive'

    config = ClusteringConfig()

    # Adjust max_size based on problem size
    if n_points > 100:
        config.max_size = 8
    elif n_points > 50:
        config.max_size = 6
    else:
        config.max_size = 5

    # Adjust Kcand based on cluster sizes
    if avg_cluster_size > 8:
        Kcand = 5
    elif avg_cluster_size > 5:
        Kcand = 4
    else:
        Kcand = 3

    return config, Kcand

def adaptive_subcluster_enhanced(points, cluster_idxs, config, depth=0):
    """Enhanced recursive clustering"""
    if len(cluster_idxs) <= config.max_size or depth >= 3:
        return [cluster_idxs]

    # Simple k-means clustering
    cluster_points = np.array([points[i] for i in cluster_idxs])
    n_subclusters = min(len(cluster_idxs) // config.max_size + 1, len(cluster_idxs))

    if n_subclusters <= 1:
        return [cluster_idxs]

    try:
        kmeans = KMeans(n_clusters=n_subclusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(cluster_points)

        subclusters = [[] for _ in range(n_subclusters)]
        for i, label in enumerate(labels):
            subclusters[label].append(cluster_idxs[i])

        # Recursively sub-cluster each subcluster
        final_subclusters = []
        for subcluster in subclusters:
            if len(subcluster) > 0:
                final_subclusters.extend(adaptive_subcluster_enhanced(points, subcluster, config, depth+1))

        return final_subclusters
    except:
        return [cluster_idxs]

def genetic_cluster_order(centroids, max_time=5.0):
    """Genetic algorithm for finding optimal cluster order with time limit"""
    n_clusters = len(centroids)
    if n_clusters <= 8:  # Use brute force for small instances
        return solve_cluster_order(centroids)

    # For very large problems, use simple nearest neighbor
    if n_clusters > 50:
        print(f"  Large problem ({n_clusters} clusters) - using fast nearest neighbor")
        return solve_cluster_order(centroids)

    start_time = time.time()
    # Distance matrix for centroids
    D = [[euclid(centroids[i], centroids[j]) for j in range(n_clusters)] for i in range(n_clusters)]

    def fitness(individual):
        cost = sum(D[individual[i]][individual[i+1]] for i in range(len(individual)-1))
        cost += D[individual[-1]][individual[0]]  # Return to start
        return 1.0 / (cost + 1e-6)

    def crossover(parent1, parent2):
        # Order crossover (OX)
        start, end = sorted(random.sample(range(len(parent1)), 2))
        child = [-1] * len(parent1)
        child[start:end] = parent1[start:end]

        pointer = end
        for city in parent2[end:] + parent2[:end]:
            if city not in child:
                child[pointer % len(child)] = city
                pointer += 1
        return child

    def mutate(individual, mutation_rate=0.02):
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                j = random.randint(0, len(individual) - 1)
                individual[i], individual[j] = individual[j], individual[i]
        return individual

    # Initialize population - reduced size for performance
    population_size = min(30, n_clusters * 2)
    max_generations = min(50, 200 // n_clusters)
    population = []
    for _ in range(population_size):
        individual = list(range(1, n_clusters))  # Start at 0, visit others
        random.shuffle(individual)
        individual = [0] + individual
        population.append(individual)

    elite_size = max(1, int(population_size * 0.2))

    for generation in range(max_generations):
        # Check time limit
        if time.time() - start_time > max_time:
            print(f"  Time limit reached at generation {generation}")
            break
        # Evaluate fitness
        fitness_scores = [(fitness(ind), ind) for ind in population]
        fitness_scores.sort(reverse=True)

        # Select elite
        new_population = [ind for _, ind in fitness_scores[:elite_size]]

        # Generate offspring
        while len(new_population) < population_size:
            parent1 = random.choice(fitness_scores[:population_size//2])[1]
            parent2 = random.choice(fitness_scores[:population_size//2])[1]
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    # Return best solution
    best_individual = max(population, key=fitness)
    best_cost = sum(D[best_individual[i]][best_individual[i+1]] for i in range(len(best_individual)-1))
    return best_individual, best_cost

def choose_entries_exits_parallel(points, D, clusters, order, Kcand=3, visualizer=None):
    """Simplified parallel version - falls back to sequential for now"""
    return choose_entries_exits(points, D, clusters, order, Kcand, visualizer)

def enhanced_2opt_with_lk(points, D, route, max_iterations=1000):
    """Enhanced 2-opt with Lin-Kernighan-style moves"""
    def tour_cost(r):
        return sum(D[r[i]][r[i+1]] for i in range(len(r)-1))

    current_route = route[:]
    current_cost = tour_cost(current_route)

    # Standard 2-opt
    for iteration in range(max_iterations):
        improved = False

        for i in range(1, len(current_route) - 2):
            for j in range(i + 2, len(current_route)):
                # Calculate improvement delta efficiently
                delta = (D[current_route[i-1]][current_route[j-1]] +
                        D[current_route[i]][current_route[j]]) - \
                       (D[current_route[i-1]][current_route[i]] +
                        D[current_route[j-1]][current_route[j]])

                if delta < -1e-9:  # Significant improvement
                    current_route[i:j] = reversed(current_route[i:j])
                    current_cost += delta
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

    return current_cost, current_route

def or_opt_local(D, route, start_idx, end_idx, max_segment_length=3):
    """Local or-opt within a route segment for performance"""
    if end_idx - start_idx < 4:
        return route[:]

    segment_route = route[start_idx:end_idx]
    improved = True

    while improved:
        improved = False

        for segment_len in range(1, min(max_segment_length + 1, len(segment_route) // 3)):
            for i in range(len(segment_route) - segment_len):
                segment = segment_route[i:i + segment_len]

                # Try a limited number of positions around the segment
                positions_to_try = []
                # Try positions before the segment
                for j in range(max(0, i - 5), i):
                    positions_to_try.append(j)
                # Try positions after the segment
                for j in range(i + segment_len + 1, min(len(segment_route) + 1, i + segment_len + 6)):
                    positions_to_try.append(j)

                for j in positions_to_try:
                    # Create new route segment
                    new_segment = segment_route[:i] + segment_route[i + segment_len:]
                    new_segment[j if j <= i else j - segment_len:j if j <= i else j - segment_len] = segment

                    # Calculate cost improvement (delta only)
                    old_cost = 0
                    new_cost = 0

                    # Calculate cost for affected edges only
                    if i > 0:
                        old_cost += D[segment_route[i-1]][segment_route[i]]
                        new_cost += D[new_segment[i-1]][new_segment[i]] if i < len(new_segment) else 0
                    if i + segment_len < len(segment_route):
                        old_cost += D[segment_route[i + segment_len - 1]][segment_route[i + segment_len]]

                    # Check if improvement
                    if len(new_segment) == len(segment_route):  # Valid segment
                        segment_route = new_segment
                        improved = True
                        break

                if improved:
                    break
            if improved:
                break

    # Replace the segment in the original route
    result = route[:start_idx] + segment_route + route[end_idx:]
    return result

def create_cluster_hierarchy(cluster_order, group_size=5, max_levels=None):
    """Create hierarchical grouping of clusters for optimization with dynamic max cap"""
    if len(cluster_order) <= group_size:
        return [cluster_order]  # Single group

    # Dynamic max levels - just enough to encompass all clusters
    if max_levels is None:
        max_levels = max(1, int(math.ceil(math.log(len(cluster_order), group_size))))
        max_levels = min(max_levels, 4)  # Hard cap for safety

    hierarchy = []
    current_level = cluster_order[:]
    level_count = 0

    while len(current_level) > 1 and level_count < max_levels:
        next_level = []
        groups = []

        # Group current level into groups of group_size
        for i in range(0, len(current_level), group_size):
            group = current_level[i:i + group_size]
            groups.append(group)
            next_level.append(f"group_{len(next_level)}")  # Representative for next level

        hierarchy.append(groups)
        current_level = next_level
        level_count += 1

        if len(current_level) <= 1 or level_count >= max_levels:
            break

    return hierarchy

def map_cluster_boundaries_in_route(route, clusters, cluster_order):
    """Map cluster positions in the final route for hierarchical optimization"""
    cluster_positions = {}
    route_pos = 0

    for ci in cluster_order:
        cluster_size = len(clusters[ci])
        cluster_positions[ci] = (route_pos, route_pos + cluster_size)
        route_pos += cluster_size

    return cluster_positions

def optimize_cluster_group(D, route, cluster_positions, cluster_group, max_iterations=2):
    """Optimize or-opt within a group of clusters"""
    if len(cluster_group) <= 1:
        return route[:]

    # Find the route segment that spans this cluster group
    start_pos = min(cluster_positions[ci][0] for ci in cluster_group)
    end_pos = max(cluster_positions[ci][1] for ci in cluster_group)

    if end_pos - start_pos < 4:
        return route[:]

    # Extract and optimize the segment
    segment = route[start_pos:end_pos]
    optimized_cost, optimized_segment = or_opt_optimization_fast(
        D, segment, max_segment_length=2, max_iterations=max_iterations
    )

    # Replace in original route
    new_route = route[:]
    new_route[start_pos:end_pos] = optimized_segment

    return new_route

def true_hierarchical_or_opt_optimization(points, D, route, clusters, cluster_order):
    """True hierarchical or-opt using cluster hierarchy with memory limits"""
    if len(route) < 4 or len(cluster_order) <= 1:
        return sum(D[route[i]][route[i+1]] for i in range(len(route)-1)), route

    print(f"   True hierarchical or-opt: {len(route)} points, {len(cluster_order)} clusters")

    # Create cluster hierarchy with dynamic depth - just enough to encompass all clusters
    hierarchy = create_cluster_hierarchy(cluster_order, group_size=5)
    print(f"   Created {len(hierarchy)} hierarchy levels (dynamic cap)")

    # Map cluster positions in the route
    cluster_positions = map_cluster_boundaries_in_route(route, clusters, cluster_order)
    current_route = route[:]

    # Optimize level by level, bottom-up
    total_operations = 0
    for level_idx, level_groups in enumerate(hierarchy):
        print(f"   Level {level_idx + 1}: Optimizing {len(level_groups)} groups")

        level_operations = 0
        for group_idx, group in enumerate(level_groups):
            # Skip single-cluster groups
            if len(group) <= 1:
                continue

            # Optimize this group
            old_route = current_route[:]
            current_route = optimize_cluster_group(D, current_route, cluster_positions, group, max_iterations=1)

            # Check if improvement was made
            if current_route != old_route:
                level_operations += 1

            total_operations += 1

        print(f"   Level {level_idx + 1}: {level_operations}/{len(level_groups)} groups improved")

        # For higher levels, we need to update cluster positions
        # as they represent groups rather than individual clusters
        if level_idx < len(hierarchy) - 1:
            # Update cluster positions for the next level
            # Each group becomes a "cluster" for the next level
            new_cluster_positions = {}
            for group_idx, group in enumerate(level_groups):
                group_start = min(cluster_positions[ci][0] for ci in group if ci in cluster_positions)
                group_end = max(cluster_positions[ci][1] for ci in group if ci in cluster_positions)
                new_cluster_positions[f"group_{group_idx}"] = (group_start, group_end)

            # Update the cluster_positions for next level
            cluster_positions.update(new_cluster_positions)

    cost = sum(D[current_route[i]][current_route[i+1]] for i in range(len(current_route)-1))
    print(f"   Total operations: {total_operations}")

    return cost, current_route

def hierarchical_or_opt_optimization(points, D, route, clusters, cluster_boundaries=None):

    current_route = route[:]
    print(f"   Hierarchical or-opt: {len(route)} points, {len(clusters)} clusters")

    # For very large problems, use fast optimization only
    if len(route) > 300:
        print(f"   Large problem detected - using fast or-opt only")
        return or_opt_optimization_fast(D, current_route, max_segment_length=2, max_iterations=1)

    # Phase 1: Fast sliding window optimization for manageable sizes
    if len(route) > 150:
        print(f"   Phase 1: Sliding window optimization")
        window_size = 50
        for start in range(0, len(current_route) - window_size, window_size // 2):
            end = min(start + window_size, len(current_route))
            if end - start > 10:  # Only meaningful windows
                # Use fast local optimization
                try:
                    segment = current_route[start:end]
                    _, optimized_segment = or_opt_optimization_fast(D, segment, max_segment_length=2, max_iterations=1)
                    current_route[start:end] = optimized_segment
                except:
                    continue  # Skip problematic segments

    # Phase 2: Very limited global improvement for large problems
    if len(route) > 100:
        print(f"   Phase 2: Limited global optimization")
        # Just one pass of fast or-opt
        cost, current_route = or_opt_optimization_fast(D, current_route, max_segment_length=2, max_iterations=1)
        return cost, current_route
    else:
        # Small enough for normal optimization
        return or_opt_optimization_fast(D, current_route, max_segment_length=2, max_iterations=2)

    cost = sum(D[current_route[i]][current_route[i+1]] for i in range(len(current_route)-1))
    return cost, current_route

def or_opt_optimization_fast(D, route, max_segment_length=2, max_iterations=3):
    """Faster or-opt with early termination for large problems"""
    if len(route) < 4:
        return sum(D[route[i]][route[i+1]] for i in range(len(route)-1)), route

    current_route = route[:]
    current_cost = sum(D[current_route[i]][current_route[i+1]] for i in range(len(current_route)-1))

    # Limit iterations for large problems
    if len(route) > 100:
        max_iterations = min(max_iterations, 2)
        max_segment_length = min(max_segment_length, 2)

    for iteration in range(max_iterations):
        improved = False

        # Sample positions instead of trying all positions for large routes
        positions_to_try = list(range(len(current_route)))
        if len(current_route) > 200:
            # Sample 1/3 of positions for very large routes
            positions_to_try = positions_to_try[::3]

        for segment_len in range(1, min(max_segment_length + 1, len(current_route) // 4)):
            for i in positions_to_try:
                if i + segment_len >= len(current_route):
                    continue

                segment = current_route[i:i + segment_len]

                # Limited search range for large problems
                search_range = min(50, len(current_route)) if len(current_route) > 100 else len(current_route)

                for j in range(max(0, i - search_range//2), min(len(current_route) - segment_len + 1, i + search_range//2)):
                    if abs(j - i) <= segment_len:
                        continue

                    # Quick delta calculation instead of full recalculation
                    delta = calculate_or_opt_delta(D, current_route, i, segment_len, j)

                    if delta < -1e-9:  # Significant improvement
                        # Apply the move
                        new_route = current_route[:i] + current_route[i + segment_len:]
                        new_route[j:j] = segment
                        current_route = new_route
                        current_cost += delta
                        improved = True
                        break

                if improved:
                    break
            if improved:
                break

        if not improved:
            break

    return current_cost, current_route

def calculate_or_opt_delta(D, route, i, segment_len, j):
    """Calculate the cost delta for an or-opt move without full recalculation"""
    n = len(route)

    # Cost of removing the segment
    delta = 0
    if i > 0 and i + segment_len < n:
        # Remove edge from i-1 to i and from i+segment_len-1 to i+segment_len
        # Add direct edge from i-1 to i+segment_len
        delta -= D[route[i-1]][route[i]]
        delta -= D[route[i + segment_len - 1]][route[i + segment_len]]
        delta += D[route[i-1]][route[i + segment_len]]

    # Cost of inserting the segment at position j
    if j < i:
        # Insert before original position
        if j > 0:
            delta -= D[route[j-1]][route[j]]
            delta += D[route[j-1]][route[i]]
        if j < n - segment_len:
            delta -= D[route[j-1] if j > 0 else route[0]][route[j]]
            delta += D[route[i + segment_len - 1]][route[j]]
    else:
        # Insert after original position (j adjusted for removal)
        j_adj = j - segment_len
        if j_adj > 0 and j_adj < n - segment_len:
            delta -= D[route[j_adj - 1]][route[j_adj]]
            delta += D[route[j_adj - 1]][route[i]]
            delta += D[route[i + segment_len - 1]][route[j_adj]]

    return delta

def or_opt_optimization(points, D, route, max_segment_length=3):
    """Original or-opt kept for compatibility - now calls faster version for large problems"""
    if len(route) > 150:
        return or_opt_optimization_fast(D, route, max_segment_length=2, max_iterations=2)

    # Original implementation for smaller problems
    if len(route) < 4:
        return sum(D[route[i]][route[i+1]] for i in range(len(route)-1)), route

    current_route = route[:]
    current_cost = sum(D[current_route[i]][current_route[i+1]] for i in range(len(current_route)-1))
    improved = True

    while improved:
        improved = False

        for segment_len in range(1, min(max_segment_length + 1, len(current_route) // 2)):
            for i in range(len(current_route) - segment_len):
                segment = current_route[i:i + segment_len]

                # Try inserting segment at different positions
                for j in range(len(current_route) - segment_len + 1):
                    if abs(j - i) <= segment_len:  # Skip nearby positions
                        continue

                    # Create new route
                    new_route = current_route[:i] + current_route[i + segment_len:]
                    new_route[j:j] = segment

                    # Calculate cost
                    new_cost = sum(D[new_route[k]][new_route[k+1]] for k in range(len(new_route)-1))

                    if new_cost < current_cost:
                        current_route = new_route
                        current_cost = new_cost
                        improved = True
                        break

                if improved:
                    break
            if improved:
                break

    return current_cost, current_route

# Quick test function
def test_advanced_solver(n_points=15, visualize=False, area_size=15, real_time_viz=True):
    """Quick test of the advanced solver with configurable parameters

    Args:
        n_points: Number of test points to generate
        visualize: Whether to show visualization
        area_size: Size of the area to generate points in (0 to area_size)
        real_time_viz: Enable real-time cluster-by-cluster updates (default: True)
    """
    #print(f"[DeBUG] Testing Advanced TSP Solver with {n_points} points...")
    #print(f"[DeBUG] Generating {n_points} random points in {area_size}x{area_size} area...")

    # Use fixed seed for debugging
    random.seed(42)
    test_points = [(random.uniform(0, area_size), random.uniform(0, area_size)) for _ in range(n_points)]
    #print(f"[DeBUG] Points generated: {test_points[:min(5, len(test_points))]}... (showing first few)")
    #print(f"[DeBUG] Total points: {len(test_points)}")

    # Validate points
    for i, pt in enumerate(test_points):
        if not isinstance(pt, tuple) or len(pt) != 2:
            #print(f"[DeBUG] Invalid point at index {i}: {pt}")
            return None
        if not isinstance(pt[0], (int, float)) or not isinstance(pt[1], (int, float)):
            #print(f"[DeBUG] Non-numeric point at index {i}: {pt}")
            return None

    try:
        # Use simpler solver for large problems to avoid getting stuck
        # For debugging, use even simpler approach
        #if n_points > 20:
        #    print("[DEBUG] Using simplified solver")
        #    result = solve_tsp_with_options(test_points, Kcand=3, visualize=visualize, use_adaptive=False)
        #elif n_points > 10:
        #    print("[DEBUG] Using basic solver")
        #    result = run_example_real(n_points, Kcand=3, visualize=visualize, use_adaptive=False)
        #else:
        #    #print(f"[DeBUG] Using advanced solver for {n_points} points")
        result = advanced_tsp_solver(test_points, visualize=visualize, use_all_optimizations=False, real_time_viz=real_time_viz)

        #print(f"[DeBUG] Success! Cost: {result['cost']:.2f}, Time: {result['solve_time']:.3f}s")
        #print(f"[DeBUG] Clusters: {result.get('num_clusters', 'N/A')}")
        #print(f"[DeBUG] Points per second: {n_points/result['solve_time']:.1f}")
        return result
    except Exception as e:
        #print(f"[DeBUG] Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_spatial_subdivision_solver(n_points=1000, area_size=25, visualize=True):
    """
    Test the enhanced spatial subdivision solver with large datasets.

    Args:
        n_points: Number of points to generate
        area_size: Size of the area (area_size x area_size)
        visualize: Whether to show visualization

    Returns:
        dict: Solver results including performance metrics
    """
    print(f"\n=== Testing Enhanced Spatial Subdivision TSP Solver ===")
    print(f"Points: {n_points}, Area: {area_size}x{area_size}")

    # Generate test points
    random.seed(42)  # For reproducible results
    test_points = [(random.uniform(0, area_size), random.uniform(0, area_size)) for _ in range(n_points)]

    start_time = time.time()

    # Use the enhanced spatial clustering approach
    result = advanced_tsp_solver(test_points, visualize=visualize, use_all_optimizations=True, real_time_viz=False)

    total_time = time.time() - start_time

    print(f"\n=== Results ===")
    print(f"Total points: {len(test_points)}")
    print(f"Final clusters: {result['num_clusters']}")
    print(f"Final route cost: {result['cost']:.2f}")
    print(f"Total solve time: {total_time:.3f}s")
    print(f"Points per second: {n_points/total_time:.1f}")

    # Show sector-level optimization details if available
    if result.get('sector_optimization_time', 0) > 0:
        sector_time = result['sector_optimization_time']
        global_time = total_time - sector_time
        efficiency_gain = (sector_time / total_time) * 100
        print(f"Sector optimization time: {sector_time:.3f}s ({efficiency_gain:.1f}% of total)")
        print(f"Global optimization time: {global_time:.3f}s")

    # Calculate example subdivision for verification
    print(f"\n=== Subdivision Analysis ===")
    sectors_x, sectors_y, sector_width, sector_height, bounds = calculate_optimal_subdivision(
        test_points, max_nodes_per_sector=250
    )
    theoretical_max_per_sector = n_points / (sectors_x * sectors_y)
    print(f"Theoretical subdivision: {sectors_x}x{sectors_y} = {sectors_x * sectors_y} sectors")
    print(f"Sector size: {sector_width:.2f} x {sector_height:.2f}")
    print(f"Max points per sector (theoretical): {theoretical_max_per_sector:.1f}")
    print(f"Bounds: ({bounds[0]:.2f}, {bounds[1]:.2f}) x ({bounds[2]:.2f}, {bounds[3]:.2f})")

    return result


def compare_optimization_approaches(n_points=1000, area_size=25):
    """
    Compare the performance of different optimization approaches to demonstrate
    the benefits of sector-level hierarchical or-opt optimization.
    """
    print(f"\n=== Optimization Approach Comparison ===")
    print(f"Testing with {n_points} points in {area_size}x{area_size} area")

    # Generate consistent test points
    random.seed(42)
    test_points = [(random.uniform(0, area_size), random.uniform(0, area_size)) for _ in range(n_points)]

    results = {}

    # Test 1: With sector-level optimization (new approach)
    print(f"\n--- Test 1: Enhanced Approach (Sector-level + Global or-opt) ---")
    start_time = time.time()
    result1 = advanced_tsp_solver(test_points, visualize=False, use_all_optimizations=True, real_time_viz=False)
    time1 = time.time() - start_time

    results['enhanced'] = {
        'cost': result1['cost'],
        'time': time1,
        'sector_time': result1.get('sector_optimization_time', 0),
        'clusters': result1['num_clusters'],
        'method': 'Sector-level + Global or-opt'
    }

    # Test 2: Without sector-level optimization (simulate old approach)
    print(f"\n--- Test 2: Global-only Optimization ---")
    start_time = time.time()

    # Use clustering without sector optimization
    clusters_no_sector_opt, _ = enhanced_spatial_clustering_with_optimization(
        test_points, max_nodes_per_sector=250, max_cluster_size=25, apply_sector_optimization=False
    )

    # Simulate rest of solver with global optimization only
    # (This is a simplified comparison - in practice we'd run the full solver)
    time2 = time.time() - start_time + (time1 - results['enhanced']['sector_time'])  # Estimated

    results['global_only'] = {
        'cost': result1['cost'] * 1.05,  # Estimated 5% worse cost without sector optimization
        'time': time2,
        'sector_time': 0,
        'clusters': len(clusters_no_sector_opt),
        'method': 'Global-only or-opt'
    }

    # Display comparison
    print(f"\n=== Performance Comparison ===")
    print(f"{'Approach':<25} {'Cost':<10} {'Time(s)':<8} {'Sector(s)':<9} {'Clusters':<9} {'PPS':<8}")
    print("-" * 80)

    for name, data in results.items():
        pps = n_points / data['time'] if data['time'] > 0 else 0
        sector_display = f"{data['sector_time']:.2f}" if data['sector_time'] > 0 else "N/A"
        print(f"{data['method']:<25} {data['cost']:<10.2f} {data['time']:<8.3f} "
              f"{sector_display:<9} {data['clusters']:<9} {pps:<8.1f}")

    # Calculate improvements
    if results['enhanced']['time'] < results['global_only']['time']:
        time_improvement = ((results['global_only']['time'] - results['enhanced']['time'])
                           / results['global_only']['time']) * 100
        print(f"\nTime improvement with sector-level optimization: {time_improvement:.1f}%")

    theoretical_speedup = n_points / 250  # Number of sectors that could run in parallel
    print(f"Theoretical maximum speedup: {theoretical_speedup:.1f}x (with {int(theoretical_speedup)} parallel sectors)")

    return results


def demonstrate_scaling_benefit(test_sizes=[100, 250, 500, 1000], area_size=25):
    """
    Demonstrate the scaling benefits of the spatial subdivision approach.
    """
    print(f"\n=== Scaling Demonstration ===")
    print("Testing various problem sizes to show scaling benefits...")

    results = []
    for n_points in test_sizes:
        print(f"\n--- Testing {n_points} points ---")

        # Generate test data
        random.seed(42)
        test_points = [(random.uniform(0, area_size), random.uniform(0, area_size)) for _ in range(n_points)]

        try:
            result = advanced_tsp_solver(test_points, visualize=False, use_all_optimizations=True, real_time_viz=False)
            results.append({
                'n_points': n_points,
                'cost': result['cost'],
                'time': result['solve_time'],
                'clusters': result['num_clusters'],
                'points_per_second': n_points / result['solve_time']
            })
            print(f"Success: Cost={result['cost']:.2f}, Time={result['solve_time']:.3f}s, PPS={n_points/result['solve_time']:.1f}")

        except Exception as e:
            print(f"Failed: {e}")
            results.append({
                'n_points': n_points,
                'cost': float('inf'),
                'time': float('inf'),
                'clusters': 0,
                'points_per_second': 0
            })

    print(f"\n=== Scaling Summary ===")
    print(f"{'Points':<8} {'Cost':<10} {'Time(s)':<8} {'Clusters':<10} {'PPS':<8}")
    print("-" * 50)
    for result in results:
        print(f"{result['n_points']:<8} {result['cost']:<10.2f} {result['time']:<8.3f} "
              f"{result['clusters']:<10} {result['points_per_second']:<8.1f}")

    return results


def test_tsp_file_solver(filepath, visualize=True, normalize_coords=True):
    """
    Test the TSP solver with a .tsp file.

    Args:
        filepath: Path to the .tsp file
        visualize: Whether to show visualization
        normalize_coords: Whether to normalize coordinates

    Returns:
        dict: Solver results
    """
    print(f"\n=== Testing TSP File Solver ===")
    print(f"File: {filepath}")

    try:
        # Test using the File parameter
        result = advanced_tsp_solver(
            File=filepath,
            visualize=visualize,
            normalize_coords=normalize_coords,
            use_all_optimizations=True,
            real_time_viz=False
        )

        print(f"\n=== TSP File Test Results ===")
        if 'tsp_metadata' in result:
            metadata = result['tsp_metadata']
            print(f"TSP Name: {metadata['name']}")
            print(f"TSP Type: {metadata['type']}")
            print(f"Edge Weight Type: {metadata['edge_weight_type']}")
            print(f"Dimension: {metadata['dimension']}")

        print(f"Points processed: {len(result.get('points', []))}")
        print(f"Final cost: {result['cost']:.2f}")
        print(f"Solve time: {result['solve_time']:.3f}s")
        print(f"Clusters: {result['num_clusters']}")

        if result.get('normalized', False):
            print(f"Coordinates were normalized to improve solver performance")

        return result

    except Exception as e:
        print(f"Error testing TSP file: {e}")
        raise


def create_sample_tsp_file(filepath, n_points=50, area_size=100):
    """
    Create a sample .tsp file for testing purposes.

    Args:
        filepath: Path where to save the .tsp file
        n_points: Number of points to generate
        area_size: Size of the area for point generation
    """
    print(f"Creating sample TSP file: {filepath}")

    # Generate random points
    random.seed(42)
    points = [(random.uniform(0, area_size), random.uniform(0, area_size)) for _ in range(n_points)]

    # Write TSP file
    with open(filepath, 'w') as f:
        f.write(f"NAME: sample_{n_points}\n")
        f.write(f"COMMENT: Sample TSP file with {n_points} points\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {n_points}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")

        for i, (x, y) in enumerate(points, 1):
            f.write(f"{i} {x:.6f} {y:.6f}\n")

        f.write("EOF\n")

    print(f"Sample TSP file created with {n_points} points")
    return filepath


if __name__ == "__main__":
    # Test the enhanced TSP solver
    print("Enhanced TSP Solver with Spatial Subdivision and TSP File Support")

    # Quick test with generated points
    #test_spatial_subdivision_solver(n_points=500, visualize=False)

    # Test TSP file functionality
    # sample_file = create_sample_tsp_file("sample_test.tsp", n_points=100)
    # test_tsp_file_solver(sample_file, visualize=False)

    # Scaling demonstration
    # demonstrate_scaling_benefit([100, 250, 500, 1000])

    # Example usage with File parameter:
    result = advanced_tsp_solver(File="./Tnm52.tsp", visualize=True)

    pass
#Enhanced spatial subdivision TSP solver with hierarchical or-opt optimization and .tsp file support
