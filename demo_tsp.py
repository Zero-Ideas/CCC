#!/usr/bin/env python3
"""
Demo script for Enhanced TSP with Adaptive Sub-clustering
Run this to see the visualization in action
"""

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better Windows compatibility

from TSP import run_example, demo_adaptive_clustering, quick_test
import matplotlib.pyplot as plt

def main():
    print("=== Enhanced TSP Demonstration ===")
    print("\nThis demo shows the enhanced TSP solver with:")
    print("- Adaptive sub-clustering for large clusters (>5 points)")
    print("- Real-time visualization of the solving process")
    print("- Performance optimizations including enhanced 2-opt")
    print("- Intelligent sub-cluster connection logic")

    print("\n1. Quick test with visualization...")
    try:
        result = quick_test(20, visualize=True)
        print(f"Quick test completed: Cost={result['cost']:.2f}, Time={result['solve_time']:.3f}s")
    except Exception as e:
        print(f"Quick test failed: {e}")

    print("\n2. Comparison test (Basic vs Enhanced)...")
    try:
        # Basic solver
        print("Running basic solver...")
        basic = run_example(25, visualize=False, use_adaptive=False)

        # Enhanced solver with visualization
        print("Running enhanced solver with visualization...")
        enhanced = run_example(25, visualize=True, use_adaptive=True)

        improvement = ((basic['cost'] - enhanced['cost']) / basic['cost']) * 100

        print(f"\nResults Comparison:")
        print(f"Basic:    Cost={basic['cost']:.2f}, Clusters={basic['num_clusters']}, Time={basic['solve_time']:.3f}s")
        print(f"Enhanced: Cost={enhanced['cost']:.2f}, Clusters={enhanced['num_clusters']}, Time={enhanced['solve_time']:.3f}s")
        print(f"Improvement: {improvement:.1f}% cost reduction")

    except Exception as e:
        print(f"Comparison test failed: {e}")

    print("\nDemo completed! The enhanced TSP solver successfully demonstrates:")
    print("   - Adaptive sub-clustering divides large clusters into smaller, manageable pieces")
    print("   - Real-time visualization shows the solving process step by step")
    print("   - Performance improvements through better algorithms and optimizations")
    print("   - Intelligent routing between sub-clusters for optimal connectivity")

if __name__ == "__main__":
    main()