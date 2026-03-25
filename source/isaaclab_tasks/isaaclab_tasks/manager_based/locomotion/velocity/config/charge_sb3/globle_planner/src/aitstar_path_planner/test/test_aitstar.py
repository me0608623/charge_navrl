#!/usr/bin/env python3
"""
Simple test script for AIT* algorithm
"""

import sys
import os

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from aitstar_path_planner.aitstar import AITStar


def simple_collision_checker(position):
    """Simple collision checker with circular obstacles"""
    obstacles = [
        (2.0, 2.0, 1.0),   # (x, y, radius)
        (5.0, 5.0, 1.5),
        (8.0, 3.0, 1.0),
    ]
    
    for ox, oy, r in obstacles:
        dist = np.sqrt((position[0] - ox)**2 + (position[1] - oy)**2)
        if dist < r:
            return False
    
    return True


def test_aitstar():
    """Test AIT* planner with simple 2D environment"""
    print("Testing AIT* Path Planner...")
    print("=" * 50)
    
    # Define start and goal
    start = np.array([0.0, 0.0])
    goal = np.array([10.0, 10.0])
    
    print(f"Start: {start}")
    print(f"Goal: {goal}")
    
    # Define bounds
    bounds_min = np.array([-2.0, -2.0])
    bounds_max = np.array([12.0, 12.0])
    
    # Create planner
    planner = AITStar(
        start=start,
        goal=goal,
        bounds=(bounds_min, bounds_max),
        collision_checker=simple_collision_checker,
        max_iterations=2000,
        max_batch_size=50,
        goal_radius=0.5,
        rewire_radius=2.0,
        goal_bias=0.05,
        adaptive_factor=1.2,
        min_informed_ratio=0.1,
    )
    
    print("\nStarting planning...")
    success, path, cost = planner.plan()
    
    print("\n" + "=" * 50)
    if success:
        print(f"✓ Path found!")
        print(f"  Cost: {cost:.2f}")
        print(f"  Path length: {len(path)} waypoints")
        print(f"  Iterations: {planner.iterations}")
        print(f"  Samples generated: {planner.samples_generated}")
        print(f"  Informed samples: {planner.informed_samples}")
        print(f"  Informed ratio: {planner.informed_samples/max(planner.samples_generated, 1)*100:.1f}%")
        print(f"  Forward tree nodes: {len(planner.forward_tree)}")
        print(f"  Reverse tree nodes: {len(planner.reverse_tree)}")
        print(f"  Search progress: {planner.search_progress*100:.1f}%")
        print(f"\nPath waypoints:")
        for i, point in enumerate(path[::max(1, len(path)//10)]):  # Show every 10th point
            print(f"  [{i*10}]: ({point[0]:.2f}, {point[1]:.2f})")
    else:
        print("✗ Path planning failed")
        print(f"  Iterations: {planner.iterations}")
        print(f"  Samples generated: {planner.samples_generated}")
        return False
    
    return True


if __name__ == '__main__':
    success = test_aitstar()
    sys.exit(0 if success else 1)
