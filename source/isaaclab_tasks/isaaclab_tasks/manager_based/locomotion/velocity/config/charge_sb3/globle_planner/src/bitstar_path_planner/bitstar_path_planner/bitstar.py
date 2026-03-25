"""
Batch Informed Trees (BIT*) Path Planning Algorithm

BIT* is an asymptotically optimal sampling-based path planning algorithm
that combines the benefits of RRT* with batch processing and informed sampling.
"""

import numpy as np
import math
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class NodeState(Enum):
    """Node states in the search tree"""
    UNVISITED = 0
    IN_QUEUE = 1
    VISITED = 2


@dataclass
class Node:
    """Node in the search tree"""
    position: np.ndarray
    cost_to_come: float = float('inf')
    cost_to_go: float = float('inf')
    parent: Optional['Node'] = None
    children: List['Node'] = None
    state: NodeState = NodeState.UNVISITED
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    @property
    def total_cost(self) -> float:
        """Total estimated cost (g + h)"""
        return self.cost_to_come + self.cost_to_go
    
    def __hash__(self):
        return hash(tuple(self.position))
    
    def __eq__(self, other):
        return np.allclose(self.position, other.position)


class BITStar:
    """
    Batch Informed Trees (BIT*) Path Planner
    
    This implementation follows the BIT* algorithm which:
    1. Uses batch sampling with informed sets
    2. Maintains forward and reverse search trees
    3. Processes edges in batches for efficiency
    """
    
    def __init__(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        bounds: Tuple[np.ndarray, np.ndarray],
        collision_checker: Callable[[np.ndarray], bool],
        distance_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        max_batch_size: int = 100,
        max_iterations: int = 10000,
        goal_radius: float = 0.5,
        rewire_radius: float = 2.0,
        goal_bias: float = 0.05,
    ):
        """
        Initialize BIT* planner
        
        Args:
            start: Start position (n-dimensional array)
            goal: Goal position (n-dimensional array)
            bounds: Tuple of (min_bounds, max_bounds) for each dimension
            collision_checker: Function that returns True if position is collision-free
            distance_metric: Custom distance function (default: Euclidean)
            max_batch_size: Maximum number of samples per batch
            max_iterations: Maximum number of iterations
            goal_radius: Radius for goal region
            rewire_radius: Maximum radius for rewiring
            goal_bias: Probability of sampling goal directly
        """
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.bounds_min = np.array(bounds[0])
        self.bounds_max = np.array(bounds[1])
        self.collision_checker = collision_checker
        self.max_batch_size = max_batch_size
        self.max_iterations = max_iterations
        self.goal_radius = goal_radius
        self.rewire_radius = rewire_radius
        self.goal_bias = goal_bias
        
        # Distance metric
        if distance_metric is None:
            self.distance = self._euclidean_distance
        else:
            self.distance = distance_metric
        
        # Initialize trees
        self.start_node = Node(self.start, cost_to_come=0.0)
        self.goal_node = Node(self.goal, cost_to_go=0.0)
        self.start_node.cost_to_go = self.distance(self.start, self.goal)
        self.goal_node.cost_to_come = self.distance(self.start, self.goal)
        
        self.forward_tree: List[Node] = [self.start_node]
        self.reverse_tree: List[Node] = [self.goal_node]
        
        # Best solution found
        self.best_cost = float('inf')
        self.best_path: List[np.ndarray] = []
        
        # Statistics
        self.iterations = 0
        self.samples_generated = 0
        
    def _euclidean_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Euclidean distance metric"""
        return np.linalg.norm(a - b)
    
    def _sample_informed_set(self, current_best: float) -> np.ndarray:
        """
        Sample from the informed set (ellipsoid)
        
        The informed set is an ellipsoid defined by:
        - Foci: start and goal
        - Major axis length: current_best_cost
        """
        if current_best == float('inf'):
            # No solution yet, sample from entire space
            return self._sample_uniform()
        
        # Sample from ellipsoid
        c_min = self.distance(self.start, self.goal)
        if current_best < c_min:
            return self._sample_uniform()
        
        # Ellipsoid parameters
        center = (self.start + self.goal) / 2.0
        c_max = math.sqrt(current_best**2 - c_min**2)
        
        # Sample in unit sphere, then transform to ellipsoid
        if np.random.random() < self.goal_bias:
            return self.goal.copy()
        
        # Uniform sampling in ellipsoid
        # Simplified: sample in bounding box and check if in ellipsoid
        max_attempts = 100
        for _ in range(max_attempts):
            sample = self._sample_uniform()
            dist_to_start = self.distance(sample, self.start)
            dist_to_goal = self.distance(sample, self.goal)
            if dist_to_start + dist_to_goal <= current_best:
                return sample
        
        return self._sample_uniform()
    
    def _sample_uniform(self) -> np.ndarray:
        """Uniform random sampling"""
        return np.random.uniform(self.bounds_min, self.bounds_max)
    
    def _nearest_neighbor(self, query: np.ndarray, tree: List[Node]) -> Node:
        """Find nearest neighbor in tree"""
        if not tree:
            return None
        
        min_dist = float('inf')
        nearest = None
        
        for node in tree:
            dist = self.distance(query, node.position)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        return nearest
    
    def _near_nodes(self, query: np.ndarray, tree: List[Node], radius: float) -> List[Node]:
        """Find all nodes within radius"""
        near_nodes = []
        for node in tree:
            if self.distance(query, node.position) <= radius:
                near_nodes.append(node)
        return near_nodes
    
    def _steer(self, from_pos: np.ndarray, to_pos: np.ndarray, max_step: float) -> np.ndarray:
        """Steer towards target with maximum step size"""
        direction = to_pos - from_pos
        dist = np.linalg.norm(direction)
        
        if dist <= max_step:
            return to_pos.copy()
        
        return from_pos + (direction / dist) * max_step
    
    def _is_collision_free(self, from_pos: np.ndarray, to_pos: np.ndarray, 
                          num_checks: int = 10) -> bool:
        """Check if path between two positions is collision-free"""
        for i in range(num_checks + 1):
            alpha = i / num_checks
            check_pos = from_pos * (1 - alpha) + to_pos * alpha
            if not self.collision_checker(check_pos):
                return False
        return True
    
    def _add_node_to_tree(self, new_node: Node, parent: Node, tree: List[Node]):
        """Add node to tree and update parent-child relationship"""
        new_node.parent = parent
        parent.children.append(new_node)
        tree.append(new_node)
    
    def _rewire(self, node: Node, near_nodes: List[Node], tree: List[Node], 
                is_forward: bool):
        """Rewire tree to potentially improve path"""
        for near_node in near_nodes:
            if near_node == node.parent:
                continue
            
            # Check if rewiring would improve cost
            new_cost = node.cost_to_come + self.distance(node.position, near_node.position)
            
            if is_forward:
                if new_cost < near_node.cost_to_come:
                    # Check collision
                    if self._is_collision_free(node.position, near_node.position):
                        # Remove from old parent
                        if near_node.parent:
                            near_node.parent.children.remove(near_node)
                        
                        # Update parent and cost
                        near_node.parent = node
                        node.children.append(near_node)
                        self._update_cost_to_come(near_node, tree, is_forward)
            else:
                if new_cost < near_node.cost_to_go:
                    if self._is_collision_free(node.position, near_node.position):
                        if near_node.parent:
                            near_node.parent.children.remove(near_node)
                        
                        near_node.parent = node
                        node.children.append(near_node)
                        self._update_cost_to_come(near_node, tree, is_forward)
    
    def _update_cost_to_come(self, node: Node, tree: List[Node], is_forward: bool):
        """Recursively update cost-to-come for node and descendants"""
        if node.parent:
            node.cost_to_come = node.parent.cost_to_come + \
                               self.distance(node.parent.position, node.position)
        else:
            node.cost_to_come = 0.0 if is_forward else float('inf')
        
        # Update cost-to-go
        if is_forward:
            node.cost_to_go = self.distance(node.position, self.goal)
        else:
            node.cost_to_go = self.distance(node.position, self.start)
        
        # Recursively update children
        for child in node.children:
            self._update_cost_to_come(child, tree, is_forward)
    
    def _try_connect_trees(self):
        """Try to connect forward and reverse trees"""
        # Find nodes in each tree that are close to each other
        connection_radius = self.rewire_radius
        
        for forward_node in self.forward_tree:
            for reverse_node in self.reverse_tree:
                dist = self.distance(forward_node.position, reverse_node.position)
                
                if dist <= connection_radius:
                    # Check if connection is collision-free
                    if self._is_collision_free(forward_node.position, reverse_node.position):
                        # Calculate total cost
                        total_cost = forward_node.cost_to_come + dist + reverse_node.cost_to_go
                        
                        if total_cost < self.best_cost:
                            self.best_cost = total_cost
                            # Reconstruct path
                            self._reconstruct_path(forward_node, reverse_node)
    
    def _reconstruct_path(self, forward_node: Node, reverse_node: Node):
        """Reconstruct path from start to goal"""
        path = []
        
        # Forward path (start to connection point)
        node = forward_node
        forward_path = []
        while node:
            forward_path.append(node.position)
            node = node.parent
        forward_path.reverse()
        
        # Reverse path (connection point to goal)
        node = reverse_node
        reverse_path = []
        while node:
            reverse_path.append(node.position)
            node = node.parent
        
        # Combine paths
        self.best_path = forward_path + reverse_path
    
    def plan(self) -> Tuple[bool, List[np.ndarray], float]:
        """
        Execute BIT* planning
        
        Returns:
            Tuple of (success, path, cost)
        """
        # Check if start and goal are collision-free
        if not self.collision_checker(self.start):
            return False, [], float('inf')
        
        if not self.collision_checker(self.goal):
            return False, [], float('inf')
        
        # Check if start and goal are already connected
        if self.distance(self.start, self.goal) <= self.goal_radius:
            if self._is_collision_free(self.start, self.goal):
                self.best_path = [self.start, self.goal]
                self.best_cost = self.distance(self.start, self.goal)
                return True, self.best_path, self.best_cost
        
        # Main planning loop
        for iteration in range(self.max_iterations):
            self.iterations = iteration + 1
            
            # Generate batch of samples
            batch_size = min(self.max_batch_size, 
                           self.max_iterations - iteration)
            
            samples = []
            for _ in range(batch_size):
                sample = self._sample_informed_set(self.best_cost)
                if self.collision_checker(sample):
                    samples.append(sample)
                    self.samples_generated += 1
            
            # Process samples
            for sample_pos in samples:
                # Try to add to forward tree
                nearest_forward = self._nearest_neighbor(sample_pos, self.forward_tree)
                if nearest_forward:
                    steered_pos = self._steer(nearest_forward.position, sample_pos, 
                                             self.rewire_radius)
                    
                    if self._is_collision_free(nearest_forward.position, steered_pos):
                        new_node = Node(steered_pos)
                        new_node.cost_to_come = nearest_forward.cost_to_come + \
                                               self.distance(nearest_forward.position, steered_pos)
                        new_node.cost_to_go = self.distance(steered_pos, self.goal)
                        
                        # Try to find better parent
                        near_nodes = self._near_nodes(steered_pos, self.forward_tree, 
                                                     self.rewire_radius)
                        best_parent = nearest_forward
                        best_cost = new_node.cost_to_come
                        
                        for near_node in near_nodes:
                            if near_node == nearest_forward:
                                continue
                            potential_cost = near_node.cost_to_come + \
                                           self.distance(near_node.position, steered_pos)
                            if potential_cost < best_cost:
                                if self._is_collision_free(near_node.position, steered_pos):
                                    best_parent = near_node
                                    best_cost = potential_cost
                        
                        new_node.cost_to_come = best_cost
                        self._add_node_to_tree(new_node, best_parent, self.forward_tree)
                        
                        # Rewire
                        self._rewire(new_node, near_nodes, self.forward_tree, True)
                
                # Try to add to reverse tree
                nearest_reverse = self._nearest_neighbor(sample_pos, self.reverse_tree)
                if nearest_reverse:
                    steered_pos = self._steer(nearest_reverse.position, sample_pos, 
                                             self.rewire_radius)
                    
                    if self._is_collision_free(nearest_reverse.position, steered_pos):
                        new_node = Node(steered_pos)
                        new_node.cost_to_go = nearest_reverse.cost_to_go + \
                                            self.distance(steered_pos, nearest_reverse.position)
                        new_node.cost_to_come = self.distance(steered_pos, self.start)
                        
                        near_nodes = self._near_nodes(steered_pos, self.reverse_tree, 
                                                     self.rewire_radius)
                        best_parent = nearest_reverse
                        best_cost = new_node.cost_to_go
                        
                        for near_node in near_nodes:
                            if near_node == nearest_reverse:
                                continue
                            potential_cost = near_node.cost_to_go + \
                                           self.distance(steered_pos, near_node.position)
                            if potential_cost < best_cost:
                                if self._is_collision_free(steered_pos, near_node.position):
                                    best_parent = near_node
                                    best_cost = potential_cost
                        
                        new_node.cost_to_go = best_cost
                        self._add_node_to_tree(new_node, best_parent, self.reverse_tree)
                        
                        # Rewire
                        self._rewire(new_node, near_nodes, self.reverse_tree, False)
            
            # Try to connect trees
            self._try_connect_trees()
            
            # Check if goal is reached
            if self.best_cost < float('inf'):
                # Check if we can improve further
                if len(self.best_path) > 0:
                    # Early termination if solution is good enough
                    if iteration > 100 and self.best_cost < self.distance(self.start, self.goal) * 1.1:
                        break
        
        # Final check
        if len(self.best_path) > 0:
            return True, self.best_path, self.best_cost
        
        return False, [], float('inf')
