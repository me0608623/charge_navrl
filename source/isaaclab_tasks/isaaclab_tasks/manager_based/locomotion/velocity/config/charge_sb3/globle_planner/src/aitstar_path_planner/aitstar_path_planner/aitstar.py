#!/usr/bin/env python3
"""
AIT* (Adaptive Informed Trees) 路徑規劃演算法
============================================

AIT* 結合了 RRT* 的快速探索能力與 Informed RRT* 的啟發式採樣優化，
並透過批次處理與自適應採樣策略實現高效的最優路徑規劃。

主要特點：
- 啟發式採樣：在橢球區域內進行知情採樣
- 批次處理：批量新增頂點以提高效率
- 自適應策略：根據搜尋進度動態調整採樣區域
- 漸近最優：保證收斂至最優解

參考文獻：
Strub, M. P., & Gammell, J. D. (2020). 
"Adaptively Informed Trees (AIT*): Fast Asymptotically Optimal Path Planning 
through Adaptive Heuristics"

作者: Claude
版本: 1.0.0
"""

from __future__ import annotations

import math
import heapq
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict, Callable
from abc import ABC, abstractmethod
import numpy as np


# =============================================================================
# 資料結構定義
# =============================================================================

@dataclass
class State:
    """狀態空間中的一個點"""
    position: np.ndarray
    
    def __hash__(self):
        return hash(tuple(self.position))
    
    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return np.allclose(self.position, other.position)
    
    def distance_to(self, other) -> float:
        """計算到另一個狀態的歐氏距離

        Args:
            other: 可以是 State 或 Vertex 對象
        """
        # 獲取自己的位置
        my_pos = np.array(self.position) if not isinstance(self.position, np.ndarray) else self.position

        # 獲取對方的位置
        if isinstance(other, State):
            other_pos = np.array(other.position) if not isinstance(other.position, np.ndarray) else other.position
        elif hasattr(other, 'state'):  # Vertex 對象
            other_pos = np.array(other.state.position) if not isinstance(other.state.position, np.ndarray) else other.state.position
        else:
            other_pos = np.array(other)

        return np.linalg.norm(my_pos - other_pos)


@dataclass
class Vertex:
    """圖中的頂點"""
    state: State
    parent: Optional['Vertex'] = None
    children: Set['Vertex'] = field(default_factory=set)
    cost_to_come: float = float('inf')  # g(v): 從起點到此頂點的代價
    cost_to_go: float = float('inf')     # h(v): 從此頂點到終點的啟發式估計
    
    def __hash__(self):
        return hash(self.state)
    
    def __eq__(self, other):
        if not isinstance(other, Vertex):
            return False
        return self.state == other.state
    
    def __lt__(self, other):
        """用於優先佇列排序"""
        return self.get_key() < other.get_key()
    
    def get_key(self) -> float:
        """取得頂點的排序鍵值 (f = g + h)"""
        return self.cost_to_come + self.cost_to_go


@dataclass
class Edge:
    """圖中的邊"""
    source: Vertex
    target: Vertex
    cost: float
    
    def __hash__(self):
        return hash((self.source, self.target))
    
    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return self.source == other.source and self.target == other.target
    
    def __lt__(self, other):
        """用於優先佇列排序"""
        return self.get_key() < other.get_key()
    
    def get_key(self) -> float:
        """取得邊的排序鍵值"""
        return (self.source.cost_to_come + self.cost + 
                self.target.cost_to_go)


# =============================================================================
# 碰撞檢測介面
# =============================================================================

class CollisionChecker(ABC):
    """碰撞檢測抽象介面"""
    
    @abstractmethod
    def is_state_valid(self, state: State) -> bool:
        """檢查狀態是否有效（無碰撞）"""
        pass
    
    @abstractmethod
    def is_edge_valid(self, state1: State, state2: State) -> bool:
        """檢查兩狀態之間的邊是否有效（無碰撞）"""
        pass


class SimpleCollisionChecker(CollisionChecker):
    """簡單的圓形障礙物碰撞檢測器"""
    
    def __init__(self, 
                 obstacles: List[Tuple[np.ndarray, float]],
                 bounds: Tuple[np.ndarray, np.ndarray],
                 robot_radius: float = 0.0):
        """
        初始化碰撞檢測器
        
        Args:
            obstacles: 障礙物列表 [(中心點, 半徑), ...]
            bounds: 空間邊界 (最小值, 最大值)
            robot_radius: 機器人半徑
        """
        self.obstacles = obstacles
        self.bounds_min, self.bounds_max = bounds
        self.robot_radius = robot_radius
    
    def is_state_valid(self, state: State) -> bool:
        """檢查狀態是否在邊界內且無碰撞"""
        pos = state.position
        
        # 檢查邊界
        if np.any(pos < self.bounds_min) or np.any(pos > self.bounds_max):
            return False
        
        # 檢查障礙物碰撞
        for obs_center, obs_radius in self.obstacles:
            dist = np.linalg.norm(pos - obs_center)
            if dist <= obs_radius + self.robot_radius:
                return False
        
        return True
    
    def is_edge_valid(self, state1: State, state2: State, 
                      resolution: float = 0.1) -> bool:
        """
        檢查邊是否有效（沿路徑採樣檢測）
        
        Args:
            state1: 起始狀態
            state2: 終止狀態
            resolution: 採樣解析度
        """
        if not self.is_state_valid(state1) or not self.is_state_valid(state2):
            return False
        
        direction = state2.position - state1.position
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            return True
        
        n_steps = max(2, int(np.ceil(distance / resolution)))
        
        for i in range(1, n_steps):
            t = i / n_steps
            intermediate_pos = state1.position + t * direction
            intermediate_state = State(intermediate_pos)
            if not self.is_state_valid(intermediate_state):
                return False
        
        return True


# =============================================================================
# 採樣器
# =============================================================================

class Sampler:
    """狀態空間採樣器"""
    
    def __init__(self, 
                 bounds_min: np.ndarray, 
                 bounds_max: np.ndarray,
                 rng: Optional[np.random.Generator] = None):
        """
        初始化採樣器
        
        Args:
            bounds_min: 空間最小邊界
            bounds_max: 空間最大邊界
            rng: 隨機數生成器
        """
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max
        self.dimension = len(bounds_min)
        self.rng = rng if rng is not None else np.random.default_rng()
    
    def uniform_sample(self) -> State:
        """均勻隨機採樣"""
        position = self.rng.uniform(self.bounds_min, self.bounds_max)
        return State(position)
    
    def informed_sample(self, 
                       start: State, 
                       goal: State, 
                       c_best: float) -> State:
        """
        在橢球區域內進行知情採樣 (Informed Sampling)
        
        橢球定義：所有從 start 到 goal 經過的路徑長度 <= c_best 的點
        
        Args:
            start: 起始狀態
            goal: 目標狀態
            c_best: 當前最佳路徑代價
        """
        c_min = start.distance_to(goal)
        
        # 如果 c_best 無限大或接近 c_min，使用均勻採樣
        if c_best >= float('inf') or c_best <= c_min + 1e-6:
            return self.uniform_sample()
        
        # 計算橢球參數
        center = (start.position + goal.position) / 2.0
        
        # 主軸長度
        r1 = c_best / 2.0
        # 次軸長度
        ri = math.sqrt(c_best**2 - c_min**2) / 2.0
        
        # 計算旋轉矩陣
        a1 = (goal.position - start.position) / c_min
        rotation = self._rotation_to_world_frame(a1)
        
        # 在單位球內採樣
        x_ball = self._sample_unit_ball()
        
        # 縮放到橢球
        radii = np.array([r1] + [ri] * (self.dimension - 1))
        x_ellipse = np.diag(radii) @ x_ball
        
        # 旋轉並平移
        position = rotation @ x_ellipse + center
        
        # 確保在邊界內
        position = np.clip(position, self.bounds_min, self.bounds_max)
        
        return State(position)
    
    def _sample_unit_ball(self) -> np.ndarray:
        """在單位球內均勻採樣"""
        # 使用拒絕採樣
        while True:
            point = self.rng.uniform(-1, 1, self.dimension)
            if np.linalg.norm(point) <= 1.0:
                return point
    
    def _rotation_to_world_frame(self, a1: np.ndarray) -> np.ndarray:
        """
        計算從橢球坐標系到世界坐標系的旋轉矩陣
        
        Args:
            a1: 橢球主軸方向（世界坐標系）
        """
        # 使用 Rodrigues 旋轉公式
        e1 = np.zeros(self.dimension)
        e1[0] = 1.0
        
        # Gram-Schmidt 正交化建構完整的旋轉矩陣
        M = np.outer(a1, e1)
        U, _, Vt = np.linalg.svd(M)
        
        # 處理反射情況
        det = np.linalg.det(U @ Vt)
        diag = np.eye(self.dimension)
        diag[-1, -1] = det
        
        return U @ diag @ Vt


# =============================================================================
# AIT* 主演算法
# =============================================================================

class AITStar:
    """
    AIT* (Adaptive Informed Trees) 路徑規劃演算法
    
    AIT* 透過以下機制實現高效的漸近最優路徑規劃：
    1. 批次頂點新增：一次新增多個頂點以提高效率
    2. 知情採樣：在橢球區域內採樣以聚焦搜尋
    3. 自適應啟發式：動態更新啟發式估計
    4. 延遲碰撞檢測：優先檢查最有希望的邊
    """
    
    def __init__(self,
                 collision_checker: CollisionChecker,
                 bounds_min: np.ndarray,
                 bounds_max: np.ndarray,
                 batch_size: int = 100,
                 rewire_factor: float = 1.1,
                 goal_bias: float = 0.05,
                 max_iterations: int = 1000,
                 convergence_threshold: float = 1e-3,
                 seed: Optional[int] = None):
        """
        初始化 AIT* 規劃器
        
        Args:
            collision_checker: 碰撞檢測器
            bounds_min: 空間最小邊界
            bounds_max: 空間最大邊界
            batch_size: 每批次新增的頂點數
            rewire_factor: 重連半徑因子 (> 1.0)
            goal_bias: 目標偏向機率
            max_iterations: 最大迭代次數
            convergence_threshold: 收斂閾值
            seed: 隨機種子
        """
        self.collision_checker = collision_checker
        self.bounds_min = np.array(bounds_min)
        self.bounds_max = np.array(bounds_max)
        self.dimension = len(bounds_min)
        
        self.batch_size = batch_size
        self.rewire_factor = rewire_factor
        self.goal_bias = goal_bias
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        self.rng = np.random.default_rng(seed)
        self.sampler = Sampler(bounds_min, bounds_max, self.rng)
        
        # 規劃狀態
        self.vertices: Set[Vertex] = set()
        self.start_vertex: Optional[Vertex] = None
        self.goal_vertex: Optional[Vertex] = None
        self.solution_cost: float = float('inf')
        
        # 優先佇列
        self.edge_queue: List[Edge] = []
        self.vertex_queue: List[Vertex] = []
        
        # 統計資訊
        self.stats = {
            'iterations': 0,
            'vertices_sampled': 0,
            'edges_checked': 0,
            'collision_checks': 0,
            'solutions_found': 0
        }
    
    def plan(self,
             start: np.ndarray,
             goal: np.ndarray,
             initial_solution: Optional[List[np.ndarray]] = None
             ) -> Tuple[Optional[List[np.ndarray]], float]:
        """
        執行路徑規劃

        Args:
            start: 起始位置
            goal: 目標位置
            initial_solution: 初始解（可選，用於熱啟動）

        Returns:
            (路徑點列表, 路徑代價) 或 (None, inf) 如果找不到路徑
        """
        # 🔥 禁用調試輸出：初始化信息
        # print(f"[AIT* plan] 開始初始化...")
        self._initialize(start, goal)
        # print(f"[AIT* plan] 初始化完成，開始規劃循環...")

        if initial_solution is not None:
            self._set_initial_solution(initial_solution)

        iteration = 0
        prev_cost = float('inf')

        while iteration < self.max_iterations:
            iteration += 1
            self.stats['iterations'] = iteration

            # 🔥 禁用調試輸出：每 10 次迭代打印一次進度
            # if iteration % 10 == 0:
            #     print(f"[AIT* plan] 迭代 {iteration}/{self.max_iterations}, solution_cost={self.solution_cost:.4f}")

            # 檢查是否需要新增新批次
            should_add = self._should_add_batch()
            # 🔥 禁用調試輸出：批次信息
            # if iteration % 10 == 0:
            #     print(f"[AIT* plan] should_add_batch: {should_add}, vertices={len(self.vertices)}")

            if should_add:
                # 🔥 禁用調試輸出：批次添加
                # print(f"[AIT* plan] 正在添加新批次...")
                self._add_batch()
                # print(f"[AIT* plan] 批次添加完成，總頂點數: {len(self.vertices)}")

            # 處理頂點和邊佇列
            # 🔥 禁用調試輸出：佇列處理
            # print(f"[AIT* plan] 開始處理佇列...")
            self._process_queues()
            # print(f"[AIT* plan] 佇列處理完成")

            # 檢查收斂
            if self.solution_cost < float('inf') and prev_cost < float('inf'):
                improvement = (prev_cost - self.solution_cost) / max(prev_cost, 1e-6)
                if improvement < self.convergence_threshold and iteration > 10:
                    # 🔥 保留收斂信息（關鍵）
                    print(f"[AIT* plan] 收斂！迭代 {iteration}, improvement={improvement:.6f}")
                    break
                prev_cost = self.solution_cost

        # 🔥 保留規劃結果信息（關鍵）
        print(f"[AIT* plan] 規劃完成: {iteration} 次迭代, solution_cost={self.solution_cost:.4f}")
        return self._extract_solution()
    
    def _initialize(self, start: np.ndarray, goal: np.ndarray):
        """初始化規劃問題"""
        self.vertices.clear()
        self.edge_queue.clear()
        self.vertex_queue.clear()
        self.solution_cost = float('inf')
        
        # 建立起點和終點頂點
        start_state = State(np.array(start))
        goal_state = State(np.array(goal))
        
        self.start_vertex = Vertex(start_state)
        self.start_vertex.cost_to_come = 0.0
        self.start_vertex.cost_to_go = start_state.distance_to(goal_state)
        
        self.goal_vertex = Vertex(goal_state)
        self.goal_vertex.cost_to_come = float('inf')
        self.goal_vertex.cost_to_go = 0.0
        
        # 驗證起點和終點
        if not self.collision_checker.is_state_valid(start_state):
            raise ValueError("起始位置處於碰撞狀態")
        if not self.collision_checker.is_state_valid(goal_state):
            raise ValueError("目標位置處於碰撞狀態")
        
        self.vertices.add(self.start_vertex)
        self.vertices.add(self.goal_vertex)
        
        # 初始化頂點佇列
        heapq.heappush(self.vertex_queue, self.start_vertex)
    
    def _set_initial_solution(self, path: List[np.ndarray]):
        """設定初始解（熱啟動）"""
        if len(path) < 2:
            return
        
        prev_vertex = self.start_vertex
        total_cost = 0.0
        
        for i in range(1, len(path) - 1):
            state = State(np.array(path[i]))
            vertex = Vertex(state)
            
            cost = prev_vertex.state.distance_to(state)
            total_cost += cost
            
            vertex.cost_to_come = total_cost
            vertex.cost_to_go = state.distance_to(self.goal_vertex.state)
            vertex.parent = prev_vertex
            prev_vertex.children.add(vertex)
            
            self.vertices.add(vertex)
            prev_vertex = vertex
        
        # 連接到目標
        final_cost = prev_vertex.state.distance_to(self.goal_vertex.state)
        total_cost += final_cost
        
        if self.collision_checker.is_edge_valid(prev_vertex.state, 
                                                 self.goal_vertex.state):
            self.goal_vertex.parent = prev_vertex
            self.goal_vertex.cost_to_come = total_cost
            prev_vertex.children.add(self.goal_vertex)
            self.solution_cost = total_cost
            self.stats['solutions_found'] += 1
    
    def _should_add_batch(self) -> bool:
        """判斷是否需要新增新的頂點批次"""
        # 當頂點佇列和邊佇列都為空時新增批次
        return len(self.vertex_queue) == 0 and len(self.edge_queue) == 0
    
    def _add_batch(self):
        """新增一批新的頂點"""
        new_vertices = []
        
        for _ in range(self.batch_size):
            # 採樣新狀態
            if self.rng.random() < self.goal_bias:
                state = self.goal_vertex.state
            elif self.solution_cost < float('inf'):
                # 知情採樣
                state = self.sampler.informed_sample(
                    self.start_vertex.state,
                    self.goal_vertex.state,
                    self.solution_cost
                )
            else:
                # 均勻採樣
                state = self.sampler.uniform_sample()
            
            self.stats['vertices_sampled'] += 1
            
            # 檢查狀態有效性
            if not self.collision_checker.is_state_valid(state):
                self.stats['collision_checks'] += 1
                continue
            
            # 建立新頂點
            vertex = Vertex(state)
            vertex.cost_to_come = float('inf')
            vertex.cost_to_go = state.distance_to(self.goal_vertex.state)
            
            # 檢查是否已存在相似頂點
            if self._find_similar_vertex(state) is not None:
                continue
            
            new_vertices.append(vertex)
            self.vertices.add(vertex)
        
        # 將新頂點加入佇列
        for vertex in new_vertices:
            heapq.heappush(self.vertex_queue, vertex)
        
        # 為現有頂點更新潛在邊
        self._update_edge_queue()
    
    def _find_similar_vertex(self, state: State, 
                             threshold: float = 1e-3) -> Optional[Vertex]:
        """尋找與給定狀態相近的現有頂點"""
        for vertex in self.vertices:
            if vertex.state.distance_to(state) < threshold:
                return vertex
        return None
    
    def _update_edge_queue(self):
        """更新邊優先佇列"""
        self.edge_queue.clear()
        
        for vertex in self.vertices:
            if vertex.cost_to_come >= float('inf'):
                continue
            
            # 計算連接半徑
            r = self._get_connection_radius()
            
            # 尋找鄰近頂點並新增潛在邊
            for neighbor in self.vertices:
                if neighbor == vertex:
                    continue
                if neighbor in vertex.children:
                    continue
                
                dist = vertex.state.distance_to(neighbor.state)
                if dist > r:
                    continue
                
                # 計算邊的代價
                edge_cost = vertex.cost_to_come + dist + neighbor.cost_to_go
                
                # 剪枝：如果邊代價已超過當前最佳解，跳過
                if edge_cost >= self.solution_cost:
                    continue
                
                edge = Edge(vertex, neighbor, dist)
                heapq.heappush(self.edge_queue, edge)
    
    def _get_connection_radius(self) -> float:
        """計算 RRT* 風格的連接半徑"""
        n = len(self.vertices)
        if n < 2:
            return float('inf')
        
        # 計算最優半徑
        # r = gamma * (log(n) / n)^(1/d)
        gamma = self.rewire_factor * 2 * (1 + 1/self.dimension) ** (1/self.dimension)
        volume = np.prod(self.bounds_max - self.bounds_min)
        unit_ball_volume = (math.pi ** (self.dimension / 2)) / math.gamma(self.dimension / 2 + 1)
        
        r = gamma * ((volume / unit_ball_volume) * (math.log(n) / n)) ** (1 / self.dimension)
        
        return min(r, np.linalg.norm(self.bounds_max - self.bounds_min))
    
    def _process_queues(self):
        """處理頂點和邊佇列"""
        # 限制每次處理的數量以保持響應性
        max_operations = self.batch_size
        operations = 0

        while operations < max_operations:
            operations += 1

            # 🔥 禁用調試輸出：每 10 次操作打印一次調試信息
            # if operations % 10 == 0:
            #     print(f"[AIT* _process_queues] 操作 {operations}/{max_operations}, "
            #           f"vertex_queue={len(self.vertex_queue)}, edge_queue={len(self.edge_queue)}, "
            #           f"solution_cost={self.solution_cost:.4f}")

            # 選擇處理頂點還是邊
            if self.edge_queue and self.vertex_queue:
                # 比較最佳頂點和最佳邊
                best_edge = self.edge_queue[0]
                best_vertex = self.vertex_queue[0]

                if best_vertex.get_key() <= best_edge.get_key():
                    self._expand_vertex()
                else:
                    self._process_edge()
            elif self.vertex_queue:
                self._expand_vertex()
            elif self.edge_queue:
                self._process_edge()
            else:
                break

        # 🔥 禁用調試輸出：處理完成後打印統計
        # if operations >= max_operations:
        #     print(f"[AIT* _process_queues] 完成一批處理，操作數: {operations}")
    
    def _expand_vertex(self):
        """擴展頂點佇列中的最佳頂點"""
        if not self.vertex_queue:
            return
        
        vertex = heapq.heappop(self.vertex_queue)
        
        # 如果頂點代價已超過當前最佳解，跳過
        if vertex.get_key() >= self.solution_cost:
            return
        
        # 尋找可連接的鄰居
        r = self._get_connection_radius()
        
        for neighbor in self.vertices:
            if neighbor == vertex:
                continue
            
            dist = vertex.state.distance_to(neighbor.state)
            if dist > r:
                continue
            
            # 建立邊並加入佇列
            potential_cost = vertex.cost_to_come + dist + neighbor.cost_to_go
            
            if potential_cost < self.solution_cost:
                edge = Edge(vertex, neighbor, dist)
                heapq.heappush(self.edge_queue, edge)
    
    def _process_edge(self):
        """處理邊佇列中的最佳邊"""
        if not self.edge_queue:
            return
        
        edge = heapq.heappop(self.edge_queue)
        self.stats['edges_checked'] += 1
        
        # 檢查邊是否仍有改進潛力
        if edge.get_key() >= self.solution_cost:
            return
        
        source, target = edge.source, edge.target
        new_cost = source.cost_to_come + edge.cost
        
        # 如果沒有改進，跳過
        if new_cost >= target.cost_to_come:
            return
        
        # 執行碰撞檢測
        self.stats['collision_checks'] += 1
        if not self.collision_checker.is_edge_valid(source.state, target.state):
            return
        
        # 更新連接
        if target.parent is not None:
            target.parent.children.discard(target)
        
        target.parent = source
        target.cost_to_come = new_cost
        source.children.add(target)
        
        # 如果連接到目標，更新最佳解
        if target == self.goal_vertex:
            if new_cost < self.solution_cost:
                self.solution_cost = new_cost
                self.stats['solutions_found'] += 1
        
        # 傳播代價更新到子節點
        self._propagate_cost_update(target)
    
    def _propagate_cost_update(self, vertex: Vertex):
        """遞迴傳播代價更新到子節點"""
        for child in vertex.children:
            new_cost = vertex.cost_to_come + vertex.state.distance_to(child.state)
            if new_cost < child.cost_to_come:
                child.cost_to_come = new_cost
                
                if child == self.goal_vertex:
                    if new_cost < self.solution_cost:
                        self.solution_cost = new_cost
                
                self._propagate_cost_update(child)
    
    def _extract_solution(self) -> Tuple[Optional[List[np.ndarray]], float]:
        """提取最終路徑"""
        # 🔥 禁用調試輸出：詳細檢查信息
        # print(f"[AIT* _extract_solution] goal_vertex.parent is None: {self.goal_vertex.parent is None}")
        if self.goal_vertex.parent is None:
            # 🔥 保留錯誤信息（關鍵）
            print(f"[AIT* _extract_solution] ⚠️  沒有找到路徑！")
            return None, float('inf')

        path = []
        current = self.goal_vertex

        while current is not None:
            path.append(current.state.position.copy())
            current = current.parent

        path.reverse()
        # 🔥 保留成功信息（關鍵）
        print(f"[AIT* _extract_solution] ✓ 路徑提取成功，路徑點數: {len(path)}")
        return path, self.solution_cost
    
    def get_tree(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """取得搜尋樹的所有邊（用於視覺化）"""
        edges = []
        for vertex in self.vertices:
            if vertex.parent is not None:
                edges.append((
                    vertex.parent.state.position.copy(),
                    vertex.state.position.copy()
                ))
        return edges
    
    def get_statistics(self) -> Dict:
        """取得規劃統計資訊"""
        return {
            **self.stats,
            'total_vertices': len(self.vertices),
            'solution_cost': self.solution_cost
        }


# =============================================================================
# 路徑平滑與後處理
# =============================================================================

class PathSmoother:
    """路徑平滑處理器"""
    
    def __init__(self, collision_checker: CollisionChecker):
        """
        初始化路徑平滑器
        
        Args:
            collision_checker: 碰撞檢測器
        """
        self.collision_checker = collision_checker
    
    def shortcut(self, path: List[np.ndarray], 
                 max_iterations: int = 100) -> List[np.ndarray]:
        """
        路徑捷徑優化：嘗試跳過中間點以縮短路徑
        
        Args:
            path: 原始路徑
            max_iterations: 最大迭代次數
        """
        if len(path) < 3:
            return path
        
        smoothed = [p.copy() for p in path]
        
        for _ in range(max_iterations):
            if len(smoothed) < 3:
                break
            
            # 隨機選擇兩個非相鄰點
            i = random.randint(0, len(smoothed) - 3)
            j = random.randint(i + 2, len(smoothed) - 1)
            
            # 嘗試直接連接
            state_i = State(smoothed[i])
            state_j = State(smoothed[j])
            
            if self.collision_checker.is_edge_valid(state_i, state_j):
                # 移除中間點
                smoothed = smoothed[:i+1] + smoothed[j:]
        
        return smoothed
    
    def interpolate(self, path: List[np.ndarray], 
                    resolution: float = 0.1) -> List[np.ndarray]:
        """
        路徑插值：在路徑點之間插入中間點
        
        Args:
            path: 原始路徑
            resolution: 插值解析度
        """
        if len(path) < 2:
            return path
        
        interpolated = [path[0].copy()]
        
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            
            direction = end - start
            distance = np.linalg.norm(direction)
            
            if distance < 1e-6:
                continue
            
            n_segments = max(1, int(np.ceil(distance / resolution)))
            
            for j in range(1, n_segments + 1):
                t = j / n_segments
                point = start + t * direction
                interpolated.append(point.copy())
        
        return interpolated


# =============================================================================
# 測試與範例
# =============================================================================

def create_test_environment():
    """建立測試環境"""
    # 空間邊界
    bounds_min = np.array([0.0, 0.0])
    bounds_max = np.array([10.0, 10.0])
    
    # 障礙物 (圓形)
    obstacles = [
        (np.array([3.0, 3.0]), 1.0),
        (np.array([5.0, 5.0]), 1.5),
        (np.array([7.0, 3.0]), 1.0),
        (np.array([3.0, 7.0]), 0.8),
        (np.array([7.0, 7.0]), 1.2),
    ]
    
    collision_checker = SimpleCollisionChecker(
        obstacles=obstacles,
        bounds=(bounds_min, bounds_max),
        robot_radius=0.2
    )
    
    return collision_checker, bounds_min, bounds_max, obstacles


def main():
    """主測試函式"""
    print("=" * 60)
    print("AIT* 路徑規劃演算法測試")
    print("=" * 60)
    
    # 建立測試環境
    collision_checker, bounds_min, bounds_max, obstacles = create_test_environment()
    
    # 建立規劃器
    planner = AITStar(
        collision_checker=collision_checker,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        batch_size=50,
        rewire_factor=1.1,
        goal_bias=0.05,
        max_iterations=500,
        seed=42
    )
    
    # 定義起點和終點
    start = np.array([1.0, 1.0])
    goal = np.array([9.0, 9.0])
    
    print(f"\n起點: {start}")
    print(f"終點: {goal}")
    print(f"障礙物數量: {len(obstacles)}")
    
    # 執行規劃
    import time
    start_time = time.time()
    path, cost = planner.plan(start, goal)
    planning_time = time.time() - start_time
    
    # 輸出結果
    print(f"\n規劃時間: {planning_time:.3f} 秒")
    stats = planner.get_statistics()
    print(f"迭代次數: {stats['iterations']}")
    print(f"頂點數量: {stats['total_vertices']}")
    print(f"採樣頂點: {stats['vertices_sampled']}")
    print(f"邊檢查數: {stats['edges_checked']}")
    print(f"碰撞檢查: {stats['collision_checks']}")
    print(f"找到解數: {stats['solutions_found']}")
    
    if path is not None:
        print(f"\n找到路徑!")
        print(f"路徑代價: {cost:.3f}")
        print(f"路徑點數: {len(path)}")
        
        # 路徑平滑
        smoother = PathSmoother(collision_checker)
        smoothed_path = smoother.shortcut(path, max_iterations=50)
        smoothed_cost = sum(
            np.linalg.norm(smoothed_path[i+1] - smoothed_path[i]) 
            for i in range(len(smoothed_path) - 1)
        )
        
        print(f"\n平滑後路徑:")
        print(f"路徑代價: {smoothed_cost:.3f}")
        print(f"路徑點數: {len(smoothed_path)}")
    else:
        print("\n未找到有效路徑!")
    
    print("\n" + "=" * 60)
    return path, cost


if __name__ == "__main__":
    main()