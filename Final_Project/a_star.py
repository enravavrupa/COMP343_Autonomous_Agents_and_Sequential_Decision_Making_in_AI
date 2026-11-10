"""
pathfinding.py -- adapts ForkliftPlanner (planner.py) to the frontend's
grid format, so frontend.py can call a plain find_path(start, goal, grid).

This file is integration glue only -- the actual A* logic lives in
planner.py, which you wrote and fixed yourself.
"""

from __future__ import annotations
from typing import List, Optional, Tuple

from planner import ForkliftPlanner

Cell = Tuple[int, int]


class GridEnvironment:
    """Minimal adapter exposing the same interface ForkliftPlanner expects
    (is_valid, get_neighbors, manhattan_distance, euclidean_distance),
    backed directly by the frontend's own grid -- no config.py, no
    SurgicalLabEnvironment obstacle generation involved."""

    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0]) if grid else 0

    def is_valid(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height and self.grid[y][x] == 0

    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        return [(x + dx, y + dy) for dx, dy in directions if self.is_valid(x + dx, y + dy)]

    def manhattan_distance(self, x1, y1, x2, y2) -> float:
        return abs(x1 - x2) + abs(y1 - y2)

    def euclidean_distance(self, x1, y1, x2, y2) -> float:
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def find_path(start: Cell, goal: Cell, grid: List[List[int]]) -> Optional[List[Cell]]:
    """Matches the interface frontend.py expects: cells as (col, row),
    grid[row][col] with 0 = open, 1 = wall. Returns None if unreachable."""
    env = GridEnvironment(grid)
    planner = ForkliftPlanner(env)
    path, nodes_expanded, elapsed = planner.a_star(start, goal, heuristic_type='manhattan')
    return path if path else None
