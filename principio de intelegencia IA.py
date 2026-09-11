import heapq
import time
from collections import deque


GOAL = (
    1, 2, 3,
    4, 5, 6,
    7, 8, 0,
)


def get_neighbors(state):
    """Generate all valid states from the current state."""
    neighbors = []

    zero = state.index(0)
    row = zero // 3
    col = zero % 3

    moves = [
        (-1, 0),  # Up
        (1, 0),   # Down
        (0, -1),  # Left
        (0, 1),   # Right
    ]

    for dr, dc in moves:
        new_row = row + dr
        new_col = col + dc

        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_zero = new_row * 3 + new_col

            new_state = list(state)
            new_state[zero], new_state[new_zero] = (
                new_state[new_zero],
                new_state[zero]
            )

            neighbors.append(tuple(new_state))

    return neighbors


def manhattan_distance(state):
    """Calculate the Manhattan distance heuristic."""
    distance = 0

    for index, value in enumerate(state):
        if value != 0:
            current_row = index // 3
            current_col = index % 3

            goal_index = GOAL.index(value)
            goal_row = goal_index // 3
            goal_col = goal_index % 3

            distance += abs(current_row - goal_row)
            distance += abs(current_col - goal_col)

    return distance


def reconstruct_path(parent, state):
    """Reconstruct the solution path."""
    path = []

    while state is not None:
        path.append(state)
        state = parent[state]

    path.reverse()
    return path


def bfs(start):
    """Breadth-First Search."""
    queue = deque([start])
    parent = {start: None}
    expanded = 0

    while queue:
        state = queue.popleft()
        expanded += 1

        if state == GOAL:
            return reconstruct_path(parent, state), expanded

        for neighbor in get_neighbors(state):
            if neighbor not in parent:
                parent[neighbor] = state
                queue.append(neighbor)

    return None, expanded


def UCS(start):
    """Uniform-Cost Search."""
    priority_queue = [(0, start)]
    parent = {start: None}
    cost = {start: 0}
    expanded = 0

    while priority_queue:
        current_cost, state = heapq.heappop(priority_queue)

        if current_cost != cost[state]:
            continue

        expanded += 1

        if state == GOAL:
            return reconstruct_path(parent, state), expanded

        for neighbor in get_neighbors(state):
            new_cost = current_cost + 1
            if neighbor not in cost or new_cost < cost[neighbor]:
                cost[neighbor] = new_cost
                parent[neighbor] = state
                heapq.heappush(priority_queue, (new_cost, neighbor))

    return None, expanded


def astar(start):
    """A* Search Using Manhattan distance."""
    priority_queue = [(manhattan_distance(start), 0, start)]
    parent = {start: None}
    cost = {start: 0}
    expanded = 0

    while priority_queue:
        _, current_cost, state = heapq.heappop(priority_queue)

        if current_cost != cost[state]:
            continue

        expanded += 1

        if state == GOAL:
            return reconstruct_path(parent, state), expanded

        for neighbor in get_neighbors(state):
            new_cost = current_cost + 1

            if neighbor not in cost or new_cost < cost[neighbor]:
                cost[neighbor] = new_cost
                parent[neighbor] = state

                priority = new_cost + manhattan_distance(neighbor)
                heapq.heappush(priority_queue, (priority, new_cost, neighbor))

    return None, expanded


def run_algorithm(name, algorithm, start):
    """Run an algorithm and measure execution time."""
    start_time = time.perf_counter()

    path, expanded = algorithm(start)

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    if path:
        solution_length = len(path) - 1
    else:
        solution_length = -1

    return {
        "algorithm": name,
        "solution_length": solution_length,
        "expanded": expanded,
        "time": elapsed,
    }


instance = [
    (
        "Instance 1",
        (1, 2, 3,
         4, 5, 6,
         7, 0, 8),
    ),
    (
        "Instance 2",
        (1, 2, 3,
         4, 5, 6,
         0, 7, 8),
    ),
    (
        "instance 3",
        (1, 2, 3,
         5, 0, 6,
         4, 7, 8),
    ),
]

algorithms = [
    ("BFS", bfs),
    ("UCS", UCS),
    ("A*", astar),
]

for instance_name, initial_state in instance:
    print(f"\n{instance_name}")
    print("-" * 50)

    for name, algorithm in algorithms:
        result = run_algorithm(name, algorithm, initial_state)

        print(
            f"{result['algorithm']:5} | "
            f"Solution: {result['solution_length']:2} | "
            f"Expanded: {result['expanded']:5}| "
            f"Time:{result['time']:.6f} s"
        )
            
                
