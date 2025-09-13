import heapq

def generate_grid(R, C, M):
    """Generates the grid following the cyclic row-wise pattern."""
    grid = []
    for i in range(R):
        row = []
        for j in range(C):
            value = ((i * C + j) % M) + 1  # Generate correct pattern
            row.append(value)
        grid.append(row)
    return grid

def CheapestConnectingPath(R, C, M):
    grid = generate_grid(R, C, M)
    priorityQueue = []
    cost = [[float('inf')] * C for _ in range(R)]
    parent = [[None] * C for _ in range(R)]  # To store the parent of each cell
    
    for col in range(C):
        heapq.heappush(priorityQueue, (grid[0][col], 0, col))
        cost[0][col] = grid[0][col]

    directions = [(1, -1), (1, 0), (1, 1)]

    # Dijkstra's Algorithm to find the minimum cost path
    while priorityQueue:
        currentCost, r, c = heapq.heappop(priorityQueue)

        if r == R - 1:  # If we reach the last row, we've found the minimum cost path
            break

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C:
                new_cost = currentCost + grid[nr][nc]

                if new_cost < cost[nr][nc]:
                    cost[nr][nc] = new_cost
                    parent[nr][nc] = (r, c)  # Store the parent to backtrack later
                    heapq.heappush(priorityQueue, (new_cost, nr, nc))

    # Backtrack from the last row to the first row to find the path
    # We need to find the minimum cost in the last row and backtrack from there
    min_cost = float('inf')
    end_col = -1
    for col in range(C):
        if cost[R-1][col] < min_cost:
            min_cost = cost[R-1][col]
            end_col = col

    # Backtrack and mark the path with 'X'
    r, c = R - 1, end_col
    while r is not None:
        grid[r][c] = 'X'
        r, c = parent[r][c] if r >= 0 else (None, None)  # Get the parent of the current tile

    # Return the minimum cost and print the grid with the path
    for row in grid:
        print(row)
    
    return min_cost

# Example Usage:
R = 30  # Number of rows
C = 52  # Number of columns
M = 7   # Max tile cost

print("Minimum Path Cost:", CheapestConnectingPath(R, C, M))  # Expected to print the cost and the grid with 'X' path
