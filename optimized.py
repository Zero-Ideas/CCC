import heapq

def generateGrid(R, C, M):
    grid = []
    for i in range(R):
        row = []
        for j in range(C):
            value = ((i * C + j) % M) + 1
            row.append(value)
        grid.append(row)
    return grid

def CheapestConnectingPath(R, C, M):
    grid = generateGrid(R, C, M)
    priorityQueue = []
    cost = [[float('inf')] * C for _ in range(R)]
    
    for col in range(C):
        heapq.heappush(priorityQueue, (grid[0][col], 0, col))
        cost[0][col] = grid[0][col]

    directions = [(1, -1), (1, 0), (1, 1)]

    while priorityQueue:
        currentCost, r, c = heapq.heappop(priorityQueue)

        if r == R - 1:
            return currentCost


        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C:

                new_cost = currentCost + grid[nr][nc]
                

                if new_cost < cost[nr][nc]:
                    cost[nr][nc] = new_cost
                    heapq.heappush(priorityQueue, (new_cost, nr, nc))

    return -1  # fucked up somewhere if we ever reach here


#R = int(input())
#C = int(input())
#M = int(input())
R=20000
C = 19244
M = 5213
print(CheapestConnectingPath(R, C, M))
