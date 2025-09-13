
D = int(input())
E = int(input()) 
for _ in range(E):
    event_type = input().strip()
    Q = int(input())
    if event_type == "+":
        D += Q
    elif event_type == "-":
        D -= Q
print(D)