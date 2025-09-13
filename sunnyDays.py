N = int(input())
days = []
for _ in range(N):
    days.append(input().strip())
streaks = []
maxSunnyDays = 0
currentSunnyDays = 0
for day in days:
    if day == "S":
        currentSunnyDays += 1
    else:
        if currentSunnyDays > 0:
            streaks.append(currentSunnyDays)
            maxSunnyDays = max(maxSunnyDays, currentSunnyDays)
        streaks.append(0)
        currentSunnyDays = 0
if currentSunnyDays > 0:
    streaks.append(currentSunnyDays)
    maxSunnyDays = max(maxSunnyDays, currentSunnyDays)
maxStreak = maxSunnyDays
for i in range(len(streaks) - 2):
    if streaks[i] > 0 and streaks[i + 1] == 0 and streaks[i + 2] > 0:

        maxStreak = max(maxStreak, streaks[i] + streaks[i + 2] + 1)
if 0 in streaks:
    maxStreak = max(maxStreak, maxSunnyDays + 1)

print(maxStreak)