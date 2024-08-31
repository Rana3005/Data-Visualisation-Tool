import numpy as np
import random

a = ['test'] * 3

cross = 'test' in a
print(cross)

crossover_heuristics = [1]
f3 = [0] * 3

for i in crossover_heuristics:  # Prevent crossover from being selected
    f3[i] = float('-inf')
    print(f3[i])
f3[1] += .1
print(f3)

print('no' if None else 'b')

def scramble_subtour_operator(solution):  #3
    i, j = sorted(random.sample(range(len(solution)), 2))
    sublist = solution[i: j]
    print(sublist)
    random.shuffle(sublist)
    solution[i:j] = sublist
    return solution

test = scramble_subtour_operator([1,2,3,4,5,6,7])
#print(test)

