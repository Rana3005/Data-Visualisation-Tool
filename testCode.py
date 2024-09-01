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

def two_OptHeursitic(solution):   #1
    i, j = sorted(random.sample(range(len(solution)), 2))
    #,j = 0,4
    print("i+j: ", [i,j])
    new_solution = solution[:i+1] + solution[i+1:j+1][::-1] + solution[j+1:]

    #new_solution[i+1:j+1] = reversed(new_solution[i+1:j+1])
    return new_solution

print(two_OptHeursitic([0,1,4,3,2,5,6,7,8]))

unvisited = set([1,2,3])
city = 1
while unvisited:
    unvisited.remove(city)
    print(unvisited)
    city += 1


