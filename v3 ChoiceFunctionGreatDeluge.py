import random
import time
import numpy as np
from tsplib_distance_matrix import load_tsplib_distance_matrix

class ProblemDomain:
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.solution = []
        self.best_solution = []
        self.best_solution_value = float('inf')

    def calculateTotalDistance(self, solution):
        total_distance = 0
        for i in range(len(solution)):
            total_distance += self.distance_matrix[solution[i-1]][solution[i]]
        return total_distance
    
    def initialiseSolution(self, index):
        self.solution = list(range(self.num_cities))
        random.shuffle(self.solution)
        self.best_solution = self.solution[:]
        self.best_solution_value = self.getFunctionValue()

    def getFunctionValue(self):
        return self.calculateTotalDistance(self.solution)
    
    def applyHeuristic(self, heuristic_index, current_index, new_index):
        solution_copy = self.solution.copy()
        if heuristic_index == 0:
            new_solution = self.swapHeursitic(solution_copy)
        elif heuristic_index == 1:
            new_solution = self.two_OptHeursitic(solution_copy)
        elif heuristic_index == 2:
            new_solution = self.reverse_subtour_operator(solution_copy)
        elif heuristic_index == 3:
            new_solution = self.scramble_subtour_operator(solution_copy)
        #need multi solution tsp
    
        return self.calculateTotalDistance(new_solution)
    
    def getNumberOfHeuristics(self):
        return len(self.getHeursiticsOfType("LOCAL_SEARCH")) + \
            len(self.getHeursiticsOfType("CROSSOVER")) + \
            len(self.getHeursiticsOfType("MUTATION"))
    
    def getBestSolution(self):
        return self.best_solution
    
    def getBestSolutionValue(self):
        return self.best_solution_value
    
    def getHeursiticsOfType(self, heuristicType):
        if heuristicType == "LOCAL_SEARCH":
            return [0, 1, 2, 3]
        elif heuristicType == "CROSSOVER":
            return []
        elif heuristicType == "MUTATION":
            return []
        return []
    
    ### Local Search Heuristics ###
    def swapHeursitic(self, solution):  #0
        i, j = random.sample(range(len(solution)), 2)
        solution[i], solution[j] = solution[j], solution[i]
        return solution
    
    def two_OptHeursitic(self, solution):   #1
        i, j = sorted(random.sample(range(len(solution)), 2))
        solution[i:j+1] = reversed(solution[i:j+1])
        return solution
    
    def reverse_subtour_operator(self, solution):   #2
        i, j = sorted(random.sample(range(len(solution)), 2))
        solution[i:j] = reversed(solution[i:j])
        return solution

    def scramble_subtour_operator(self, solution):  #3
        i, j = sorted(random.sample(range(len(solution)), 2))
        sublist = solution[i: j]
        random.shuffle(sublist)
        solution[i:j] = sublist
        return solution
    

class ChoiceFunctionGreatDeluge:
    def __init__(self, r_seed=None, maxIteration=None, decayRate=None, inital_water_lvl=None):
        random.seed(r_seed)
        self.start_time = time.time()
        self.time_limit = 30

        #Great Deluge Parameters
        self.max_iterations = maxIteration
        self.decay_rate = decayRate
        self.initial_water_level = inital_water_lvl

    def hasTimeExpired(self):
        return time.time() - self.start_time > self.time_limit
    
    def setTimeLimit(self, timeLimit):
        self.time_limit = timeLimit
    
    def roundTwoDecimals(self, num):
        return round(num, 2)

    def solve(self, problem: ProblemDomain):
        problem.initialiseSolution(0)

        #Choice Function Parameters
        phi = 0.5
        delta = 0.5
        heuristic_to_apply = 0.0
        init_flag = 0.0
        new_obj_function_value = 0.0    #new_solution_value
        number_of_heuristics = problem.getNumberOfHeuristics()
        current_obj_function_value = problem.getFunctionValue()    #current_solution_value
        best_heuristic_score = 0.0
        fitness_change = 0.0
        prev_fitness_change = 0.0
        F = [0.0] * number_of_heuristics
        f1 = [0.0] * number_of_heuristics
        f3 = [0.0] * number_of_heuristics
        f2 = np.zeros((number_of_heuristics, number_of_heuristics))
        last_heuristic_called = 0
        crossover_heuristics = problem.getHeursiticsOfType('CROSSOVER')

        for i in crossover_heuristics:
            f3[i] = float('-inf')

        #Great Deluge Parameters
        initial_water_level = self.initial_water_level
        water_level = initial_water_level if initial_water_level else current_obj_function_value
        decay_rate = self.decay_rate

        while not self.hasTimeExpired():
            if init_flag > 1:
                best_heuristic_score = 0.0
                for i in range(number_of_heuristics):
                    F[i] = phi * f1[i] + phi * f2[i][last_heuristic_called] + delta * f3[i]
                    if F[i] > best_heuristic_score:
                        heuristic_to_apply = i
                        best_heuristic_score = F[i]
            else:
                crossflag = True
                while crossflag:
                    heuristic_to_apply = random.randint(0, number_of_heuristics - 1)
                    crossflag = heuristic_to_apply in crossover_heuristics

            time_exp_before = time.time()
            new_obj_function_value = problem.applyHeuristic(heuristic_to_apply, 0, 0)
            time_exp_after = time.time()
            time_to_apply = time_exp_after - time_exp_before + 1

            fitness_change = current_obj_function_value - new_obj_function_value

            #Great Deluge acceptance criterion
            if new_obj_function_value < water_level:
                current_obj_function_value = new_obj_function_value     #new_solution
                
                if current_obj_function_value < problem.best_solution_value:
                    problem.best_solution = problem.solution[:]
                    problem.best_solution_value = current_obj_function_value
            else:
                fitness_change = 0.0

            #Update water level
            water_level -= decay_rate
            #Prevent water level from rising too high, keep it close to best solution
            if water_level > problem.best_solution_value:
                water_level = problem.best_solution_value


            if init_flag > 1:
                f1[heuristic_to_apply] = fitness_change / time_to_apply + phi * f1[heuristic_to_apply]
                f2[heuristic_to_apply][last_heuristic_called] = prev_fitness_change + fitness_change / time_to_apply + phi * f2[heuristic_to_apply][last_heuristic_called]
            elif init_flag == 1:
                f1[heuristic_to_apply] = fitness_change /time_to_apply
                f2[heuristic_to_apply][last_heuristic_called] = prev_fitness_change + fitness_change / time_to_apply + prev_fitness_change
                init_flag += 1
            else:
                f1[heuristic_to_apply] = fitness_change / time_to_apply
                init_flag += 1

            for i in range(number_of_heuristics):
                f3[i] += time_to_apply
            f3[heuristic_to_apply] = 0.0

            if fitness_change > 0.0:    #Improvement
                phi = 0.99
                delta = 0.01
                prev_fitness_change = fitness_change / time_to_apply
            else:   #Non-improvment
                phi = max(0.01, phi - 0.01)
                phi = self.roundTwoDecimals(phi)
                delta = self.roundTwoDecimals(1 - phi)
                prev_fitness_change = 0.0

            last_heuristic_called = heuristic_to_apply


if __name__ == '__main__':
    distance_matrix = [
        [0, 2, 4, 6, 8],
        [3, 0, 3, 5, 7],
        [4, 7, 0, 4, 6],
        [5, 5, 3, 0, 7],
        [6, 3, 4, 5, 0],
    ]

    tsplib = load_tsplib_distance_matrix("tsplib/a280.tsp")
    #problem = ProblemDomain(distance_matrix)
    problem = ProblemDomain(tsplib)
    
    hyperH = ChoiceFunctionGreatDeluge(decayRate=0.01)
    hyperH.setTimeLimit(20)
    hyperH.solve(problem)

    print(f"Best tour: {problem.best_solution}")
    print(f"Shortest distance: {problem.best_solution_value}")