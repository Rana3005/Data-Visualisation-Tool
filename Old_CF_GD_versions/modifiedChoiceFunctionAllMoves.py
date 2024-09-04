import random
import time
import numpy as np
from tsplib_distance_matrix import load_tsplib_distance_matrix

class ProblemDomain:
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.solution = list(range(self.num_cities))
        random.shuffle(self.solution)
        self.best_solution = self.solution.copy()
        self.best_distance = self.calculateTotalDistance(self.solution)
        self.heuristics = [self.swapHeursitic, self.two_OptHeursitic]

    def calculateTotalDistance(self, solution):
        total_distance = 0
        for i in range(len(solution)):
            total_distance += self.distance_matrix[solution[i-1]][solution[i]]
        return total_distance
    
    def initialiseSolution(self):
        random.shuffle(self.solution)

    def getFunctionValue(self):
        return self.calculateTotalDistance(self.solution)
    
    def applyHeuristic(self, heuristic, index):
        new_solution = heuristic(self.solution.copy())
        new_distance = self.calculateTotalDistance(new_solution)
        if new_distance < self.best_distance:
            self.best_solution = new_solution
            self.best_distance = new_distance
        self.solution = new_solution
        return new_distance
    
    def getNumberOfHeuristics(self):
        return len(self.heuristics)
    
    def getBestSolution(self):
        return self.best_solution
    
    def getBestDistance(self):
        return self.best_distance
    
    def swapHeursitic(self, solution):
        i, j = random.sample(range(len(solution)), 2)
        solution[i], solution[j] = solution[j], solution[i]
        return solution
    
    def two_OptHeursitic(self, solution):
        i, j = sorted(random.sample(range(len(solution)), 2))
        solution[i:j+1] = reversed(solution[i:j+1])
        return solution
    
    def getHeursiticsOfType(self, heuristicType):
        if heuristicType == "CROSSOVER":
            return []
        elif heuristicType == "LOCAL_SEARCH":
            return [self.swapHeursitic, self.two_OptHeursitic]
        return []

class ModifiedChoiceFunction:
    def __init__(self, seed):
        #random.seed(seed)
        self.start_time = time.time()
        self.time_limit = 30        #time limit 30 sec

    def hasTimeExpired(self):
        return time.time() - self.start_time > self.time_limit
    
    def setTimeLimit(self, timeLimit):
        self.time_limit = timeLimit

    def roundTwoDecimals(self, num):
        return round(num, 2)
    
    def solve(self, problem):
        problem.initialiseSolution()
        phi = 0.5
        delta = 0.5
        init_flag = 0
        new_obj_function_value = 0.0
        number_of_heuristics = problem.getNumberOfHeuristics()
        current_obj_function_value = problem.getFunctionValue()
        best_heuristic_score = 0.0
        fitness_change = 0.0
        prev_fitness_change = 0.0
        F = [0.0] * number_of_heuristics
        f1 = [0.0] * number_of_heuristics
        f3 = [0.0] * number_of_heuristics
        f2 = np.zeros((number_of_heuristics, number_of_heuristics))
        last_heursitic_called = 0

        while not self.hasTimeExpired():
            if init_flag > 1:
                best_heuristic_score = 0.0
                for i in range(number_of_heuristics):
                    F[i] = phi * f1[i] + phi * f2[i][last_heursitic_called] + delta * f3[i]
                    if F[i] > best_heuristic_score:
                        heuristic_to_apply = i
                        best_heuristic_score = F[i]
            
            else:
                heuristic_to_apply = random.randint(0, number_of_heuristics - 1)

            time_exp_before = time.time()
            if heuristic_to_apply == 0:
                new_obj_function_value = problem.applyHeuristic(problem.swapHeursitic, 0)
            else:
                new_obj_function_value = problem.applyHeuristic(problem.two_OptHeursitic, 0)
            time_exp_after = time.time()
            time_to_apply = time_exp_after - time_exp_before + 1

            fitness_change = current_obj_function_value - new_obj_function_value
            current_obj_function_value = new_obj_function_value

            if init_flag > 1:
                f1[heuristic_to_apply] = fitness_change / time_to_apply + phi * f1[heuristic_to_apply]
                f2[heuristic_to_apply][last_heursitic_called] = prev_fitness_change + fitness_change / time_to_apply + phi * f2[heuristic_to_apply][last_heursitic_called]
            elif init_flag == 1:
                f1[heuristic_to_apply] = fitness_change /time_to_apply
                f2[heuristic_to_apply][last_heursitic_called] = prev_fitness_change + fitness_change / time_to_apply + prev_fitness_change
                init_flag += 1
            else:
                f1[heuristic_to_apply] = fitness_change / time_to_apply
                init_flag += 1

            for i in range(number_of_heuristics):
                f3[i] += time_to_apply
            f3[heuristic_to_apply] = 0.0

            if fitness_change > 0.0:
                phi = 0.99
                delta = 0.01
                prev_fitness_change = fitness_change / time_to_apply
            else:
                phi = max(0.01, phi - 0.01)
                phi = self.roundTwoDecimals(phi)
                delta = self.roundTwoDecimals(1 - phi)
                prev_fitness_change = 0.0

            last_heursitic_called = heuristic_to_apply

        
    def getBestSolution(self, problem):
        return problem.getBestSolution()
    
    def getBestDistance(self, problem):
        return problem.getBestDistance()
    

if __name__ == '__main__':
    distance_matrix = [
        [0, 2, 4, 6, 8],
        [0, 0, 3, 5, 7],
        [0, 6, 0, 4, 6],
        [0, 5, 4, 0, 7],
        [0, 5, 6, 7, 0],
    ]

    tsplib = load_tsplib_distance_matrix("tsplib/a280.tsp")
    problem = ProblemDomain(distance_matrix)
    #problem = ProblemDomain(tsplib)
    
    hyperH = ModifiedChoiceFunction(123)
    hyperH.setTimeLimit(20)

    hyperH.solve(problem)

    best_solution = hyperH.getBestSolution(problem)
    best_distance = hyperH.getBestDistance(problem)

    print(f"Best Solution: {best_solution}")
    print(f"Best Distance: {best_distance}")
