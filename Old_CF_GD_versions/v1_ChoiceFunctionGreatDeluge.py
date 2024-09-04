import random
import math
import time
import numpy as np

class ProblemDomain:
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.solution = list(range(self.num_cities))
        random.shuffle(self.solution)
        self.best_solution = self.solution.copy()
        self.best_distance = self.calculateTotalDistance(self.solution)
        self.heuristics = [self.swapHeursitic, self.two_OptHeursitic]               #Heursitics List

    def calculateTotalDistance(self, solution):
        total_distance = 0
        for i in range(len(solution)):
            total_distance += self.distance_matrix[solution[i-1]][solution[i]]
        return total_distance
    
    def initialiseSolution(self):
        random.shuffle(self.solution)

    def getFunctionValue(self):
        return self.calculateTotalDistance(self.solution)
    
    def applyHeuristic(self, heuristic, current_index, new_index):
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

class ChoiceFunctionGreatDeluge:
    def __init__(self, problem_domain, max_iterations, initial_water_level=None, decay_rate=0.95):
        self.problem = problem_domain
        self.start_time = time.time()
        self.time_limit = 30 

        #Great Deluge Parameters
        self.max_iterations = max_iterations
        self.decay_rate = decay_rate
        self.initial_water_level = initial_water_level
        self.current_water_level = initial_water_level if initial_water_level else self.problem.getFunctionValue()
    
    def hasTimeExpired(self):
        return time.time() - self.start_time > self.time_limit
    
    def setTimeLimit(self, timeLimit):
        self.time_limit = timeLimit

    def roundTwoDecimals(self, num):
        return round(num, 2)

    def solve(self):
        self.problem.initialiseSolution()

        #Choice Function Parameters
        phi = 0.5
        delta = 0.5
        heuristic_to_apply = 0.0
        init_flag = 0.0
        number_of_heuristics = self.problem.getNumberOfHeuristics()
        best_heuristic_score = 0.0
        fitness_change = 0.0
        prev_fitness_change = 0.0
        F = [0.0] * number_of_heuristics
        f1 = [0.0] * number_of_heuristics
        f3 = [0.0] * number_of_heuristics
        f2 = np.zeros((number_of_heuristics, number_of_heuristics))
        last_heuristic_called = 0
        crossover_heuristics = self.problem.getHeursiticsOfType('CROSSOVER')
        
        new_solution_value = 0  #new_obj_function_value
        current_solution_value = self.problem.getFunctionValue()    #current_obj_function_value
        best_solution = self.problem.getBestSolution()
        best_solution_value = self.problem.getBestDistance()

        for i in crossover_heuristics:
            f3[i] = float('-inf')

        while not self.hasTimeExpired():
            if init_flag > 1:
                best_heuristic_score = 0.0
                for i in range(number_of_heuristics):
                    F[i] = phi * f1[i] + phi * f2[i][last_heuristic_called] + delta * f3[i]
                    if F[i] > best_heuristic_score:
                        heuristic_to_apply = i
                        best_heuristic_score = F[i]
            else:
                #check flag missing to make sure crossover is not selected, heuristic indexes
                heuristic_to_apply = random.randint(0, number_of_heuristics - 1)

            time_exp_before = time.time()
            ## Needs heuristic index in Problem Class
            #self.problem.applyHeuristic(self.problem.heuristics[heuristic_to_apply])
            if heuristic_to_apply == 0:
                new_solution_value = self.problem.applyHeuristic(self.problem.swapHeursitic, 0, 0)
            else:
                new_solution_value = self.problem.applyHeuristic(self.problem.two_OptHeursitic, 0, 0)
            time_exp_after = time.time()
            time_to_apply = time_exp_after - time_exp_before + 1

            fitness_change = current_solution_value - new_solution_value

            #Great Deluge acceptance criterion
            if new_solution_value < self.current_water_level:
                current_solution_value = new_solution_value
                #self.problem.solution = new_solution
                if new_solution_value < best_solution_value:
                    best_solution_value = new_solution_value
            else:
                fitness_change = 0

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
            
            self.current_water_level *= self.decay_rate
            if self.current_water_level > best_solution_value:
                self.current_water_level = best_solution_value
            
        self.problem.solution = best_solution
        self.problem.best_solution = best_solution
        self.problem.best_distance = best_solution_value

        """
        for iteration in range(self.max_iterations):
            # Choose a heuristic to apply
            heuristic_to_apply = random.randint(0, number_of_heuristics - 1)
            new_solution, new_solution_value = self.problem.applyHeuristic(self.problem.heuristics[heuristic_to_apply])

            # Great Deluge acceptance criterion
            if new_solution_value < self.current_water_level:
                # Accept the new solution
                current_solution_value = new_solution_value
                self.problem.solution = new_solution
                if new_solution_value < best_solution_value:
                    best_solution = new_solution
                    best_solution_value = new_solution_value

            # Update the water level
            self.current_water_level *= self.decay_rate
            
            # To prevent water level from rising too high, keep it close to the best solution found
            if self.current_water_level > best_solution_value:
                self.current_water_level = best_solution_value
            
        self.problem.solution = best_solution
        self.problem.best_solution = best_solution
        self.problem.best_distance = best_solution_value
        """
    
    def getBestSolution(self):
        return self.problem.getBestSolution()
    
    def getBestDistance(self):
        return self.problem.getBestDistance()

# Example usage:
distance_matrix = [
        [0, 2, 4, 6, 8],
        [0, 0, 3, 5, 7],
        [0, 6, 0, 4, 6],
        [0, 5, 4, 0, 7],
        [0, 5, 6, 7, 0],
    ]

problem_domain = ProblemDomain(distance_matrix)
gd_solver = ChoiceFunctionGreatDeluge(problem_domain, max_iterations=1000, initial_water_level=100, decay_rate=0.995)
gd_solver.setTimeLimit(5)
gd_solver.solve()
print("Best solution:", gd_solver.getBestSolution())
print("Best distance:", gd_solver.getBestDistance())
