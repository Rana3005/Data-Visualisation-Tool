import random
import time
import numpy as np
from tsplib_functions import load_tsplib_distance_matrix

class ProblemDomain:
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.solution = []
        self.best_solution = []
        self.best_solution_value = float('inf')
        self.mutation = 0
        self.localSearch = 0
        self.crossover =0
        self.ruinrecreate = 0

    def calculateTotalDistance(self, solution):
        total_distance = 0
        for i in range(len(solution)):
            total_distance += self.distance_matrix[solution[i-1]][solution[i]]
        return total_distance
    
    def getDistance(self, city1, city2):
        return self.distance_matrix[city1][city2]
    
    def initialiseSolution(self, index):
        self.solution = list(range(self.num_cities))
        random.shuffle(self.solution)
        self.best_solution = self.solution[:]
        self.best_solution_value = self.getFunctionValue()

    def getFunctionValue(self):
        return self.calculateTotalDistance(self.solution)
    
    def applyHeuristic(self, heuristic_index, current_index, new_index):
        # Returns the solution value 
        #solution_copy = self.solution.copy()
        if heuristic_index == 0:
            self.mutation += 1
            new_solution = self.swapHeursitic(self.solution)
        elif heuristic_index == 1:
            self.mutation += 1
            new_solution = self.inversionHeursitic(self.solution)
        elif heuristic_index == 2:
            self.mutation += 1
            new_solution = self.scramble_subtourHeuristic(self.solution)
        elif heuristic_index == 3:
            self.mutation += 1
            new_solution = self.insertHeursitic(self.solution)
        elif heuristic_index == 4:
            self.mutation += 1
            new_solution = self.displacementHeuristic(self.solution)
        elif heuristic_index == 5:
            self.localSearch += 1
            new_solution = self.two_OptHeursitic(self.solution)
        elif heuristic_index == 6:
            self.localSearch += 1
            new_solution = self.nearestNeighbor_Heuristic(self.solution)
        elif heuristic_index == 7:
            self.localSearch += 1
            new_solution = self.simulatedAnnealing_Heuristic(self.solution)
        elif heuristic_index == 8:
            self.crossover += 1
            new_solution = self.order_crossover(self.solution)
        elif heuristic_index == 9:
            self.crossover += 1
            new_solution = self.pmx_crossover(self.solution)
        elif heuristic_index == 10:
            self.crossover += 1
            new_solution = self.pbx_crossover(self.solution)
        elif heuristic_index == 11:
            self.crossover += 1
            new_solution = self.oneX_crossover(self.solution)
        elif heuristic_index == 12:
            self.ruinrecreate += 1
            new_solution = self.ruin_recreate_operator(self.solution)
        #need multi solution tsp
    
        return self.calculateTotalDistance(new_solution)
    
    def applyHeuristicSolution(self, heuristic_index, solution):
        #solution_copy = self.solution.copy()
        if heuristic_index == 0:
            new_solution = self.swapHeursitic(solution)
        elif heuristic_index == 1:
            new_solution = self.inversionHeursitic(solution)
        elif heuristic_index == 2:
            new_solution = self.scramble_subtourHeuristic(solution)
        elif heuristic_index == 3:
            new_solution = self.insertHeursitic(solution)
        elif heuristic_index == 4:
            new_solution = self.displacementHeuristic(solution)
        elif heuristic_index == 5:
            new_solution = self.two_OptHeursitic(solution)
        elif heuristic_index == 6:
            new_solution = self.nearestNeighbor_Heuristic(solution)
        elif heuristic_index == 7:
            new_solution = self.simulatedAnnealing_Heuristic(solution)
        elif heuristic_index == 8:
            new_solution = self.order_crossover(solution)
        elif heuristic_index == 9:
            new_solution = self.pmx_crossover(solution)
        elif heuristic_index == 10:
            new_solution = self.pbx_crossover(solution)
        elif heuristic_index == 11:
            new_solution = self.oneX_crossover(solution)
        elif heuristic_index == 12:
            new_solution = self.ruin_recreate_operator(solution)

        return new_solution
    
    def getNumberOfHeuristics(self):
        return len(self.getHeursiticsOfType("LOCAL_SEARCH")) + \
            len(self.getHeursiticsOfType("CROSSOVER")) + \
            len(self.getHeursiticsOfType("MUTATION")) + \
            len(self.getHeursiticsOfType("RUIN_RECREATE"))
    
    def getBestSolution(self):
        return self.best_solution
    
    def getBestSolutionValue(self):
        return self.best_solution_value
    
    def getHeursiticsOfType(self, heuristicType):
        if heuristicType == "MUTATION":
            return [0, 1, 2, 3, 4]
        elif heuristicType == "LOCAL_SEARCH":
            return [5, 6, 7]
        elif heuristicType == "CROSSOVER":
            return [8, 9, 10, 11]
        elif heuristicType == "RUIN_RECREATE":
            return [12]
        return []
    
    def getHeuristics(self):
        return self.getHeursiticsOfType("LOCAL_SEARCH") + \
            self.getHeursiticsOfType("CROSSOVER") + \
            self.getHeursiticsOfType("MUTATION") + \
            self.getHeursiticsOfType("RUIN_RECREATE")

    ### Mutation Operators ###
    def swapHeursitic(self, solution):  #0
        i, j = random.sample(range(len(solution)), 2)
        new_solution = solution.copy()
        
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        return new_solution
    
    def inversionHeursitic(self, solution):  #1
        i, j = sorted(random.sample(range(len(solution)), 2))
        new_solution = solution.copy()
        
        new_solution[i:j] = reversed(new_solution[i:j])
        return new_solution

    def scramble_subtourHeuristic(self, solution):  #2
        i, j = sorted(random.sample(range(len(solution)), 2))
        new_solution = solution.copy()
        
        sublist = new_solution[i: j]
        random.shuffle(sublist)
        new_solution[i:j] = sublist
        return new_solution
    
    def insertHeursitic(self, solution):  #3
        i, j = sorted(random.sample(range(len(solution)), 2))
        new_solution = solution.copy()
        
        city = new_solution.pop(i)
        new_solution.insert(j, city)
        return new_solution
    
    def displacementHeuristic(self, solution):  #4
        i, j = sorted(random.sample(range(len(solution)), 2))
        new_solution = solution.copy()

        sublist = new_solution[i: j+1]
        new_solution = new_solution[:i] + new_solution[j+1:]
        insert_pos = random.randint(0, len(new_solution))

        # Adds randomly selected sublist into insert position
        new_solution[insert_pos:insert_pos] = sublist

        return new_solution
    

    ### Local Search Operators ###
    def two_OptHeursitic(self, solution):   #5
        current_solution = solution.copy()
        current_solution_value = self.calculateTotalDistance(current_solution)
        
        while True:
            improved = False
            i, j = sorted(random.sample(range(len(solution)), 2))
            new_solution = solution[:i+1] + solution[i+1:j+1][::-1] + solution[j+1:]
            new_solution_value = self.calculateTotalDistance(new_solution)

            if new_solution_value < current_solution_value:
                current_solution = new_solution
                current_solution_value = new_solution_value
                improved = True
            
            if not improved:
                break 
            
        return current_solution
        
    def nearestNeighbor_Heuristic(self, solution):  #6
        start = random.choice(solution)
        unvisited = set(solution)
        unvisited.remove(start)
        tour = [start]
        current = start

        while unvisited:
            # key function for min(), compares cities based on distace from current city
            # takes city as input and returns distance between that city and current city
            next_city = min(unvisited, key = lambda city : self.getDistance(current, city))
            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city
        
        return tour
    
    def simulatedAnnealing_Heuristic(self, solution):   #7
        current_solution = solution.copy()
        current_distance = self.calculateTotalDistance(current_solution)
        best_solution = current_solution.copy()
        best_distance = current_distance

        T = 1000.0  # Initial temperature
        T_min = 1.0  # Minimum temperature
        alpha = 0.99  # Cooling rate

        while T > T_min:
            neighbor_solution = self.swapHeursitic(current_solution)
            neighbor_distance = self.calculateTotalDistance(neighbor_solution)
            delta = neighbor_distance - current_distance

            if delta < 0 or random.random() < np.exp(-delta / T):
                current_solution = neighbor_solution
                current_distance = neighbor_distance

                if current_distance < best_distance:
                    best_solution = current_solution.copy()
                    best_distance = current_distance

            T *= alpha  # Decrease temperature

        return best_solution


    ### Crossover Operators ###
    def order_crossover(self, solution, solution2 = None):  #8
        #order crossover
        if solution2 == None:
            heuristicIndex = random.choice([heuristic for heuristic in self.getHeuristics() if heuristic not in self.getHeursiticsOfType("CROSSOVER")])
            solution2 = self.applyHeuristicSolution(heuristicIndex, solution)
        solutionCopy = solution.copy()
        solution2Copy = solution2.copy()

        size = len(solutionCopy)
        a, b = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[a:b] = solutionCopy[a:b]
        ptr = b     #pointer
        for city in solution2Copy[b:] + solution2Copy[:b]:
            if city not in child:
                if ptr >= size:
                    ptr = 0
                child[ptr] = city
                ptr += 1
        return child

    def pmx_crossover(self, solution, solution2 = None):    #9
        #partially mapped crossover
        if solution2 == None:
            heuristicIndex = random.choice([heuristic for heuristic in self.getHeuristics() if heuristic not in self.getHeursiticsOfType("CROSSOVER")])
            solution2 = self.applyHeuristicSolution(heuristicIndex, solution)
        solutionCopy = solution.copy()
        solution2Copy = solution2.copy()

        size = len(solutionCopy)
        a, b = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[a:b] = solutionCopy[a:b]
        for i in range(a, b):
            if solution2Copy[i] not in child:
                pos = i
                while True:
                    city = solutionCopy[pos]
                    pos = solution2Copy.index(city)
                    if child[pos] is None:
                        child[pos] = solution2Copy[i]
                        break
        for i in range(size):
            if child[i] is None:
                child[i] = solution2Copy[i]
        return child

    def pbx_crossover(self, solution, solution2 = None):    #10
        #position-based mapped crossover
        if solution2 == None:
            heuristicIndex = random.choice([heuristic for heuristic in self.getHeuristics() if heuristic not in self.getHeursiticsOfType("CROSSOVER")])
            solution2 = self.applyHeuristicSolution(heuristicIndex, solution)
        solutionCopy = solution.copy()
        solution2Copy = solution2.copy()

        size = len(solutionCopy)
        child = [None] * size
        positions = sorted(random.sample(range(size), size // 2))
        for pos in positions:
            child[pos] = solutionCopy[pos]
        ptr = 0
        for city in solution2Copy:
            if city not in child:
                while ptr < size and child[ptr] is not None:
                    ptr += 1
                if ptr < size:
                    child[ptr] = city
        return child

    def oneX_crossover(self, solution, solution2 = None):   #11
        #one-point mapped crossover
        if solution2 == None:
            heuristicIndex = random.choice([heuristic for heuristic in self.getHeuristics() if heuristic not in self.getHeursiticsOfType("CROSSOVER")])
            solution2 = self.applyHeuristicSolution(heuristicIndex, solution)
        solutionCopy = solution.copy()
        solution2Copy = solution2.copy()

        size = len(solutionCopy)
        point = random.randint(1, size - 1)
        child = solutionCopy[:point]
        for city in solution2Copy:
            if city not in child:
                child.append(city)
        return child


    ### Ruin-Recreate Operators ###
    def ruin_recreate_operator(self, solution):     #12
        #remove a subset and reinsert using nearest insertion
        solution_copy = solution.copy()
        ruin_size = max(2, len(solution_copy) // 10)
        indices = sorted(random.sample(range(len(solution_copy)), ruin_size))
        removed = [solution_copy[i] for i in indices]
        remaining = [city for city in solution_copy if city not in removed]
        # Reinsert removed cities one by one using nearest insertion
        for city in removed:
            if not remaining:
                remaining.append(city)
                continue
            best_position = 0
            best_distance = float('inf')
            for i in range(len(remaining) + 1):
                new_tour = remaining[:i] + [city] + remaining[i:]
                distance = self.calculateTotalDistance(new_tour)
                if distance < best_distance:
                    best_distance = distance
                    best_position = i
            remaining = remaining[:best_position] + [city] + remaining[best_position:]
        return remaining

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
                #print(f"current value: {current_obj_function_value}")
                #print(f"new value: {new_obj_function_value}")
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
            else:   #No-improvment
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

    distance_matrix2 = [
        [0, 251, 146],
        [251, 0, 225],
        [146, 225, 0]
    ]

    distance_matrix3 = [
        [0, 247, 246, 299],
        [247, 0, 61,  158],
        [246, 61,  0,  41],
        [299, 158, 41,  0],
    ]

    tsplib = load_tsplib_distance_matrix("tsplib/br17.atsp")
    #problem = ProblemDomain(distance_matrix2)
    problem = ProblemDomain(tsplib)

    hyperH = ChoiceFunctionGreatDeluge(decayRate=0.1)
    hyperH.setTimeLimit(30)
    hyperH.solve(problem)

    print(f"m: {problem.mutation}, ls: {problem.localSearch}, c: {problem.crossover}, r: {problem.ruinrecreate}")
    print(f"Best tour: {problem.best_solution}")
    print(f"Shortest distance: {problem.best_solution_value}")