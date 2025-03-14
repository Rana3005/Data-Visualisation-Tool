import unittest
import numpy as np
from unittest.mock import patch
from sklearn.metrics import pairwise_distances_argmin_min
import random
from random_cities import generate_random_cities
from tsplib_functions import load_tsplib_distance_matrix, calculate_coordinate_distance_matrix

class TestTSPFunctions(unittest.TestCase):

    def test_generate_random_cities(self):
        # Test basic functionality of generating cities
        num_cities = 10
        region_width = 100
        region_height = 100
        distance_matrix, coordinates = generate_random_cities(num_cities, region_width, region_height, seed=42)
        
        # Assert that the number of cities matches
        self.assertEqual(len(coordinates), num_cities)
        self.assertEqual(distance_matrix.shape, (num_cities, num_cities))
        
        # Assert that all diagonal elements of the distance matrix are zero (distance to itself)
        self.assertTrue(np.all(np.diag(distance_matrix) == 0))
        
        # Assert that the matrix is symmetric
        self.assertTrue(np.allclose(distance_matrix, distance_matrix.T))

    def test_generate_random_cities_min_limit(self):
        # Test if function enforces minimum number of cities
        distance_matrix, coordinates = generate_random_cities(3, 100, 100, seed=42)
        self.assertEqual(len(coordinates), 4)  # Minimum number of cities should be 4

    def test_generate_random_cities_max_limit(self):
        # Test if function enforces maximum number of cities
        distance_matrix, coordinates = generate_random_cities(600, 1000, 1000, seed=42)
        self.assertEqual(len(coordinates), 500)  # Maximum number of cities should be 500

    def test_load_tsplib_distance_matrix(self):
        # Test loading distance matrix from a tsplib file (mocking the tsplib95 call)
        with patch('tsplib95.load') as mock_load:
            mock_problem_instance = mock_load.return_value
            mock_problem_instance.dimension = 5
            mock_problem_instance.get_weight = lambda x, y: np.random.randint(1, 100)
            mock_problem_instance.get_edges = lambda: [(i, j) for i in range(5) for j in range(i + 1, 5)]
            mock_problem_instance.is_depictable = lambda: False  # No coordinates

            distance_matrix, coord_array = load_tsplib_distance_matrix("dummy_file.tsp")

            # Assert that the distance matrix and coordinates have the correct size
            self.assertEqual(distance_matrix.shape, (5, 5))

    def test_calculate_coordinate_distance_matrix(self):
        # Test multidimensional scaling (MDS) to calculate coordinates from the distance matrix
        dummy_distance_matrix = np.array([[0, 2], [2, 0]])
        coordinates = calculate_coordinate_distance_matrix(dummy_distance_matrix)
        
        # Assert that the output coordinates match the expected shape
        self.assertEqual(coordinates.shape, (2, 2))

    def test_calculate_total_distance(self):
        # Test the total distance calculation
        distance_matrix = np.array([[0, 1, 2], 
                                    [1, 0, 1], 
                                    [2, 1, 0]])
        solution = [0, 1, 2]
        tsp_problem = ProblemDomain(distance_matrix=distance_matrix)
        
        total_distance = tsp_problem.calculateTotalDistance(solution)
        
        # Expected total distance: 0->1 (1) + 1->2 (1) + 2->0 (2) = 4
        self.assertEqual(total_distance, 4)

    def test_initialise_solution_random(self):
        # Test if "Random" initialisation works
        tsp_problem = TSPProblem(num_cities=5)
        tsp_problem.initialiseSolution("Random")
        
        # Assert that the solution contains all cities
        self.assertEqual(set(tsp_problem.solution), set(range(5)))
        # Assert that the solution is shuffled
        self.assertNotEqual(tsp_problem.solution, list(range(5)))

    def test_initialise_solution_nearest_neighbour(self):
        # Test if "Nearest Neighbour" heuristic initialization works
        tsp_problem = TSPProblem(num_cities=5)
        tsp_problem.initialiseSolution("Nearest Neighbour")
        
        # Assert that the solution contains all cities
        self.assertEqual(set(tsp_problem.solution), set(range(5)))

    def test_set_phi_delta_cf(self):
        # Test setting phi and delta for the choice function
        tsp_problem = TSPProblem()
        tsp_problem.setPhiDelta_CF(0.5, 0.1)
        
        # Assert that phi and delta are set correctly
        self.assertEqual(tsp_problem.phi, 0.5)
        self.assertEqual(tsp_problem.delta, 0.1)


class TSPProblem:
    """ Mock class for TSP problem used in testing """
    def __init__(self, num_cities=None, distance_matrix=None):
        self.num_cities = num_cities
        self.distance_matrix = distance_matrix
        self.solution = []
        self.best_solution = []
        self.init_solution_value = 0
        self.best_solution_value = 0

    def calculateTotalDistance(self, solution):
        total_distance = 0
        for i in range(len(solution)):
            total_distance += self.distance_matrix[solution[i - 1]][solution[i]]
        return total_distance

    def initialiseSolution(self, initial_type):
        if initial_type == "Random":
            self.solution = list(range(self.num_cities))
            random.shuffle(self.solution)
        elif initial_type == 'Nearest Neighbour':
            self.solution = self.applyHeuristicSolution(6, list(range(self.num_cities)))
        self.init_solution_value = self.getFunctionValue()

    def setPhiDelta_CF(self, phi, delta):
        self.phi = phi
        self.delta = delta

    def getFunctionValue(self):
        return self.calculateTotalDistance(self.solution)

class ProblemDomain:
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix

if __name__ == '__main__':
    unittest.main()
