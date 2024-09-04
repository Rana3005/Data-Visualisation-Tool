import tsplib95
import numpy as np

#Return distance matrix from tsplib file
def load_tsplib_distance_matrix(tsplib_file: str) -> np.ndarray:
    tsp_problem_instance = tsplib95.load(tsplib_file)

    # Creates array containing distances between all pairs of cities in tsp problem
    # For each edge, it gets the distance between two cities using get_weight()
    flattened_distance_matrix = np.array(
        [tsp_problem_instance.get_weight(*edge) for edge in tsp_problem_instance.get_edges()]
        )
    
    # Reshapes flattened array into square matrix, each row and column representes city
    # Dimensions of matrix determined by number of cities in TSP problem
    distance_matrix = np.reshape(flattened_distance_matrix, 
                                 (tsp_problem_instance.dimension, tsp_problem_instance.dimension))
    
    # Set diagonal elements of distance matrix to zero. As distance from a city to itself is usually zero
    np.fill_diagonal(distance_matrix, 0)

    coordinates = tsp_problem_instance.node_coords
    #print(coordinates)

    return distance_matrix

#load_tsplib_distance_matrix("tsplib/a280.tsp")