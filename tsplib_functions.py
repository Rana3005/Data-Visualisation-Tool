import tsplib95
import numpy as np
from sklearn.manifold import MDS

#Return distance matrix from tsplib file
def load_tsplib_distance_matrix(tsplib_file: str) -> np.ndarray:
    tsp_problem_instance = tsplib95.load(tsplib_file)
    num_cities = tsp_problem_instance.dimension

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

    # Check if coordinates are available
    #if hasattr(tsp_problem_instance, 'node_coords'):
    if tsp_problem_instance.is_depictable():
        coordinates = tsp_problem_instance.node_coords

        coord_array = np.array([coordinates[i + 1 ] for i in range(num_cities)])
    else:
        # No coordinates available in file
        # Estimate coordinate using MDS
        coord_array = calculate_coordinate_distance_matrix(distance_matrix)

    return distance_matrix, coord_array
    
def calculate_coordinate_distance_matrix(distance_matrix):
    # Estimates 2d coordinates from a distance matrix using mutlidimensional scaling
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coordinates = mds.fit_transform(distance_matrix)
    return coordinates


if __name__ == "__main__":
    d, c = load_tsplib_distance_matrix("tsplib_data/ali535.tsp")
