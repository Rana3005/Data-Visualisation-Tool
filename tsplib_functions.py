import tsplib95
import numpy as np
from sklearn.manifold import MDS
import os

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

def save_tsp_instance(tsplibfileName, coordinates):
    # Create TSPLIB problem
    #tsp_problem = tsplib95.models.StandardProblem()
    fileName = os.path.basename(tsplibfileName)

    # File name
    name = fileName
    # Problem dimensions - number of cities
    dimension = coordinates.shape[0]

    tsp_data = create_tsplib_content(fileName, coordinates)
    
    # save tsp problem
    with open(tsplibfileName, 'w') as tsp_file:
        tsp_file.write(tsp_data)

def create_tsplib_content(name, coordinates):
        # Creates the TSPLIB file content as a string
        num_cities = coordinates.shape[0]

        # Header section
        content = f"NAME: {name}\n"
        content += "TYPE: TSP\n"
        content += f"DIMENSION: {num_cities}\n"
        content += "EDGE_WEIGHT_TYPE: EUC_2D\n"
        content += "NODE_COORD_SECTION\n"

        # Add the coordinates
        for i, (x, y) in enumerate(coordinates, start=1):
            content += f"{i} {x} {y}\n"

        # End of file marker
        content += "EOF\n"

        return content


if __name__ == "__main__":
    d, c = load_tsplib_distance_matrix("tsplib_data/test.tsp")
    test = np.array([
            [10, 15],
            [20, 25],
            [30, 35],
            [40, 45]
        ])
    
    #save_tsp_instance("C:/Users/chira/Documents/University Masters/Final Project/Data Visualisation Tool/tsplib_data/test.tsp", test)
