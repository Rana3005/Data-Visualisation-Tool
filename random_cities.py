import numpy as np

def generate_random_cities(num_cities, region_width, region_height, seed=None):
    # Generate symmetric distance matrix and related coordinates
    
    if seed is not None:
        np.random.seed(seed)
    
    # Minimum 4 cities
    if num_cities < 4:
        num_cities = 4
    if num_cities > 500:
        num_cities = 500

    # Create numpy array of coordinate based on number of cities
    # // rounds result down to nearest whole number
    x_coordinates = np.random.randint(-region_width // 2, region_width // 2, size=num_cities)
    y_coordinates = np.random.randint(-region_height // 2, region_height // 2, size=num_cities)
    #x_coordinates = np.random.randint(0, region_width, size=num_cities)
    #y_coordinates = np.random.randint(0, region_height, size=num_cities)
    coordinates = np.column_stack((x_coordinates, y_coordinates))

    # Distance matrix
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i+1, num_cities):
            # Pythagoras theorem
            distance = np.sqrt((coordinates[i][0] - coordinates [j][0]) ** 2 +
                               (coordinates[i][1] - coordinates [j][1]) ** 2)
            distance_matrix[i][j] = distance_matrix[j][i] = distance

    return distance_matrix, coordinates