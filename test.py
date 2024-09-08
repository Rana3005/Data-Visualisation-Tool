import matplotlib.pyplot as plt
import numpy as np
import math

# Parameters
initial_water_level = 1.0
decay_rate = 0.1
num_time_steps = 100

# Time steps
time_steps = np.arange(1, num_time_steps + 1)

# Calculate water levels for different decay functions
linear_water_levels = initial_water_level - decay_rate * time_steps
exponential_water_levels = initial_water_level * np.exp(-decay_rate * time_steps)
sinusoidal_water_levels = initial_water_level * (1 + np.sin(decay_rate * time_steps))
polynomial_water_levels = initial_water_level - decay_rate * np.power(time_steps, 2)  # Quadratic decay
logarithmic_water_levels = initial_water_level - decay_rate * np.log(time_steps)

# Plot the results
plt.plot(time_steps, linear_water_levels, label="Linear")
plt.plot(time_steps, exponential_water_levels, label="Exponential")
plt.plot(time_steps, sinusoidal_water_levels, label="Sinusoidal")
#plt.plot(time_steps, polynomial_water_levels, label="Polynomial")
plt.plot(time_steps, logarithmic_water_levels, label="Logarithmic")

plt.xlabel("Time Step")
plt.ylabel("Water Level")
plt.title("Great Deluge Water Level Decay Functions")
plt.legend()
plt.grid(True)
plt.show()