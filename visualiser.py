import sys
import numpy as np

from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PySide6.QtCore import QTimer, Qt

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation

from tsplib_distance_matrix import load_tsplib_distance_matrix
from random_cities import generate_random_cities
from v4_ChoiceFunctionGreatDeluge import *


class TSPVisualizer(QMainWindow):
    def __init__(self, solution, steps_solution, cities_coord):
        super().__init__()
        self.solution = solution
        #self.solution.append(0)
        self.solution_steps = steps_solution
        self.citiesCoord = cities_coord

        self.setupUI()

    def setupUI(self):
        # Set the main window
        self.setWindowTitle("TSP Visualizer")
        self.setGeometry(100, 100, 1000, 700)

        # Widget set as central widget
        widget = QWidget()
        self.setCentralWidget(widget)

        # Layout to hold plot
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Matplotlib figure
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)

        # Draw cities
        self.scatter = self.ax.scatter(self.citiesCoord[:, 0], self.citiesCoord[:, 1], color='blue')
        self.ax.autoscale(enable=True, axis='both')
        
        # Draw line
        self.line, = self.ax.plot([], [], 'r-', lw=2)
        self.ani = animation.FuncAnimation(self.figure, self.update, frames=len(self.solution), interval=3, repeat=False)
        #self.ani = animation.FuncAnimation(self.figure, self.update, frames=len(self.solution_steps), interval=50, repeat=False)

    def update(self, i):
        if i == 0:
            self.line.set_data([], [])
        else:
            x, y = self.citiesCoord[self.solution[:i + 1], 0], self.citiesCoord[self.solution[:i + 1], 1]
            #x = [self.citiesCoord[j, 0] for j in self.solution_steps[i] + [self.solution_steps[i][0]]]
            #y = [self.citiesCoord[j, 1] for j in self.solution_steps[i] + [self.solution_steps[i][0]]]
            self.line.set_data(x, y)
        return self.line,


if __name__ == '__main__':
    # QApplications class, manages main event loop, window system integration, settings // [] to pass command-line arguments
    app = QApplication([])

    distance_matrix, coordinates = generate_random_cities(15, 500, 500)
    problem = ProblemDomain(distance_matrix)
    hyperH = ChoiceFunctionGreatDeluge(decayRate=0.1)
    hyperH.setTimeLimit(5)
    hyperH.solve(problem)
    solution = problem.getBestSolution()
    steps_solution = hyperH.all_step_solution
    print(solution)

    visualizer = TSPVisualizer(solution, steps_solution, coordinates)
    visualizer.show()
    # app.exec() - event loop, wait for user action // sys.exit() called when terminate program
    sys.exit(app.exec())