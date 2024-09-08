import sys
import numpy as np

from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PySide6.QtCore import QTimer, Qt

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation

from tsplib_functions import load_tsplib_distance_matrix
from random_cities import generate_random_cities
from v4_ChoiceFunctionGreatDeluge import *


class TSPVisualizer(QMainWindow):
    def __init__(self, solution, steps_solution, cities_coord):
        super().__init__()
        self.solution = solution
        self.solution.append(0)
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

        # Line of path
        self.line, = self.ax.plot([], [], 'r1-', lw=2)

        # Animation
        self.ani = animation.FuncAnimation(self.figure, self.update, frames=range(0, len(self.solution_steps)), interval=300, init_func=self.init, repeat=False)

    def init(self):
        # Draw cities
        x = [self.citiesCoord[i][0] for i in self.solution_steps[0]]
        y = [self.citiesCoord[i][1] for i in self.solution_steps[0]]
        plt.plot(x, y, 'co')
        self.ax.autoscale(enable=True, axis='both')

        # Initialize empty solution
        self.line.set_data([], [])

        return self.line,

    def update(self, frame):
        # Update for every frame the solution on the graph
        x = [self.citiesCoord[i, 0] for i in self.solution_steps[frame] + [self.solution_steps[frame][0]]]
        y = [self.citiesCoord[i, 1] for i in self.solution_steps[frame] + [self.solution_steps[frame][0]]]
        #print(f"Solutions changes: {self.solution_steps[frame]}")

        self.line.set_data(x, y)
        return self.line

if __name__ == '__main__':
    # QApplications class, manages main event loop, window system integration, settings // [] to pass command-line arguments
    app = QApplication([])

    #tsplib = load_tsplib_distance_matrix("tsplib_data/a280.tsp")
    #problem = ProblemDomain(tsplib)

    distance_matrix, coordinates = generate_random_cities(25, 500, 500)
    problem = ProblemDomain(distance_matrix)
    hyperH = ChoiceFunctionGreatDeluge(decayRate=0.1)
    hyperH.setTimeLimit(3)
    hyperH.solve(problem)
    solution = problem.getBestSolution()
    steps_solution = hyperH.all_step_solution
    print(len(steps_solution))

    visualizer = TSPVisualizer(solution, steps_solution, coordinates)
    visualizer.show()
    # app.exec() - event loop, wait for user action // sys.exit() called when terminate program
    sys.exit(app.exec())