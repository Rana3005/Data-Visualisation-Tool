import sys
import numpy as np

from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QLineEdit,
                               QPushButton, QCheckBox, QGridLayout, QRadioButton, QComboBox, QGroupBox,
                               QListWidget, QListWidgetItem, QDialog, QDialogButtonBox)
from PySide6.QtCore import QTimer, Qt

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation

from tsplib_functions import load_tsplib_distance_matrix
from random_cities import generate_random_cities
from v5_ChoiceFunctionGreatDeluge import *


class TSPVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUI()

        self.heuristicDialog = HeuristicDialog()
        self.heuristicSelctions = self.heuristicDialog.getSelected()

    def setupUI(self):
        # Set the main window
        self.setWindowTitle("TSP Visualizer")
        self.setGeometry(100, 50, 1000, 700)

        # Widget set as central widget
        widget = QWidget()
        self.setCentralWidget(widget)

        # Vertical Layout to hold widget
        layout = QVBoxLayout()
        widget.setLayout(layout)

        inputLayout = QHBoxLayout()

        # TSP grid box 
        tspSetup_gridLayout = QGridLayout()
        # Time limit input
        self.timeLimitCheckbox = QRadioButton("Time Limit:")
        self.timeLimitInput = QLineEdit()
        self.timeLimitInput.setPlaceholderText("Set time limit (seconds)")
        tspSetup_gridLayout.addWidget(self.timeLimitCheckbox, 0, 0)
        tspSetup_gridLayout.addWidget(self.timeLimitInput, 0, 1)

        # Max iteration input
        self.iterationCheckBox = QRadioButton("Max Iteration:")
        self.iterationLimitInput = QLineEdit()
        self.iterationLimitInput.setPlaceholderText("Set max iteration")
        tspSetup_gridLayout.addWidget(self.iterationCheckBox, 1, 0)
        tspSetup_gridLayout.addWidget(self.iterationLimitInput, 1, 1)

        self.timeLimitCheckbox.toggled.connect(self.time_iteration_stateChange)
        self.iterationCheckBox.toggled.connect(self.time_iteration_stateChange)
        self.timeLimitCheckbox.setChecked(True)

        # Solution initalisation drop-down
        self.solution_initialize_label = QLabel("Solution Initialization:")
        self.solution_initialize_comboBox = QComboBox()
        self.solution_initialize_comboBox.addItem('None')
        self.solution_initialize_comboBox.addItem('Random')
        self.solution_initialize_comboBox.addItem('Nearest Neighbour')
        tspSetup_gridLayout.addWidget(self.solution_initialize_label, 2, 0)
        tspSetup_gridLayout.addWidget(self.solution_initialize_comboBox, 2, 1)

        # Heuristic operator selection
        self.low_lvl_heuristicLabel = QLabel("Low-Level Heurisitics:")
        self.low_lvl_heuristicButton = QPushButton("Select Operators")
        self.low_lvl_heuristicButton.clicked.connect(self.selectHeuristics)
        tspSetup_gridLayout.addWidget(self.low_lvl_heuristicLabel, 3, 0)
        tspSetup_gridLayout.addWidget(self.low_lvl_heuristicButton, 3, 1)
        
        self.crossover_checkBox = QCheckBox("Enable Crossover")
        tspSetup_gridLayout.rowStretch(4)
        tspSetup_gridLayout.addWidget(self.crossover_checkBox, 4, 0)  # Position of widgit is 4th row, 0 column, size takes 1 row and columns
        self.crossover_checkBox.toggled.connect(self.enableCrossover)

        inputLayout.addLayout(tspSetup_gridLayout)

        #Hyper-Heurisitc group box
        hyperheuristic_groupBox = QGroupBox("Hyper Heuristic")
        hyperh_groupBoxLayout = QVBoxLayout()
        hyperheuristic_groupBox.setLayout(hyperh_groupBoxLayout)
        # Great Deluge decay type
        self.gd_label = QLabel("Great Deluge")
        self.gd_waterLevel_label = QLabel("Water Level Decay Type:")
        self.gd_waterLevel_label_comboBox = QComboBox()
        self.gd_waterLevel_label_comboBox.addItem('Linear')
        self.gd_waterLevel_label_comboBox.addItem('Exponential')
        self.gd_waterLevel_label_comboBox.addItem('Sinusoidal')
        self.gd_decayRate_label = QLabel("Decay Rate:")
        self.gd_decayRate_input = QLineEdit()
        self.gd_decayRate_input.setPlaceholderText("Set decay rate (e.g. 0.1)")
        
        gd_gridLayout = QGridLayout()
        gd_gridLayout.addWidget(self.gd_waterLevel_label, 0, 0)
        gd_gridLayout.addWidget(self.gd_waterLevel_label_comboBox, 0, 1)
        gd_gridLayout.addWidget(self.gd_decayRate_label, 1, 0)
        gd_gridLayout.addWidget(self.gd_decayRate_input, 1, 1)
        hyperh_groupBoxLayout.addWidget(self.gd_label)
        hyperh_groupBoxLayout.addLayout(gd_gridLayout)

        inputLayout.addWidget(hyperheuristic_groupBox)

        # Number of cities input
        self.numCitiesInput = QLineEdit()
        self.numCitiesInput.setPlaceholderText("Set number of cities (4-500)")
        inputLayout.addWidget(QLabel("Number of Cities:"))
        inputLayout.addWidget(self.numCitiesInput)

        # Start button
        self.startButton = QPushButton("Start")
        self.startButton.clicked.connect(self.startSolving)
        inputLayout.addWidget(self.startButton)

        # TSP plot canvas
        self.canvas = TSPCanvas(self)
        layout.addWidget(self.canvas)

        layout.addLayout(inputLayout)
        
        
    def selectHeuristics(self):
        if(self.heuristicDialog.exec()):
            self.heuristicSelctions = self.heuristicDialog.getSelected()
            #print(self.heuristicSelctions)
        else:
            #print("cancel")
            self.heuristicDialog.resetSelection()

    def enableCrossover(self):
        if self.crossover_checkBox.checkState() == Qt.CheckState.Checked:
            print("check")
        elif self.crossover_checkBox.checkState() == Qt.CheckState.Unchecked:
            print("noCheck")
    
    # Disable input area based on checked box
    def time_iteration_stateChange(self):
        if self.timeLimitCheckbox.isChecked():
            self.timeLimitInput.setEnabled(True)
            self.iterationLimitInput.setEnabled(False)
        elif self.iterationCheckBox.isChecked():
            self.timeLimitInput.setEnabled(False)
            self.iterationLimitInput.setEnabled(True)

    def startSolving(self):
        self.startButton.setEnabled(False)
        self.canvas.clear_fig()

        num_cities = int(self.numCitiesInput.text())

        distance_matrix, coordinates = generate_random_cities(num_cities, 500, 500)
        problem = ProblemDomain(distance_matrix)
        hyperH = ChoiceFunctionGreatDeluge()
        
        # Check if time limit or iteration checked
        if self.timeLimitCheckbox.isChecked():
            time_limit = float(self.timeLimitInput.text())
            hyperH.setTimeLimit(time_limit)
        elif self.iterationCheckBox.isChecked():
            iteration_limit = int(self.iterationLimitInput.text())
            hyperH.setMaxIteration(iteration_limit)

        # Set the initialization
        current_initialization = self.solution_initialize_comboBox.currentText()
        hyperH.setInitialSolution(current_initialization)

        # Great Deluge decay parameters
        gd_decay_type = self.gd_waterLevel_label_comboBox.currentText()
        hyperH.setDeacyModel = gd_decay_type
        gd_decay_rate = float(self.gd_decayRate_input.text())
        hyperH.setDecayRate(gd_decay_rate)

        hyperH.solve(problem)

        solution = problem.getBestSolution()
        solution_steps = hyperH.all_solution_step
        #print(len(solution_steps))

        self.canvas.start_animation(solution_steps, coordinates)
        self.startButton.setEnabled(True)


class HeuristicDialog(QDialog):
    # Dialog box to select low-level heuristics
    def __init__(self, parent=None):
        super().__init__(parent)
        self.crossover = ProblemDomain(np.zeros((2,2))).getHeursiticsOfType("CROSSOVER")
        
        self.low_lvl_heuristicList = QListWidget()
        self.add_checkableItems([
                                ("Swap Mutation", 0),
                                ("Inversion Mutation", 1), 
                                ("Scramble Mutation", 2), 
                                ("Insert Mutation", 3),
                                ("Displacement Mutation", 4), 
                                ("Two Opt", 5), 
                                ("Nearest Neighhour", 6),
                                ("Simulated Annealing", 7),
                                ("Ruin-Recreate", 12),
                                ("Order Crossover", 8),
                                ("Partially Mapped Crossover", 9),
                                ("Position-based Crossover", 10),
                                ("One-Point Crossover", 11)])

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok
                                      | QDialogButtonBox.Cancel)
        
        layout = QVBoxLayout()
        layout.addWidget(self.low_lvl_heuristicList)
        layout.addWidget(self.button_box)

        self.setLayout(layout)

        self.setWindowTitle("Low-Level heuristic Selection")

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

    # Creates checkable items
    def add_checkableItems(self, items):
        for heuristic_name, heuristic_value in items:
            list_item = QListWidgetItem(heuristic_name)
            list_item.setFlags(list_item.flags() | Qt.ItemFlag.ItemIsUserCheckable) # Enable checkbox, bitwise OR operator with current flags and checkable flag
            list_item.setCheckState(Qt.CheckState.Checked)
            list_item.setData(Qt.ItemDataRole.UserRole, heuristic_value)    # Stores value to item
            self.low_lvl_heuristicList.addItem(list_item)

    # Return selected items
    def getSelected(self):
        currently_selected = []
        for i in range(self.low_lvl_heuristicList.count()):
            item = self.low_lvl_heuristicList.item(i)
            if item.checkState() == Qt.CheckState.Checked and item.flags() & Qt.ItemFlag.ItemIsEnabled:     # Check if item is checked and enabled, bitwise AND operator item's flag and enabled flag
                currently_selected.append(item.data(Qt.UserRole))
        
        return currently_selected
    
    def resetSelection(self):
        for i in range(self.low_lvl_heuristicList.count()):
            item = self.low_lvl_heuristicList.item(i)
            item.setCheckState(Qt.CheckState.Checked)

    def toggleCrossoverSelection(self, isEnabled):
        for i in range(self.low_lvl_heuristicList.count()):
            item = self.low_lvl_heuristicList.item(i)

            
                  


# Inherits from FigureCanvas
class TSPCanvas(FigureCanvasQTAgg):
    # Parent argument is qt widget where canvas is displayed
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots()
        # Pass figure to super class figureCanvas
        super().__init__(self.fig)
        
        # Set parent widget for hierarchy
        self.setParent(parent)
        self.solution_steps = []
        self.citiesCoord = []
        self.line, = self.ax.plot([], [], 'ro-', lw=2)

    def init_plot(self):
        # Draw cities
        x = [self.citiesCoord[i][0] for i in self.solution_steps[0]]
        y = [self.citiesCoord[i][1] for i in self.solution_steps[0]]
        # Plots points in cyan circles
        plt.plot(x, y, 'co')
        self.ax.autoscale(enable=True, axis='both')

        # Initialize empty solution
        self.line.set_data([], [])

        return self.line,

    def update_plot(self, frame):
        x = [self.citiesCoord[i][0] for i in self.solution_steps[frame] + [self.solution_steps[frame][0]]]
        y = [self.citiesCoord[i][1] for i in self.solution_steps[frame] + [self.solution_steps[frame][0]]]
        self.line.set_data(x, y)
        #print(f"Solutions changes: {self.solution_steps[frame]}")
        return self.line

    def start_animation(self, solution_steps, citiesCoord):
        self.solution_steps = solution_steps
        self.citiesCoord = citiesCoord
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, frames=range(len(solution_steps)), 
                                           interval=300, init_func=self.init_plot, repeat=False)
        self.draw()

    def clear_fig(self):
        self.ax.clear()
        self.line, = self.ax.plot([], [], 'ro-', lw=2)
        self.ax.autoscale(enable=True, axis='both')
        self.draw()


if __name__ == '__main__':
    # QApplications class, manages main event loop, window system integration, settings // [] to pass command-line arguments
    app = QApplication([])

    visualizer = TSPVisualizer()
    visualizer = HeuristicDialog()
    visualizer.show()
    
    # app.exec() - event loop, wait for user action // sys.exit() called when terminate program
    sys.exit(app.exec())