import sys
import numpy as np
import os

from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QLineEdit,
                               QPushButton, QCheckBox, QGridLayout, QRadioButton, QComboBox, QGroupBox, QMessageBox,
                               QListWidget, QListWidgetItem, QDialog, QDialogButtonBox, QFileDialog, QFrame, QStyleFactory, 
                               QTextBrowser)
from PySide6.QtCore import QThread, Qt, Signal, Slot, QTimer

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT 
from matplotlib.figure import Figure
import matplotlib.animation as animation

import numpy as np
from tsplib_functions import load_tsplib_distance_matrix, save_tsp_instance
from random_cities import generate_random_cities
from v6_ChoiceFunctionGreatDeluge import *


class TSPVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.app = app  # Declare an app memeber
        self.heuristicDialog = HeuristicDialog()

        self.loaded_distanceMatrix = None
        self.loaded_matrixCoordinates = None
        self.main_distanceMatrix = None
        self.main_coordinates = None

        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.updateAnimationStep)
        self.current_frame = 0
        self.solution_steps = None
        self.logEntries = None

        # Other visualsitons
        self.objectiveVisual = None

        self.setupUI()

    def setupUI(self):
        # Set the main window
        self.setWindowTitle("TSP Visualizer")
        self.setGeometry(100, 40, 1200, 750)

        # Widget set as central widget
        widget = QWidget()
        self.setCentralWidget(widget)

        # Vertical Layout to hold widget
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Menu bar and items
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        self.load_action = file_menu.addAction("Load")
        self.save_action = file_menu.addAction("Save")
        self.quit_action = file_menu.addAction("Quit")
        self.load_action.triggered.connect(self.loadTSPLIB)
        self.save_action.triggered.connect(self.saveTSPLIB)
        self.quit_action.triggered.connect(self.quitApp)

        inputLayout = QHBoxLayout()

        setup = QGroupBox("Setup")
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
        self.iterationCheckBox.setChecked(True)

        # Solution initalisation drop-down
        self.solution_initialize_label = QLabel("Solution Initialization:")
        self.solution_initialize_comboBox = QComboBox()
        self.solution_initialize_comboBox.addItem('None')
        self.solution_initialize_comboBox.addItem('Random')
        self.solution_initialize_comboBox.addItem('Nearest Neighbour')
        tspSetup_gridLayout.addWidget(self.solution_initialize_label, 2, 0)
        tspSetup_gridLayout.addWidget(self.solution_initialize_comboBox, 2, 1)

        # Decorative Line
        heuristicLine = QFrame()
        heuristicLine.setFrameShape(QFrame.HLine)    # Set frame shape to horizontal
        heuristicLine.setFrameShadow(QFrame.Raised)  # Set frame shadow
        tspSetup_gridLayout.addWidget(heuristicLine, 3, 0, 1, 2) 

        # Heuristic operator selection
        self.low_lvl_heuristicLabel = QLabel("Low-Level Heurisitics:")
        self.low_lvl_heuristicButton = QPushButton("Select Operators")
        self.low_lvl_heuristicButton.clicked.connect(self.selectHeuristics)
        tspSetup_gridLayout.addWidget(self.low_lvl_heuristicLabel, 4, 0)
        tspSetup_gridLayout.addWidget(self.low_lvl_heuristicButton, 4, 1)
        
        self.crossover_checkBox = QCheckBox("Enable Crossover")
        tspSetup_gridLayout.rowStretch(4)
        tspSetup_gridLayout.addWidget(self.crossover_checkBox, 5, 0)  # Position of widgit is 4th row, 0 column, size takes 1 row and columns
        self.crossover_checkBox.toggled.connect(self.enableCrossover)
        self.enableCrossover(False)
        self.heuristicDialog.selections = self.heuristicDialog.getSelected()

        setup.setLayout(tspSetup_gridLayout)
        inputLayout.addWidget(setup)

        #Hyper-Heurisitc group box
        hyperheuristic_groupBox = QGroupBox("Great Deluge (GD) - Choice Function (CF)")
        hyperh_groupBoxLayout = QVBoxLayout()
        hyperheuristic_groupBox.setLayout(hyperh_groupBoxLayout)
        # Great Deluge decay type
        self.gd_waterLevel_label = QLabel("GD Level Decay Type:")
        self.gd_waterLevel_label_comboBox = QComboBox()
        self.gd_waterLevel_label_comboBox.addItem('Linear')
        self.gd_waterLevel_label_comboBox.addItem('Exponential')
        self.gd_waterLevel_label_comboBox.addItem('Sinusoidal')
        self.gd_decayRate_label = QLabel("GD Decay Rate:")
        self.gd_decayRate_input = QLineEdit()
        self.gd_decayRate_input.setPlaceholderText("Set decay rate (e.g. 0.1)")
        
        gd_cf_gridLayout = QGridLayout()
        gd_cf_gridLayout.addWidget(self.gd_waterLevel_label, 0, 0)
        gd_cf_gridLayout.addWidget(self.gd_waterLevel_label_comboBox, 0, 1)
        gd_cf_gridLayout.addWidget(self.gd_decayRate_label, 1, 0)
        gd_cf_gridLayout.addWidget(self.gd_decayRate_input, 1, 1)

        # Decorative Line
        cfLine = QFrame()
        cfLine.setFrameShape(QFrame.HLine)    # Set frame shape to horizontal
        cfLine.setFrameShadow(QFrame.Raised)  # Set frame shadow
        gd_cf_gridLayout.addWidget(cfLine, 2, 0, 1, 2) 

        # Choice Funciton parameters
        self.cf_phi_label = QLabel("Choice Function Phi:")
        self.cf_delta_label = QLabel("Choice Function Delta:")
        self.cf_phi_input = QLineEdit("0.5")
        self.cf_phi_input.setPlaceholderText("Set CF phi parameter")
        self.cf_delta_input = QLineEdit("0.5")
        self.cf_delta_input.setPlaceholderText("Set CF delta parameter")

        gd_cf_gridLayout.addWidget(self.cf_phi_label, 3, 0)
        gd_cf_gridLayout.addWidget(self.cf_phi_input, 3, 1)
        gd_cf_gridLayout.addWidget(self.cf_delta_label, 4, 0)
        gd_cf_gridLayout.addWidget(self.cf_delta_input, 4, 1)

        hyperh_groupBoxLayout.addLayout(gd_cf_gridLayout)

        inputLayout.addWidget(hyperheuristic_groupBox)

        #TSP Instance group box
        tspInstance_groupBox = QGroupBox("TSP Instance")
        tspInstance_groupBoxLayout = QVBoxLayout()
        tspInstance_groupBox.setLayout(tspInstance_groupBoxLayout)
        
        tspInstance_gridLayout = QGridLayout()
        # Number of TSP cities input
        self.numCitiesRadioButton = QRadioButton("Number of Cities:")
        self.numCitiesInput = QLineEdit()
        self.numCitiesInput.setPlaceholderText("Set number of cities (4-500)")
        tspInstance_gridLayout.addWidget(self.numCitiesRadioButton, 0, 0)
        tspInstance_gridLayout.addWidget(self.numCitiesInput, 0, 1)

        # Decorative Line
        tsplibLine = QFrame()
        tsplibLine.setFrameShape(QFrame.HLine)    # Set frame shape to horizontal
        tsplibLine.setFrameShadow(QFrame.Raised)  # Set frame shadow
        tspInstance_gridLayout.addWidget(tsplibLine, 1, 0, 1, 2) 

        # Load TSPLIB file
        self.tsplib_load_RadioButton = QRadioButton("Load TSPLIB File:")
        self.tsplibButton = QPushButton("Load")
        self.tsplib_loadedLabel = QLabel("Loaded File:")
        self.tsplib_loadedFileLabel = QLabel("Empty")
        self.tsplibButton.clicked.connect(self.loadTSPLIB)
        tspInstance_gridLayout.addWidget(self.tsplib_load_RadioButton, 2, 0)
        tspInstance_gridLayout.addWidget(self.tsplibButton, 2, 1)
        tspInstance_gridLayout.addWidget(self.tsplib_loadedLabel, 3, 0)
        tspInstance_gridLayout.addWidget(self.tsplib_loadedFileLabel, 3, 1)

        # Decorative Line
        manualCitiesLine = QFrame()
        manualCitiesLine.setFrameShape(QFrame.HLine)    # Set frame shape to horizontal
        manualCitiesLine.setFrameShadow(QFrame.Raised)  # Set frame shadow
        tspInstance_gridLayout.addWidget(manualCitiesLine, 4, 0, 1, 2) 

        # Creat manual cities
        self.manual_cities_RadioButton = QRadioButton("Manually Create Cities:")
        tspInstance_gridLayout.addWidget(self.tsplib_load_RadioButton, 5, 0)

        self.numCitiesRadioButton.toggled.connect(self.tspSelection_stateChange)
        self.tsplib_load_RadioButton.toggled.connect(self.tspSelection_stateChange)
        self.numCitiesRadioButton.setChecked(True)

        tspInstance_groupBoxLayout.addLayout(tspInstance_gridLayout)
        inputLayout.addWidget(tspInstance_groupBox)

        # Buttons box
        buttonsBox = QHBoxLayout()
        # Pause/Resume button
        self.solvingLabel = QLabel("")

        buttonsBox.addStretch()
        buttonsBox.addWidget(self.solvingLabel)

        buttonsBox.addStretch()

        # Decorative line after button box
        buttonBoxLine = QFrame()
        buttonBoxLine.setFrameShape(QFrame.HLine)    # Set frame shape to horizontal
        buttonBoxLine.setFrameShadow(QFrame.Raised)  # Set frame shadow

        # Start button
        self.startButton = QPushButton("Start")
        self.startButton.clicked.connect(self.startSolving)
        buttonsBox.addWidget(self.startButton)

        # TSP plot canvas
        viualsLayout = QHBoxLayout()
        self.canvas = TSPCanvas(self)
        #grid_plot.addWidget(self.canvas, 0, 0, 2, 3)
        #grid_plot.setColumnStretch(0,2)
        #grid_plot.setColumnMinimumWidth(3, 200)
        viualsLayout.addWidget(self.canvas)

        otherVisLayout = QVBoxLayout()
        self.otherVisualLabel = QLabel("Other Visualisations")
        self.objectiveVis_button = QPushButton("Objective Value")
        self.gdwaterVis_button = QPushButton("Great Deluge Water Value")
        self.heatmapVis_button = QPushButton("Heuristic Frequency Chart")

        self.objectiveVis_button.clicked.connect(self.displayObjectiveVis)
        self.gdwaterVis_button.clicked.connect(self.displayGDwaterVis)
        self.heatmapVis_button.clicked.connect(self.displayHeatmapVis)
        
        #grid_plot.addWidget(self.otherVisualLabel, 0, 3)
        otherVisLayout.addWidget(self.otherVisualLabel)

        # Decorative line after label
        otherVisLine = QFrame()
        otherVisLine.setFrameShape(QFrame.HLine)    # Set frame shape to horizontal
        otherVisLine.setFrameShadow(QFrame.Raised)  # Set frame shadow
        otherVisLayout.addWidget(otherVisLine)

        otherVisLayout.addWidget(self.objectiveVis_button)
        otherVisLayout.addWidget(self.gdwaterVis_button)
        otherVisLayout.addWidget(self.heatmapVis_button)
        otherVisLayout.addStretch()
        
        viualsLayout.addLayout(otherVisLayout)
        viualsLayout.setStretch(0, 8)
        viualsLayout.setStretch(1, 2)
        
        # Algorithm log
        self.heuristicLogs = QTextBrowser()
        
        layout.addLayout(viualsLayout)
        layout.addLayout(buttonsBox)
        layout.addWidget(buttonBoxLine)
        layout.addLayout(inputLayout)
        layout.addWidget(QLabel("TSP Log"))
        layout.addWidget(self.heuristicLogs)
        
        self.objectiveValue_canvas = ObjectiveValueCanvas(self)
        self.objectiveValue_canvas.setWindowTitle("Objective Value Change")
        self.gdWaterValue_canvas = GDWaterValueCanvas(self)
        self.gdWaterValue_canvas.setWindowTitle("Water Level Change")
        self.histogram_canvas = HeuristicHistogramCanvas(self)
        self.histogram_canvas.setWindowTitle("Heuristic Histogram")

    def displayObjectiveVis(self):
        self.objectiveValue_canvas.show()
    
    def displayGDwaterVis(self):
        self.gdWaterValue_canvas.show()

    def displayHeatmapVis(self):
        self.histogram_canvas.show()

    def quitApp(self):
        self.app.quit()

    def saveTSPLIB(self):
        if self.main_coordinates is not None:           #Can only save if coordinates set
            filePath, _ = QFileDialog.getSaveFileName(self,
                                                    "Save TSP Instance", 
                                                    '',
                                                    "TSPLIB (*.tsp)")
            if not filePath:
                return
            
            try:
                save_tsp_instance(filePath, self.main_coordinates)

                QMessageBox.information(self, "File Saved", f'TSP instance saved to: "{filePath}"')

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")
        

    # Loads TSPLIB instance 
    def loadTSPLIB(self):
        filePath, fileType = QFileDialog.getOpenFileName(self, 
                                                "Select TSPLIB file",
                                                '',
                                                "TSPLIB (*.tsp)")
        
        if not filePath:
            return
        
        if not filePath.endswith('.tsp'):
            filePath += '.tsp'

        fileName = os.path.basename(filePath)
        try:
            distanceMatrix, coordinates = load_tsplib_distance_matrix(filePath)
            # Set currently loaded file
            self.tsplib_loadedFileLabel.setText(fileName)
            self.loaded_distanceMatrix = distanceMatrix
            self.loaded_matrixCoordinates = coordinates

            QMessageBox.information(self, "File Loaded", f'Loaded file: {fileName}')

        except Exception as e:
            QMessageBox.critical(self, "Unable to open file",
                                    f'There was an error opening "{fileName}": \n{str(e)}')
            return
    
    # Updates selected heuristics list
    def selectHeuristics(self):
        if(self.heuristicDialog.exec()):
            self.heuristicDialog.selections = self.heuristicDialog.getSelected()
            #print(self.heuristicDialog.selections)
        else:
            #print("cancel")
            self.heuristicDialog.resetSelection()

    # Enable crossover heuristics to be applied
    def enableCrossover(self, isEnabled=None):
        if self.crossover_checkBox.checkState() == Qt.CheckState.Checked or isEnabled == True:
            self.heuristicDialog.toggleCrossoverSelection(True)
        elif self.crossover_checkBox.checkState() == Qt.CheckState.Unchecked or isEnabled == False:
            self.heuristicDialog.toggleCrossoverSelection(False)
    
    # Disable input area based on checked box
    def time_iteration_stateChange(self):
        if self.timeLimitCheckbox.isChecked():
            self.timeLimitInput.setEnabled(True)
            self.iterationLimitInput.setEnabled(False)
        elif self.iterationCheckBox.isChecked():
            self.timeLimitInput.setEnabled(False)
            self.iterationLimitInput.setEnabled(True)

    def tspSelection_stateChange(self):
        if self.numCitiesRadioButton.isChecked():   # Random input city enabled
            self.numCitiesInput.setEnabled(True)
            self.tsplibButton.setEnabled(False)
            self.load_action.setEnabled(False)
        elif self.tsplib_load_RadioButton.isChecked():  # Load TSPLIb enabled
            self.numCitiesInput.setEnabled(False)
            self.tsplibButton.setEnabled(True)
            self.load_action.setEnabled(True)

    def startSolving(self):
        self.canvas.clear_fig()
        self.heuristicLogs.clear()

        try:
            # Set the TSP instance
            if self.numCitiesRadioButton.isChecked():
                num_cities = int(self.numCitiesInput.text())
                self.main_distanceMatrix, self.main_coordinates = generate_random_cities(num_cities, 500, 500)
            elif self.tsplib_load_RadioButton.isChecked():
                self.main_distanceMatrix = self.loaded_distanceMatrix
                self.main_coordinates = self.loaded_matrixCoordinates

            # Initialise ProblemDomain and Hyper-Heuristic
            problem = ProblemDomain(self.main_distanceMatrix)
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
            hyperH.setDeacyModel(gd_decay_type)
            gd_decay_rate = float(self.gd_decayRate_input.text())
            hyperH.setDecayRate(gd_decay_rate)

            # Choice Function parameters
            phi = float(self.cf_phi_input.text())
            delta = float(self.cf_delta_input.text())
            hyperH.setPhiDelta_CF(phi, delta)

            # Heuristic Selections
            hyperH.isCrossoverAllowed(self.crossover_checkBox.isChecked())
            hyperH.setSelectedHeuristic(self.heuristicDialog.selections)
            #print(self.heuristicDialog.selections)

            hyperH.solve(problem)
            #solution = problem.getBestSolution()
            self.solution_steps = hyperH.all_solution_step
            #print(len(self.solution_steps))
            self.logEntries = hyperH.tsp_log.log_entries
            #print(self.logEntries[0])

            self.solvingLabel.setText("Solving...")
            self.startButton.setEnabled(False)
            
            # Start animation
            self.canvas.start_animation(self.solution_steps, self.main_coordinates)
            self.current_frame = 0
            self.animation_timer.start(300) # Update every 300ms
            
            # Plot objective value graph
            self.objectiveValue_canvas.plotObjectiveValue(hyperH.all_Objective_value, problem.init_solution_value, problem.getBestSolutionValue())
            self.objectiveValue_canvas.draw_idle()

            # Plot Great Deluge water level graph
            self.gdWaterValue_canvas.plotWaterValue(hyperH.all_waterLevelChange)
            self.gdWaterValue_canvas.draw_idle()

            # Plot Histogram of heuristic usage
            heuristicNames = problem.getHeuristicNames()
            heuristicCount = problem.getHeuristicCounts()

            self.histogram_canvas.plotHistogram(heuristicNames, heuristicCount)
            self.histogram_canvas.draw_idle()


        except Exception as e:
            QMessageBox.information(self, "Error - Unable to Start",
                                    f'Please Ensure Correct Values are Entered: \n{str(e)}')
            self.animation_timer.stop()
            self.finalLog()
            self.solvingLabel.setText("")
            self.startButton.setEnabled(True)
            return
        

    def updateAnimationStep(self):
        if self.current_frame < len(self.solution_steps):
            self.canvas.animate_step(self.current_frame)
            self.updateLog(self.current_frame)
            #print(self.current_frame)
            self.current_frame += 1
        else:
            #print("final")
            self.animation_timer.stop()
            self.finalLog()
            self.solvingLabel.setText("")
            self.startButton.setEnabled(True)

    def updateLog(self, frame):
        log = self.logEntries[frame]
        text = (f"Step: {log['iteration']}  -  Objective Value: {log['best_obj_value']}  -  Heuristic: {log['move_type']}" 
                f"\nWater Level: {log['gd_waterlevel']}; Phi: {log['phi']}, Delta: {log['delta']} "
                "\n-------------------")
        self.heuristicLogs.append(text)

    def finalLog(self):
        log = self.logEntries[-1]
        text = ("--------Final--------\n"
                f"Solution: {log['solution']}\nStep: {log['iteration']}  -  Best Value: {log['best_obj_value']}  -  Convergence: {log['convergence']} " 
                f"\nWater Level:  {log['gd_waterlevel']}  -  Phi: {log['phi']}  -  Delta: {log['delta']}"
                "\n-------------------")
        self.heuristicLogs.append(text)
        

class HeuristicDialog(QDialog):
    # Dialog box to select low-level heuristics
    def __init__(self, parent=None):
        super().__init__(parent)
        self.crossover = ProblemDomain(np.zeros((2,2))).getHeursiticsOfType("CROSSOVER")
        self.selections = []
        
        self.low_lvl_heuristicList = QListWidget()
        # Holds list of selectable heuristics
        self.add_checkableItems([
                                ("Swap Mutation", 0),
                                ("Inversion Mutation", 1), 
                                ("Scramble Mutation", 2), 
                                ("Insert Mutation", 3),
                                ("Displacement Mutation", 4), 
                                ("Two Opt", 5), 
                                ("Nearest Neighhour", 6),
                                ("Simulated Annealing", 7),
                                ("Order Crossover", 8),
                                ("Partially Mapped Crossover", 9),
                                ("Position-based Crossover", 10),
                                ("One-Point Crossover", 11),
                                ("Ruin-Recreate", 12)])

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok
                                      | QDialogButtonBox.Cancel)

        layout = QVBoxLayout()
        layout.addWidget(self.low_lvl_heuristicList)
        layout.addWidget(self.button_box)

        self.setLayout(layout)

        self.setWindowTitle("Low-Level Heuristic Selection")

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

        self.selections = self.getSelected()
        #print(self.selections)

    def toggleCrossoverSelection(self, isEnabled):
        for i in range(self.low_lvl_heuristicList.count()):
            item = self.low_lvl_heuristicList.item(i)
            if item.data(Qt.UserRole) in self.crossover: 
                if not isEnabled:
                    item.setFlags(Qt.ItemFlag.NoItemFlags)
                else:
                    item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            else:
                if isEnabled:
                    item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        
        self.selections = self.getSelected()
        #print(self.selections)


# Inherits from FigureCanvas
class TSPCanvas(FigureCanvasQTAgg):
    # Parent argument is qt widget where canvas is displayed
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0,500)
        self.ax.set_ylim(0,500)
        # Pass figure to super class figureCanvas
        super().__init__(self.fig)
        
        # Set parent widget for hierarchy
        self.setParent(parent)
        self.solution_steps = []
        self.citiesCoord = []
        #self.line, = self.ax.plot([], [], 'ro-', lw=2)
        self.line_segments = []     # Store the lines between cities
        self.ani = None
        self.default_colour = 'green'
        self.changed_colour = 'r'

    def init_plot(self):
        # Draw cities
        x = [self.citiesCoord[i][0] for i in self.solution_steps[0]]
        y = [self.citiesCoord[i][1] for i in self.solution_steps[0]]
        # Plots points in cyan circles
        self.ax.plot(x, y, 'co')
        self.ax.autoscale(enable=True, axis='both')
        """
        # draw axes slighty bigger
        extra_x = (max(x) - min(x)) * 0.05
        extra_y = (max(y) - min(y)) * 0.05
        self.ax.set_xlim(min(x) - extra_x, max(x) + extra_x)
        self.ax.set_ylim(min(y) - extra_y, max(y) + extra_y)
        """

        # Initialize empty solution
        #self.line.set_data([], [])
        self.line_segments.clear()

        initial_solution = self.solution_steps[0]
        for i in range(len(initial_solution)):
            start_city = initial_solution[i]
            end_city = initial_solution[(i + 1) % len(initial_solution)]  # Loop back to the start
            # Plot line between cities
            line, = self.ax.plot(
                [self.citiesCoord[start_city][0], self.citiesCoord[end_city][0]],
                [self.citiesCoord[start_city][1], self.citiesCoord[end_city][1]],
                color=self.default_colour, lw=2)
            self.line_segments.append(line)

        # Store the initial solution as the previous solution
        self.prev_solution = initial_solution.copy()

        #return self.line,

    def update_plot(self, frame):
        x = [self.citiesCoord[i][0] for i in self.solution_steps[frame] + [self.solution_steps[frame][0]]]
        y = [self.citiesCoord[i][1] for i in self.solution_steps[frame] + [self.solution_steps[frame][0]]]
        self.line.set_data(x, y)
        #print(f"Solutions changes: {self.solution_steps[frame]}")
        return self.line

    def start_animation(self, solution_steps, citiesCoord):
        self.solution_steps = solution_steps
        self.citiesCoord = citiesCoord
        #self.ani = animation.FuncAnimation(self.fig, self.update_plot, frames=range(len(solution_steps)), 
        #                                   interval=300, init_func=self.init_plot, repeat=False)
        self.init_plot()
        self.draw()

    def animate_step(self, frame):
        """x = [self.citiesCoord[i][0] for i in self.solution_steps[frame] + [self.solution_steps[frame][0]]]
        y = [self.citiesCoord[i][1] for i in self.solution_steps[frame] + [self.solution_steps[frame][0]]]
        self.line.set_data(x, y)
        self.draw()"""

        if frame == 0:
            return  # Skip the first frame, as there's no previous solution to compare with.

        current_solution = self.solution_steps[frame]

        # We will batch update only the changed lines.
        changed_indices = []

        # Loop through each city in the current solution and compare with the previous solution
        for i in range(len(current_solution)):
            start_city_current = current_solution[i]
            end_city_current = current_solution[(i + 1) % len(current_solution)]  # Loop back to the start

            start_city_prev = self.prev_solution[i]
            end_city_prev = self.prev_solution[(i + 1) % len(self.prev_solution)]

            if {start_city_current, end_city_current} != {start_city_prev, end_city_prev}:
                # The connection between these cities has changed, update the coordinates and highlight it
                self.line_segments[i].set_data(
                    [self.citiesCoord[start_city_current][0], self.citiesCoord[end_city_current][0]],
                    [self.citiesCoord[start_city_current][1], self.citiesCoord[end_city_current][1]]
                )
                self.line_segments[i].set_color(self.changed_colour)
                changed_indices.append(i)
            else:
                # Reset to default color if no change
                self.line_segments[i].set_color(self.default_colour)

        # Batch update: only draw once after all the changes are made
        if changed_indices:
            self.draw()

        # Update the previous solution to the current one
        self.prev_solution = current_solution.copy()

    def clear_fig(self):
        self.ax.clear()
        #self.line, = self.ax.plot([], [], 'ro-', lw=2)
        self.line_segments.clear()
        self.ax.autoscale(enable=True, axis='both')
        self.draw()


class ObjectiveValueCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.figObjective, self.axObjective = plt.subplots()
        # Pass figure to super class figureCanvas
        super().__init__(self.figObjective)

        #self.toolbar = NavigationToolbar2QT

    def plotObjectiveValue(self, objectiveValueList, initSolution, bestSolution):
        self.axObjective.clear()

        self.axObjective.plot([i for i in range(len(objectiveValueList))], objectiveValueList)
        # Line Labels
        init_line = self.axObjective.axhline(y=initSolution, color='r', linestyle='--')
        best_line = self.axObjective.axhline(y=bestSolution, color='g', linestyle='--')
        self.axObjective.legend([init_line, best_line], ['Initial Objective Value', 'Optimized Objective Value'])
        
        # Axis labels
        self.axObjective.set_ylabel('Objective Value')
        self.axObjective.set_xlabel('Iteration')
        
        self.axObjective.set_title("Objective Level Over Time")
        self.draw_idle()


class GDWaterValueCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots()
        # Pass figure to super class figureCanvas
        super().__init__(self.fig)

    def plotWaterValue(self, water_level):
        self.ax.clear()

        self.ax.plot([i for i in range(len(water_level))], water_level, label="Water Level")

        self.ax.set_ylabel('Water Level')
        self.ax.set_xlabel('Iteration')
        self.ax.set_title("Great Deluge Water Level Over Time")
        #self.ax.legend()

        self.draw_idle()


class HeuristicHistogramCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots(figsize=(8,6))
        # Pass figure to super class figureCanvas
        super().__init__(self.fig)
        self.ax.autoscale(enable=True, axis='both')

    def plotHistogram(self, heuristic_names, heuristic_count):
        self.ax.clear()

        # Plot bar chart
        bars = self.ax.bar(heuristic_names, heuristic_count, color='skyblue')

        # Add labels and title
        self.ax.set_xlabel('Heuristics (Operators)')
        self.ax.set_ylabel('Frequency of Use')
        self.ax.set_title('Operator Usage Histogram')

        self.fig.subplots_adjust(bottom=0.3)

        # Rotate x-axis labels 
        self.ax.set_xticks(range(len(heuristic_names)))
        self.ax.set_xticklabels(heuristic_names, rotation=60, ha="right", fontsize=8)

        # frequency labels on top of each bar
        for bar in bars:
            yval = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom')  # Add value label

        self.ax.get_tightbbox()

        self.draw_idle()

if __name__ == '__main__':
    # QApplications class, manages main event loop, window system integration, settings // [] to pass command-line arguments
    app = QApplication([])

    visualizer = TSPVisualizer()
    #visualizer = HeuristicDialog()
    visualizer.show()
    
    # app.exec() - event loop, wait for user action // sys.exit() called when terminate program
    sys.exit(app.exec())