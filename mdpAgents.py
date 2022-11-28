# mdpAgents.py
# parsons/20-nov-2017
#
# Version 1
#
# The starting point for CW2.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here is was written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import math
import game
import util


class Grid:

    # Constructor
    #
    # Note that it creates variables:
    #
    # grid:   an array that has one position for each element in the grid.
    # width:  the width of the grid
    # height: the height of the grid
    #
    # Grid elements are not restricted, so you can place whatever you
    # like at each location. You just have to be careful how you
    # handle the elements when you use them.
    def __init__(self, width, height):
        self.width = width
        self.height = height
        subgrid = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(0)
            subgrid.append(row)

        self.grid = subgrid

    # The display function prints the grid out upside down. This
    # prints the grid out so that it matches the view we see when we
    # look at Pacman.
    def prettyDisplay(self):
        for i in range(self.height):
            for j in range(self.width):
                # print grid elements with no newline
                print self.grid[self.height - (i + 1)][j],
            # A new line after each line of the grid
            print
            # A line after the grid
        print

    # Set and get the values of specific elements in the grid.
    # Here x and y are indices.
    def setValue(self, x, y, value):
        self.grid[y][x] = value

    def getValue(self, x, y):
        return self.grid[y][x]

    # Return width and height to support functions that manipulate the
    # values stored in the grid.
    def getHeight(self):
        return self.height

    def getWidth(self):
        return self.width

class MDPAgent(Agent):
    GAMMA = 0.5
    GAMMA_SMALL_MAP = 0.85
    EPSILON = 0.0001
    FOOD_REWARD = 10
    CAPSULE_REWARD = 80
    GHOST_REWARD = -70
    SURROUNDING_GHOST_REWARD = -50
    GHOST_REWARD_SMALL_MAP = -70
    SURROUNDING_GHOST_REWARD_SMALL_MAP = -60
    SCARED_GHOST_REWARD = 100
    LAST_FOOD_REWARD = 30
    DEFAULT_REWARD = 0
    SMALL_MAP_REWARD = -6
    DEFAULT_UTILITY = 0

    # Constructor: this gets run when we first invoke pacman.py
    def __init__(self):
        print ("Starting up MDPAgent!")
        self.pacman_location = None
        self.food_locations = None
        self.corner_locations = None
        self.ghost_states = None
        self.capsule_locations = None
        self.wall_locations = None
        self.legal_actions = None
        self.utility_dictionary = {}
        self.reward_dictionary = {}
        self.ghost_locations = None
        self.map = None

    # Gets run after an MDPAgent object is created and once there is
    # game state to access.
    def registerInitialState(self, state):
        print ("Running registerInitialState for MDPAgent!")
        print ("I'm at:")
        print api.whereAmI(state)
        
    # This is what gets run in between multiple games
    def final(self, state):
        print "Looks like the game just ended!"
        self.pacman_location = None
        self.food_locations = None
        self.corner_locations = None
        self.ghost_states = None
        self.capsule_locations = None
        self.wall_locations = None
        self.legal_actions = None
        self.utility_dictionary = {}
        self.reward_dictionary = {}
        self.ghost_locations = None
        self.map = None

    def setUpStates(self, state):
        self.pacman_location = api.whereAmI(state)
        self.food_locations = api.food(state)
        self.capsule_locations = api.capsules(state)
        self.ghost_states = api.ghostStates(state)
        self.wall_locations = api.walls(state)
        self.corner_locations = api.corners(state)
        self.legal_actions = api.legalActions(state)
        self.ghost_locations = api.ghosts(state)

    def getHeight(self):
        height = -1
        for i in range(len(self.corner_locations)):
            if self.corner_locations[i][1] > height:
                height = self.corner_locations[i][1]
        return height + 1

    def getWidth(self):
        width = -1
        for i in range(len(self.corner_locations)):
            if self.corner_locations[i][0] > width:
                width = self.corner_locations[i][0]
        return width + 1

    def getGrid(self):
        grid = []
        width = self.getWidth()
        height = self.getHeight()
        for i in range(width):
            for j in range(height):
                grid.append((i, j))
        return grid

    def isMapSmall(self):
        return self.getWidth() <= 10 and self.getHeight() <= 10

    def createRewardAndUtilityMap(self):
        grid = self.getGrid()
        # Assigning default rewards and utility
        for location in grid:
            if location not in self.wall_locations:
                self.utility_dictionary[location] = self.DEFAULT_UTILITY
                self.reward_dictionary[location] = self.SMALL_MAP_REWARD if self.isMapSmall() else self.DEFAULT_REWARD

        # Food and capsules
        for food_location in self.food_locations:
            self.reward_dictionary[food_location] = self.FOOD_REWARD

        for capsule_location in self.capsule_locations:
            self.reward_dictionary[capsule_location] = self.CAPSULE_REWARD

        if len(self.food_locations) == 1:
            self.reward_dictionary[self.food_locations[0]] = self.LAST_FOOD_REWARD

    def getSurroundingCells(self, cell, border):
        surrounding_cells = []
        (x, y) = cell
        for i in range(border):
            delta = i + 1
            surrounding_cells.append((x + delta, y))
            surrounding_cells.append((x - delta, y))
            surrounding_cells.append((x, y + delta))
            surrounding_cells.append((x, y - delta))
        return surrounding_cells

    def isCellNeitherWallnorGhostCell(self, cell):
        return cell not in self.wall_locations and cell not in self.ghost_locations

    def getSurroundingCellsNotBeingObstructed(self, cells):
        return [x for x in cells if self.isCellNeitherWallnorGhostCell(x)]

    def getNeighboursOfVariousGhostCells(self, ghost_cells):
        ghost_cell_neighbours = [self.getSurroundingCells(i, 1) for i in ghost_cells]
        return self.getSurroundingCellsNotBeingObstructed(sum(ghost_cell_neighbours, []))

    def getDefaultReward(self):
        return self.SMALL_MAP_REWARD if self.isMapSmall() else self.DEFAULT_REWARD

    def getGhostReward(self):
        return self.GHOST_REWARD_SMALL_MAP if self.isMapSmall() else self.GHOST_REWARD

    def getSurroundingGhostReward(self):
        return self.SURROUNDING_GHOST_REWARD_SMALL_MAP if self.isMapSmall() else self.SURROUNDING_GHOST_REWARD

    # This function is just for big maps where it creates 2 borders to stay alive
    def getGhostNeighbours(self, ghost_cell):
        surrounding_ghost_cells = self.getSurroundingCells(ghost_cell, 1)
        surrounding_ghost_cells = self.getSurroundingCellsNotBeingObstructed(surrounding_ghost_cells)
        next_layer_cells = self.getNeighboursOfVariousGhostCells(surrounding_ghost_cells)
        surrounding_ghost_cells = [surrounding_ghost_cells, next_layer_cells]
        return set(sum(surrounding_ghost_cells, []))

    def surroundingLocationsOfGhostForSmallMap(self, ghost_location):
        (x, y) = ghost_location
        (x, y) = (int(x), int(y))
        ghost_location_upper_bound = (math.ceil(x), math.ceil(y))  # Like 4.5, 5.5 => 5, 6
        surrounding_danger_cells = []
        surrounding_danger_cells.append((x, y + 1))  # North
        surrounding_danger_cells.append((x, y - 1))  # South
        surrounding_danger_cells.append((x + 1, y))  # East
        surrounding_danger_cells.append((x - 1, y))  # West
        surrounding_danger_cells.append((x + 1, y + 1))  # North East
        surrounding_danger_cells.append((x + 1, y - 1))  # South East
        surrounding_danger_cells.append((x - 1, y + 1))  # North West
        surrounding_danger_cells.append((x - 1, y - 1))  # South West
        surrounding_danger_cells.append(ghost_location_upper_bound)
        surrounding_danger_cells = self.getSurroundingCellsNotBeingObstructed(surrounding_danger_cells)
        return surrounding_danger_cells

    def computeGhostRewards(self):
        ghost_reward = self.getGhostReward()
        surrounding_ghost_reward = self.getSurroundingGhostReward()
        for ghost_state in self.ghost_states:
            (x, y) = ghost_state[0]  # Location
            cell = (int(x), int(y))
            if ghost_state[1] == 0:
                self.reward_dictionary[cell] = ghost_reward
                self.utility_dictionary[cell] = ghost_reward
                ghost_neighbours = self.surroundingLocationsOfGhostForSmallMap((x, y)) if self.isMapSmall() else self.getGhostNeighbours(cell)
                for ghost_neighbour in ghost_neighbours:
                    self.reward_dictionary[ghost_neighbour] = surrounding_ghost_reward
            else:
                self.reward_dictionary[cell] = self.SCARED_GHOST_REWARD

    def compute_utility(self, cell, action_cell, perpendicular_cell1, perpendicular_cell2, utilDict):
        cell_1 = cell
        cell_2 = cell
        cell_3 = cell

        if action_cell not in self.wall_locations:
            cell_1 = action_cell

        if perpendicular_cell1 not in self.wall_locations:
            cell_2 = perpendicular_cell1

        if perpendicular_cell2 not in self.wall_locations:
            cell_3 = perpendicular_cell2

        return (0.8 * utilDict[cell_1]) + (0.1 * utilDict[cell_2]) + (0.1 * utilDict[cell_3])

    def getActionUtilities(self, cell, utilDict):
        x = cell[0]
        y = cell[1]

        north_cell = (x, y + 1)
        south_cell = (x, y - 1)
        east_cell = (x + 1, y)
        west_cell = (x - 1, y)

        utilities = []
        utilities.append(self.compute_utility(cell, north_cell, east_cell, west_cell, utilDict))  # North
        utilities.append(self.compute_utility(cell, south_cell, east_cell, west_cell, utilDict))  # South
        utilities.append(self.compute_utility(cell, east_cell, north_cell, south_cell, utilDict))  # East
        utilities.append(self.compute_utility(cell, west_cell, north_cell, south_cell, utilDict))  # West

        return utilities

    def getMaximumUtility(self, utilities):
        return max(utilities)

    def computeBellmanValue(self, reward, utility):
        gamma = self.GAMMA_SMALL_MAP if self.isMapSmall() else self.GAMMA
        return reward + (gamma * utility)

    def isStateNonTerminal(self, utility, terminal_utility):
        return utility != terminal_utility

    def valueIteration(self):
        terminal_utility = self.getGhostReward()
        while True:
            delta = 0
            for cell in self.utility_dictionary.items():
                utility = cell[1]
                location = cell[0]
                # This is making the ghost position terminal
                if self.isStateNonTerminal(utility, terminal_utility):
                    action_utilities = self.getActionUtilities(location, self.utility_dictionary)
                    bestUtility = self.getMaximumUtility(action_utilities)
                    reward = self.reward_dictionary[location]
                    self.utility_dictionary[location] = self.computeBellmanValue(reward, bestUtility)
                    delta = max(delta, abs(self.utility_dictionary[location] - utility))
                else:
                    self.utility_dictionary[location] = utility
            if delta < self.EPSILON * (1 - self.GAMMA) / self.GAMMA:
                break

    def getBestMove(self):
        possibleStates = [i for i in self.getSurroundingCells(self.pacman_location, 1) if i not in self.wall_locations]
        surrounding_state_utility = [self.utility_dictionary[i] for i in possibleStates]
        best_move_location = possibleStates[surrounding_state_utility.index(max(surrounding_state_utility))]
        best_x, best_y = best_move_location
        x, y = self.pacman_location
        # Getting the best location
        if x == best_x:
            if best_y < y:
                return Directions.SOUTH
            else:
                return Directions.NORTH
        else:
            if best_x < x:
                return Directions.WEST
            else:
                return Directions.EAST

    # For now I just move randomly
    def getAction(self, state):
        # Get the actions we can try, and remove "STOP" if that is one of them.
        self.setUpStates(state)
        self.createRewardAndUtilityMap()
        self.computeGhostRewards()
        self.valueIteration()
        best_move = self.getBestMove()
        if Directions.STOP in self.legal_actions:
            self.legal_actions.remove(Directions.STOP)

        return api.makeMove(best_move, self.legal_actions)
