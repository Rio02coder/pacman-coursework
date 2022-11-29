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


class MDPAgent(Agent):
    # Constants
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
    SMALL_MAP_BOUNDARY = 8

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

    # Gets run after an MDPAgent object is created and once there is
    # game state to access.
    def registerInitialState(self, state):
        print("Running registerInitialState for MDPAgent!")
        print("I'm at:")
        print(api.whereAmI(state))
        
    # This is what gets run in between multiple games
    def final(self, state):
        print("Looks like the game just ended!")
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

    def setUpStates(self, state):
        """ This method gets the state data at once
        and avoids repeat access """
        self.pacman_location = api.whereAmI(state)
        self.food_locations = api.food(state)
        self.capsule_locations = api.capsules(state)
        self.ghost_states = api.ghostStates(state)
        self.wall_locations = api.walls(state)
        self.corner_locations = api.corners(state)
        self.legal_actions = api.legalActions(state)
        self.ghost_locations = api.ghosts(state)

    def getHeight(self):
        # Code from MapAgents Simon Parsons
        height = -1
        for i in range(len(self.corner_locations)):
            if self.corner_locations[i][1] > height:
                height = self.corner_locations[i][1]
        return height + 1

    def getWidth(self):
        # Code from MapAgents Simon Parsons
        width = -1
        for i in range(len(self.corner_locations)):
            if self.corner_locations[i][0] > width:
                width = self.corner_locations[i][0]
        return width + 1

    def getGrid(self):
        """This block of code gives all the
        locations in the map currently."""
        grid = []
        width = self.getWidth()
        height = self.getHeight()
        for i in range(width):
            for j in range(height):
                grid.append((i, j))
        return grid

    def isMapSmall(self):
        """This method checks if the map
        is considerably small."""
        return self.getWidth() <= self.SMALL_MAP_BOUNDARY and self.getHeight() <= self.SMALL_MAP_BOUNDARY

    def createRewardAndUtilityMap(self):
        """This creates the reward and utility maps with
        the default values first and then updates them with
        the special locations like food and capsules. This also
        takes the last available food into consideration."""
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

        # Last food available
        if len(self.food_locations) == 1:
            self.reward_dictionary[self.food_locations[0]] = self.LAST_FOOD_REWARD

    def getSurroundingCells(self, cell, border):
        """This method gives the immediate surrounding cells
        for the location. It also takes a border, which says
        the extent of the surrounding cells required. For instance,
        if you require 2 cells E, W, N, S, then border should be 2."""
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

    def getGhostNeighbours(self, ghost_cell):
        """This method is used for big maps where
        it gives a block of surrounding cells of size
        two units."""
        surrounding_ghost_cells = self.getSurroundingCells(ghost_cell, 1)
        surrounding_ghost_cells = self.getSurroundingCellsNotBeingObstructed(surrounding_ghost_cells)
        next_layer_cells = self.getNeighboursOfVariousGhostCells(surrounding_ghost_cells)
        surrounding_ghost_cells = [surrounding_ghost_cells, next_layer_cells]
        return set(sum(surrounding_ghost_cells, []))

    def surroundingLocationsOfGhostForSmallMap(self, ghost_location):
        """This method gives the surrounding cells
        for the small map. It just gives the surrounding
        cells and the diagonals and the upper bound of ghost location"""
        (x, y) = ghost_location
        (x, y) = (int(x), int(y))
        ghost_location_upper_bound = (math.ceil(x), math.ceil(y))  # Like 4.5, 5.5 => 5, 6
        surrounding_danger_cells = []
        surrounding_danger_cells.append((x, y + 1))  # North
        surrounding_danger_cells.append((x, y - 1))  # South
        surrounding_danger_cells.append((x + 1, y))  # East
        surrounding_danger_cells.append((x - 1, y))  # West
        # Diagonals
        surrounding_danger_cells.append((x + 1, y + 1))
        surrounding_danger_cells.append((x + 1, y - 1))
        surrounding_danger_cells.append((x - 1, y + 1))
        surrounding_danger_cells.append((x - 1, y - 1))
        surrounding_danger_cells.append(ghost_location_upper_bound)
        surrounding_danger_cells = self.getSurroundingCellsNotBeingObstructed(surrounding_danger_cells)
        return surrounding_danger_cells

    def computeGhostRewards(self):
        """This method computes the ghost reward
        for the map, based on if the map is small
        or big."""
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

    def compute_utility(self, cell, action_cell, perpendicular_cell1, perpendicular_cell2, util_dict):
        """This method computes the utility of the cell.
        This takes into consideration the reachable locations."""
        cell_1 = cell
        cell_2 = cell
        cell_3 = cell

        if action_cell not in self.wall_locations:
            cell_1 = action_cell

        if perpendicular_cell1 not in self.wall_locations:
            cell_2 = perpendicular_cell1

        if perpendicular_cell2 not in self.wall_locations:
            cell_3 = perpendicular_cell2

        return (0.8 * util_dict[cell_1]) + (0.1 * util_dict[cell_2]) + (0.1 * util_dict[cell_3])

    def getActionUtilities(self, cell, util_dict):
        """This method returns the utilities for all
        the possible actions. N, S, E, W."""
        x = cell[0]
        y = cell[1]

        north_cell = (x, y + 1)
        south_cell = (x, y - 1)
        east_cell = (x + 1, y)
        west_cell = (x - 1, y)

        utilities = []
        utilities.append(self.compute_utility(cell, north_cell, east_cell, west_cell, util_dict))  # North
        utilities.append(self.compute_utility(cell, south_cell, east_cell, west_cell, util_dict))  # South
        utilities.append(self.compute_utility(cell, east_cell, north_cell, south_cell, util_dict))  # East
        utilities.append(self.compute_utility(cell, west_cell, north_cell, south_cell, util_dict))  # West

        return utilities

    def getMaximumUtility(self, utilities):
        return max(utilities)

    def computeBellmanValue(self, reward, utility):
        gamma = self.GAMMA_SMALL_MAP if self.isMapSmall() else self.GAMMA
        return reward + (gamma * utility)

    def isStateNonTerminal(self, utility, terminal_utility):
        return utility != terminal_utility

    def valueIteration(self):
        """This is the main method for value iteration.
        This uses the algorithm mentioned in Russel and Norvig.
        Conceptually, this method would run till the values converge
        within an acceptable noise or error."""
        terminal_utility = self.getGhostReward()  # This is used to check for terminal locations.
        while True:
            delta = 0
            for cell in self.utility_dictionary.items():
                utility = cell[1]
                location = cell[0]
                # This is making the ghost position terminal
                if self.isStateNonTerminal(utility, terminal_utility):
                    action_utilities = self.getActionUtilities(location, self.utility_dictionary)
                    best_utility = self.getMaximumUtility(action_utilities)
                    reward = self.reward_dictionary[location]
                    self.utility_dictionary[location] = self.computeBellmanValue(reward, best_utility)
                    delta = max(delta, abs(self.utility_dictionary[location] - utility))  # Calculating the max change
                else:
                    self.utility_dictionary[location] = utility
            if delta < self.EPSILON * (1 - self.GAMMA) / self.GAMMA:   # Idea: Figure 17.4 page 653 Russel and Norvig
                break

    def getBestMove(self):
        """This method returns the best move. It looks at
        reachable locations, along with their utility and
        returns the best action to perform."""
        reachable_states = [i for i in self.getSurroundingCells(self.pacman_location, 1) if i not in self.wall_locations]
        surrounding_state_utility = [self.utility_dictionary[i] for i in reachable_states]
        best_move_location = reachable_states[surrounding_state_utility.index(max(surrounding_state_utility))]
        best_x, best_y = best_move_location
        x, y = self.pacman_location
        # Getting the best action
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

    def getAction(self, state):
        """This is the main method for making pacman act.
        It computes the value iteration and chooses the best action
        from that result."""
        self.setUpStates(state)
        self.createRewardAndUtilityMap()
        self.computeGhostRewards()
        self.valueIteration()
        best_move = self.getBestMove()
        if Directions.STOP in self.legal_actions:
            self.legal_actions.remove(Directions.STOP)

        return api.makeMove(best_move, self.legal_actions)
