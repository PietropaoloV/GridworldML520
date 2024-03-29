import os
from datetime import datetime
from random import randint
from typing import List
import numpy as np
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import os


class Simulator():
    grids: np.ndarray = None
    grid: np.ndarray = None
    kb: np.ndarray = None
    model = None
    rowSize: int = 0
    colSize: int = 0
    currentPosition: tuple = (0, 0)
    deadNodes: set = None  # Nodes adjoining blocked cells or other dead nodes
    parent: dict = None
    visitedFromPoint: dict = None
    stats: dict = {}
    trajectoryLengths: int = 0
    knownOutOfBounds: int = 0
    knownBumps: int = 0
    pathToSaveStats: str = None

    def __init__(
        self,
        pathToGrids,
        pathToModel,
    ) -> None:
        self.grids = np.load(pathToGrids)
        self.model = load_model(pathToModel)
        self.rowSize = len(self.grids[0])
        self.colSize = len(self.grids[0][0])
        self.stats['trajectoryLengths'] = np.array([])
        self.stats['knownBumps'] = np.array([])
        self.stats['knownOutOfBounds'] = np.array([])
        self.stats['discardedGrids'] = np.array([])

    def generateWindow(self, windowSize):
        window = []
        for rowCoord in range(self.currentPosition[0] - windowSize // 2,
                              self.currentPosition[0] + windowSize // 2 + 1):
            windowRow = []
            for colCoord in range(
                    self.currentPosition[1] - windowSize // 2,
                    self.currentPosition[1] + windowSize // 2 + 1):
                if rowCoord < 0 or colCoord < 0 or rowCoord >= self.rowSize or colCoord >= self.colSize:
                    windowRow.append(-1)
                else:
                    windowRow.append(self.kb[rowCoord][colCoord])
            window.append(windowRow)

        return np.array(window)

    def getModel(self, pathToModelFile):
        return load_model(pathToModelFile)

    def inBounds(self, row, col):
        return row < self.rowSize and row >= 0 and col < self.colSize and col >= 0

    def getFourNeighbors(self, row, col):
        neighbors = []
        neighbors.append((row + 1, col))
        neighbors.append((row, col + 1))
        neighbors.append((row - 1, col))
        neighbors.append((row, col - 1))
        return neighbors

    def getEightNeighbors(self, row, col):
        neighbors = self.getFourNeighbors(row, col)
        neighbors.append((row + 1, col + 1))
        neighbors.append((row - 1, col + 1))
        neighbors.append((row - 1, col - 1))
        neighbors.append((row + 1, col - 1))
        return neighbors

    def predict(self, window):
        predictions = self.model.predict(np.reshape(window, (1, 11, 11)))[0]
        # print(predictions)
        nextPosition = None
        iterationCount = 0

        for prediction in np.argsort(-1 * predictions):
            if prediction == 0:
                nextPosition = (self.currentPosition[0] - 1,
                                self.currentPosition[1])
            elif prediction == 1:
                nextPosition = (self.currentPosition[0],
                                self.currentPosition[1] + 1)
            elif prediction == 2:
                nextPosition = (self.currentPosition[0] + 1,
                                self.currentPosition[1])
            else:
                nextPosition = (self.currentPosition[0],
                                self.currentPosition[1] - 1)

            if not self.inBounds(nextPosition[0], nextPosition[1]):
                if iterationCount == 0:
                    self.knownOutOfBounds += 1
                iterationCount += 1
                continue
            elif self.kb[nextPosition[0]][nextPosition[1]] == -1:
                if iterationCount == 0:
                    self.knownBumps += 1
                iterationCount += 1
                continue
            elif nextPosition in self.deadNodes:
                iterationCount += 1
                continue
            elif nextPosition == self.parent[
                    self.
                    currentPosition]:  # try to nudge the model to not go back unless absolutely necessary
                iterationCount += 1
                continue
            elif nextPosition in self.visitedFromPoint[self.currentPosition]:
                iterationCount += 1
                continue
            else:
                break

        if iterationCount == 4:
            return None
        else:
            return nextPosition

    def NRuns(self, n, saveStats=False, filePath=''):

        for i in range(n):
            self.OneRunOnRandomGrid(i)
            print('Grid No. : ', i + 1)
            if saveStats:
                self.saveStats(filePath)

    def saveStats(self, filePath):
        np.save(
            os.path.join(filePath, str(datetime.now())) + '_' +
            str(len(self.stats['trajectoryLengths'])) + 'Runs', self.stats)

    def OneRunOnRandomGrid(self, index: int = -1):
        if index == -1:
            index = randint(0, len(self.grids) - 1)

        self.grid = self.grids[index]

        start = (0, 0)
        goal = (self.rowSize - 1, self.colSize - 1)

        self.kb = np.array([[0] * self.colSize for _ in range(self.rowSize)])

        self.kb[start[0]][start[1]] = 2
        self.kb[goal[0]][goal[1]] = 3

        self.currentPosition = start
        self.deadNodes = set()
        self.parent = dict()
        self.parent[start] = None
        self.visitedFromPoint = {}
        self.trajectoryLength = 0
        self.knownBumps = 0
        self.knownOutOfBounds = 0
        discardGrid = False

        while self.currentPosition != goal:

            self.trajectoryLength += 1

            if self.currentPosition not in self.visitedFromPoint:
                self.visitedFromPoint[self.currentPosition] = []

            self.kb[self.currentPosition[0]][self.currentPosition[1]] = 2
            neighbors = self.getFourNeighbors(self.currentPosition[0],
                                              self.currentPosition[1])
            for neighbor in neighbors:

                if not self.inBounds(neighbor[0], neighbor[1]):
                    continue

                if self.grid[neighbor[0]][neighbor[1]] == 0:
                    self.kb[neighbor[0]][neighbor[1]] = 0
                elif self.grid[neighbor[0]][neighbor[1]] == 1:
                    self.kb[neighbor[0]][neighbor[1]] = -1

            window = self.generateWindow(11)

            nextPosition = self.predict(window)

            if nextPosition is None:
                self.deadNodes.add(self.currentPosition)
                nextPosition = self.parent[self.currentPosition]

            self.visitedFromPoint[self.currentPosition].append(nextPosition)
            if len(self.visitedFromPoint[self.currentPosition]) > 4:
                discardGrid = True
                break
            # assert len(self.visitedFromPoint[self.currentPosition]) < 5
            self.parent[nextPosition] = self.currentPosition
            self.kb[self.currentPosition[0]][self.currentPosition[1]] = 0
            self.kb[nextPosition[0]][nextPosition[1]] = 2
            self.currentPosition = nextPosition
            # print(self.currentPosition)

        if discardGrid:
            self.stats['discardedGrids'] = np.append(
                self.stats['discardedGrids'], index)

        self.stats['trajectoryLengths'] = np.append(
            self.stats['trajectoryLengths'], self.trajectoryLength)
        self.stats['knownBumps'] = np.append(self.stats['knownBumps'],
                                             self.knownBumps)
        self.stats['knownOutOfBounds'] = np.append(
            self.stats['knownOutOfBounds'], self.knownOutOfBounds)


if __name__ == '__main__':

    pathToGridFile = os.path.join(os.path.dirname(__file__),
                                  '../grids/project1-grids.npy')

    # print(grids.shape)

    modelFilePath = os.path.join(os.path.dirname(__file__), '../CNNModel')

    simulator = Simulator(pathToGridFile, modelFilePath)
    # simulator.OneRunOnRandomGrid()
    simulator.NRuns(1000, True,
                    os.path.join(os.path.dirname(__file__), 'statsDump'))
    # simulator.saveStats(os.path.dirname(__file__))
    # print(simulator.trajectoryLengths)
    # print(simulator.deadNodes)