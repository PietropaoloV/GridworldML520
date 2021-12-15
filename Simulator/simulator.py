from random import randrange
from Entity.Grid import Grid
from Entity.GridWorldInfo import GridWorldInfo
from Utility.Point import Point
from Simulator.model import Model
from numpy import ndarray
import numpy as np


class Simulator():
    grid: Grid = []
    kb: ndarray = None
    model: Model = None
    start: Point = None
    goal: Point = None
    currentPosition : Point = None
    gridWorldInfo : GridWorldInfo = None
    visitedFromPoint :dict = {} # dictionary to keep track of visited cells
    parentDict:dict={} # dictionary to keep track of parent cells

    def __init__(self, gridXSize: int, gridYSize: int, prob: int, start: Point,
                 goal: Point, pathToSavedModel: str) -> None:
        self.grid = Grid(gridXSize, gridYSize, prob)
        self.start = start
        self.goal = goal
        self.model = Model(pathToSavedModel)
        self.initKnowledgeBase()
        self.gridWorldInfo = GridWorldInfo(0,0,None)

    def initKnowledgeBase(self):
        self.kb = np.zeros((self.grid.xSize, self.grid.ySize))
        self.kb[self.start.f1][self.start.f2] = 1
        self.kb[self.goal.f1][self.goal.f2] = 2

    def nextAction(self):
        kerasModel = self.model.model
        predictions = kerasModel.predict(
            np.reshape(self.kb, (-1, self.kb.shape[0], self.kb.shape[1])))[0]
        for prediction in np.argsort(-1 * predictions):
            
            if prediction+1 == 1:
                nextPosition = Point(self.currentPosition.f1 - 1,
                                     self.currentPosition.f2)
            elif prediction+1 == 2:
                nextPosition = Point(self.currentPosition.f1,
                                     self.currentPosition.f2 + 1)
            elif prediction+1 == 3:
                nextPosition = Point(self.currentPosition.f1 + 1,
                                     self.currentPosition.f2)
            elif prediction+1 == 4:
                nextPosition = Point(self.currentPosition.f1,
                                     self.currentPosition.f2 - 1)
            
            if not self.grid.inBounds(nextPosition.f1,nextPosition.f2):
                self.gridWorldInfo.numOutOfBounds += 1
            elif self.kb[nextPosition.f1][nextPosition.f2] == -1:
                self.gridWorldInfo.numKnownBumps += 1
            elif nextPosition not in self.visitedFromPoint[self.currentPosition]:
               return prediction + 1 

    def computeTrajectoryLength(self):
        point = self.goal
        while point != self.start:
            point = self.parentDict[point]
            self.gridWorldInfo.trajectoryLength += 1
            


    def simulate(self):

        self.parentDict = {}

        self.currentPosition = self.start

        self.parentDict[self.currentPosition] = None
        # self.visitedFromPoint[self.currentPosition] = []

        while self.currentPosition != self.goal:
            self.gridWorldInfo.numberOfCellsProcessed += 1
            currentCellPosition = self.grid.getCell(self.currentPosition)
            if self.currentPosition not in self.visitedFromPoint:
                self.visitedFromPoint[self.currentPosition] = []


            if currentCellPosition.isBlocked:
                print(self.currentPosition)
                self.gridWorldInfo.numBumps += 1
                self.kb[self.currentPosition.f1][self.currentPosition.f2] = -1
                self.currentPosition = self.parentDict[self.currentPosition]
                self.kb[self.currentPosition.f1][self.currentPosition.f2] = 1
                self.gridWorldInfo.numberOfCellsProcessed += 1
                continue

            currentNeighbors = self.currentPosition.get4Neighbours()

            for neighbor in currentNeighbors:
                if self.grid.inBounds(neighbor.f1, neighbor.f2):
                    neighborCell = self.grid.getCell(neighbor)
                    if neighborCell.isBlocked:
                        self.kb[neighbor.f1][neighbor.f2] = -1

            action = self.nextAction()

            if action == 1:
                nextPosition = Point(self.currentPosition.f1 - 1,
                                     self.currentPosition.f2)
            elif action == 2:
                nextPosition = Point(self.currentPosition.f1,
                                     self.currentPosition.f2 + 1)
            elif action == 3:
                nextPosition = Point(self.currentPosition.f1 + 1,
                                     self.currentPosition.f2)
            elif action == 4:
                nextPosition = Point(self.currentPosition.f1,
                                     self.currentPosition.f2 - 1)

            
            self.visitedFromPoint[self.currentPosition].append(nextPosition)
            assert len(self.visitedFromPoint[self.currentPosition]) < 5
            self.kb[self.currentPosition.f1][self.currentPosition.f2] = 0
            self.kb[nextPosition.f1][nextPosition.f2] = 1
            self.parentDict[nextPosition] = self.currentPosition
            self.currentPosition = nextPosition
            # print('Normal Cell Position')
            print(self.currentPosition)
            # print(self.gridWorldInfo)
            
        # self.computeTrajectoryLength()



# if __name__ == '__main__':
#     simulator = Simulator(100,100,20,Point(0,0),Point(99,99),'../unPaddedTestModel')
#     print(simulator.grid)
#     simulator.simulate()
#     print(simulator.gridWorldInfo)
    # print(simulator.gridWorldInfo.numBumps)