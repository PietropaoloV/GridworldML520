from random import random
from Entity.GridCell import GridCell

from Utility.Point import Point


class Grid:

    grid = []
    xSize = 0
    ySize = 0

    def __init__(self, xSize, ySize, probability) -> None:
        self.xSize = xSize
        self.ySize = ySize
        self.grid = self.generateGrid(xSize, ySize, probability)

    def inBounds(self, x, y):
        return x >= 0 and y >= 0 and x < self.xSize and y < self.ySize

    def generateIsBlocked(self, probabilityOfBlocked):
        return random() * 100 < probabilityOfBlocked

    def generateGrid(self, dimX, dimY, probabilityOfBlocked):
        grid = []

        for y in range(dimY):
            for x in range(dimX):

                numAdj = 0

                for neighbour in Point(x, y).get8Neighbours():
                    if self.inBounds(neighbour.f1, neighbour.f2):
                        numAdj += 1

                isBlocked = self.generateIsBlocked(probabilityOfBlocked)

                grid.append(GridCell(x, y, numAdj, isBlocked, self))

        for y in range(dimY):
            for x in range(dimX):
                index = y * dimX + x

                if not grid[index].isBlocked:
                    continue

                for adj in grid[index].getLocation().get8Neighbours():
                    if not self.inBounds(adj.f1, adj.f2):
                        continue

                    grid[adj.f2 * dimX + adj.f1].addNumSensedBlocked(1)


#
        return grid

    def getCell(self, *args):
        if isinstance(args[0], int):
            if self.inBounds(args[0], args[1]):
                return self.grid[self.xSize * args[1] + args[0]]
        elif isinstance(args[0], Point):
            return self.getCell(args[0].f1, args[0].f2)

    # def getCell(self, coord):
    #     return self.getCell(coord.f1, coord.f2)

    # def getCell(self, x, y):
    #     if self.inBounds(x, y):
    #         return self.grid[self.xSize * y + x]

    def __str__(self) -> str:
        s = 'Entity.Grid{'

        for x in range(self.xSize):
            s += '\n'
            for y in range(self.ySize):
                if x == 0 and y == 0:
                    s += 'S'
                elif x == self.xSize - 1 and y == self.ySize - 1:
                    s += 'G'
                else:
                    if self.getCell(x, y).isBlocked:
                        s += 'X'
                    else:
                        s += ' '
                s += ','

            s += '\n'

        s += '\n}'
        return s

if __name__ == '__main__':
    grid = Grid(10, 10, 30)
    print(grid)