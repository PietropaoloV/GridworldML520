from Point import Point


class GridCell:

    x = 0
    y = 0
    numAdj = 0
    numSensedBlocked = 0
    numAdjBlocked = 0
    numAdjEmpty = 0
    numAdjHidden = 0
    isBlocked = False
    isVisited = False
    owner = None
    probBlocked = 0

    def __init__(self, x, y, numAdj, isBlocked, owner) -> None:
        self.x = x
        self.y = y
        self.numAdj = numAdj
        self.isBlocked = isBlocked
        self.owner = owner

    def getLocation(self):
        return Point(self.x, self.y)

    def getNumSensedEmpty(self):
        return self.numAdj - self.numSensedBlocked

    def addNumSensedBlocked(self, numSensedBlocked):
        self.numSensedBlocked += numSensedBlocked

    def addNumAdjBlocked(self, numAdjBlocked):
        self.numAdjBlocked += numAdjBlocked
        self.numAdjHidden -= numAdjBlocked
