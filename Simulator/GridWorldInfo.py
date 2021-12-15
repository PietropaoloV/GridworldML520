class GridWorldInfo:
    probability = 0
    trajectoryLength = 0
    numberOfCellsProcessed = 0
    numBumps = 0
    numPlans = 0
    numberCellsDetermined = 0
    numKnownBumps = 0  # Number of times model predicts bumping into known blocked cell
    numOutOfBounds = 0  # Number of times model predicts going out of bounds
    runtime = 0

    path = []

    def __init__(self, trajectoryLength, numberOfCellsProcessed, path) -> None:
        self.trajectoryLength = trajectoryLength
        self.numberOfCellsProcessed = numberOfCellsProcessed
        self.path = path

    def __str__(self) -> str:
        
        s = 'GridWorldInfo{\n'
        s += f'numBumps = {self.numBumps}\n'
        s += f'numKnownBumps = {self.numKnownBumps} \n'
        s += f'numOutOfBounds = {self.numOutOfBounds} \n'
        s += f'Number of cells processed = {self.numberOfCellsProcessed} \n'
        s += f'Trajectory Length = {self.trajectoryLength}\n'
        s+= '}'
        return s