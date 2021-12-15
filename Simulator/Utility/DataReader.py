import numpy as np


class DataReader:
    knownBlockedCellValue = 0
    paddingValue = 0
    xTrain = None
    yTrain = None
    xTest = None
    yTest = None
    isPadded = False
    changeBlockedCellValue = False

    def __init__(self, knownBlockedCellValue, paddingValue, isPadded,
                 changeBlockedCellValue):
        self.knownBlockedCellValue = knownBlockedCellValue
        self.paddingValue = paddingValue
        self.isPadded = isPadded
        self.changeBlockedCellValue = changeBlockedCellValue

    def getXTrainMat(self, pathToFile):
        self.xTrain = np.load(pathToFile)

        if self.changeBlockedCellValue:
            self.xTrain[self.xTrain == -1] = self.knownBlockedCellValue

        if self.isPadded:
            self.xTrain = np.pad(self.xTrain, ((0, 0), (1, 1), (1, 1)),
                                 'constant',
                                 constant_values=(self.paddingValue))
        return self.xTrain

    def getYTrainMat(self, pathToFile):
        self.yTrain = np.load(pathToFile)
        return self.yTrain

    def getXTestMat(self, pathToFile):
        self.xTest = np.load(pathToFile)

        if self.changeBlockedCellValue:
            self.xTest[self.xTest == -1] = self.knownBlockedCellValue

        if self.isPadded:
            self.xTest = np.pad(self.xTest, ((0, 0), (1, 1), (1, 1)),
                                'constant',
                                constant_values=(self.paddingValue))
        return self.xTest

    def getYTestMat(self, pathToFile):
        self.yTest = np.load(pathToFile)
        return self.yTest


if __name__ == '__main__':
    dataReaderObj = DataReader(-100, -100, True, True)
    print(dataReaderObj.getXTestMat('xTest.npy'))
    print(dataReaderObj.getYTestMat('yTest.npy'))