from Utility.Point import Point


class Heuristics:
    @staticmethod
    def euclideanDistance(currentCoordinate: Point, goalCoordinate: Point):

        return ((currentCoordinate.f2 - goalCoordinate.f2)**2 +
                (currentCoordinate.f1 - goalCoordinate.f1)**2)**0.5

    @staticmethod
    def manhattanDistance(currentCoordinate: Point, goalCoordinate: Point):
        return ((abs(currentCoordinate.f1 - goalCoordinate.f1) +
                 abs(currentCoordinate.f2 - goalCoordinate.f2)))

    @staticmethod
    def chebyshevDistance(currentCoordinate: Point, goalCoordinate: Point):
        return max(abs(currentCoordinate.f1 - goalCoordinate.f1),
                   abs(currentCoordinate.f2 - goalCoordinate.f2))


if __name__ == '__main__':
    print(Heuristics.manhattanDistance(Point(0, 0), Point(1, 1)))
