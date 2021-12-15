class Point:

    f1 = 0
    f2 = 0

    def __init__(self, f1, f2) -> None:
        self.f1 = f1
        self.f2 = f2

    def get4Neighbours(self):
        neighbours = []
        neighbours.append(Point(self.f1 + 1, self.f2))
        neighbours.append(Point(self.f1 - 1, self.f2))
        neighbours.append(Point(self.f1, self.f2 - 1))
        neighbours.append(Point(self.f1, self.f2 + 1))

        return neighbours

    def get8Neighbours(self):
        neighbours = []
        neighbours.append(Point(self.f1 + 1, self.f2))
        neighbours.append(Point(self.f1 + 1, self.f2 + 1))
        neighbours.append(Point(self.f1, self.f2 + 1))
        neighbours.append(Point(self.f1 - 1, self.f2 + 1))
        neighbours.append(Point(self.f1 - 1, self.f2))
        neighbours.append(Point(self.f1 - 1, self.f2 - 1))
        neighbours.append(Point(self.f1, self.f2 - 1))
        neighbours.append(Point(self.f1 + 1, self.f2 - 1))
        return neighbours

    def __str__(self) -> str:
        return '<' + str(self.f1) + ',' + str(self.f2) + '>'

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o,Point):
            return False
        return self.f1 == __o.f1 and self.f2 == __o.f2
    
    def __hash__(self) -> int:
        return hash((self.f1,self.f2))


if __name__ == '__main__':
    a = Point(1, 1)
    for neighbour in a.get4Neighbours():
        print(neighbour)
    print('**********')
    for neighbour in a.get8Neighbours():
        print(neighbour)