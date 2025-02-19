import pandas

class Environment():
    def __init__(self, sizeX, sizeY, perceptDist, startLoc=None):
        self.x = sizeX
        self.y = sizeY
        self.percept = perceptDist
        self.grid = pandas.DataFrame(index=range(self.x), columns=range(self.y))
        self.startLoc = startLoc

    