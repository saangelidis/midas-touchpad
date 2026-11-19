import pickle
import os

class charMap:
    def __init__(self, name, directory=f"{name}.pkl")
        self.__name = name
        self.__dir = directory 
        self.__dict = {}
        self.loadMap()
        

    def loadMap(self):
        if self.__dir != "":
            with open(self.__dir,'rb') as m:
                self.__dict = pickle.load(m)

    def save(self):
        with open(self.__dir, 'wb') as m:
            pickle.dump(self.__dict,m)
        
    def changeDict(self, newDict):
        self.__dict = newDict
        self.save()


def main():

    characterGroupings = ["math", "greek"]
    
    for cg in characterGroupings:
        if not os.path.exists(f"{cg}.pkl"):

