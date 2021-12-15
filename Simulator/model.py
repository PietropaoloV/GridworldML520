import os

from Simulator.Entity.Grid import Grid
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import numpy as np
from tensorflow.python.framework.tensor_util import constant_value
from tensorflow.keras.models import load_model


class Model():

    model = None

    def __init__(self,pathToModelDir) -> None:
        try:
            self.model = load_model(pathToModelDir)
        except:
            print("Incorrect path given !!")
    

# if __name__ == '__main__':
#     model = Model('../unPaddedTestModel')
#     grid = Grid(100,100,10)
#     kb : np.ndarray = np.array([[0]*100 for _ in range(100)])
#     kb[0][0] = 1
#     kb[99][99] = 2
#     kb = np.reshape(kb,(-1,100,100))
#     # kb = kb.flatten('A')
#     # print(kb.shape)
#     # kbPadded = np.pad(kb,(1,1),'constant',constant_values=(-1)) 
#     # print(type(model.model))
#     print(model.model.predict(kb))

