

import keras
import numpy as np

class TrainingPlot(keras.callbacks.Callback):

    def __init__(self):        
        
    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        pass

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        pass
