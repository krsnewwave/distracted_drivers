from lasagne.layers import get_all_params
from lasagne.utils import floatX
import theano as T
import numpy as np

import matplotlib.pyplot as plt

__author__ = 'dylan'

IMAGE_SIZE = 256


class AdjustVariableStepDecreasing(object):
    """This class adjusts any variable during training
    """

    def __init__(self, name, start=0.03, step=20000):
        self.name = name
        self.start = start
        self.step = step
        self.times_reduced = 0

    def __call__(self, nn, train_history):
        try:
            # when step is reached, divide by (10^ (times_reduced+1))
            epoch = train_history[-1]['epoch']
            if epoch > (self.step * (self.times_reduced + 1)):
                self.times_reduced += 1
                new_value = np.float32(self.start / (10 ** (self.times_reduced + 1)))
                getattr(nn, self.name).set_value(new_value)
        except IndexError:
            pass
            # print "Required index {} out of {}".format(train_history[-1]['epoch'], len(self.ls))


class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None
        self.done_training = False

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            self.done_training = True
            raise StopIteration()

class StoreBest(object):
    def __init__(self):
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()

# class EndTraining(Exception):
class EndTrainingFromEarlyStopping(object):
    def __call__(self, nn, train_history):
        # find early stopping here
        for variable in nn.on_epoch_finished:
            if isinstance(variable, EarlyStopping):
                # check if done_training
                if variable.done_training:
                    raise StopIteration()
