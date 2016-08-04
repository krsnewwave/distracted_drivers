from nolearn.lasagne import NeuralNet
from lasagne.layers import DenseLayer, ReshapeLayer, Upscale2DLayer, Conv2DLayer, InputLayer, DropoutLayer, \
    MaxPool2DLayer, get_all_params, batch_norm
import numpy as np
from lasagne.nonlinearities import softmax, leaky_rectify
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator, PrintLayerInfo, objective
from nolearn.lasagne import TrainSplit
from common import EarlyStopping, EndTrainingFromEarlyStopping
from lasagne.objectives import categorical_crossentropy, aggregate
import cPickle as pickle
from sklearn import metrics
import time, logging, logging.config, logging.handlers
from skimage import io, transform, exposure, color, util
import os, itertools, sys
from PIL import Image
from math import ceil, floor
import matplotlib.pyplot as plt
sys.setrecursionlimit(1000000)

input_volume_shape = (128, 128)


class AdjustVariableWithStepSize(object):
    """This class adjusts any variable during training
    """

    def __init__(self, name, start=0.03, steps=3, after_epochs=2000):
        self.name = name
        self.start = start
        self.steps=steps
        self.after_epochs=after_epochs
        self.ls = []

    def __call__(self, nn, train_history):
        if not self.ls:
            for i in range(self.steps):
                self.ls.extend(np.repeat(self.start/(np.power(10,i)), self.after_epochs))

        try:
            epoch = train_history[-1]['epoch']
            new_value = np.float32(self.ls[epoch - 1])
            getattr(nn, self.name).set_value(new_value)
        except IndexError:
            pass
        

def load_best_weights(bw_file, net):
    with open(bw_file, 'rb') as reader:
        best_weights = pickle.load(reader)
    net.load_params_from(best_weights)
    

def read_img_file_PIL(file_path, size=(32,32)):
    img = Image.open(file_path).convert('L')
    img.thumbnail(size, Image.NEAREST)
    data = np.array(img)
    shape = data.shape
    append_top = int(ceil(max(0, size[0] - shape[0])/2.0))
    append_bot = int(floor(max(0, size[0] - shape[0])/2.0))
    data = util.pad(data, ((append_top, append_bot),
                           (0,0)), mode='constant', constant_values=0)
    return data

def read_img_file(file_path, rescale=0.01):
    img = io.imread(file_path)
    img= color.rgb2gray(img)
    return transform.rescale(img, rescale)

def image_gen_from_dir_with_filenames(directory, batch_size, num_categories, size=input_volume_shape):
    result = {os.path.join(dp, f) : int(os.path.split(dp)[1]) for dp, dn, filenames in os.walk(data_dir) 
                  for f in filenames if os.path.splitext(f)[1] == '.jpg'}
    # infinite loop
    while True:
        image_files = []
        labels = []
        filenames = []
        # randomly choose batch size samples in result
        for category in range(num_categories):
            file_samples = np.random.choice([k for k, v in result.iteritems() if v == category], 
                             size=batch_size, replace=False)
            for file_sample in file_samples:
                image_files.append(read_img_file_PIL(file_sample, size=size))
            labels.extend([v for v in itertools.repeat(category, batch_size)])

        # end category loop
        X = np.asarray(image_files, dtype=np.float32)
        # -1 to 1 range
        X = exposure.rescale_intensity(X, out_range=(-1,1))
        y = np.asarray(labels, dtype=np.int32)
        yield X, y

def image_gen_from_dir(directory, batch_size, num_categories, size=input_volume_shape):
    result = {os.path.join(dp, f) : int(os.path.split(dp)[1]) for dp, dn, filenames in os.walk(data_dir) 
                  for f in filenames if os.path.splitext(f)[1] == '.jpg'}
    # infinite loop
    while True:
        image_files = []
        labels = []
        # randomly choose batch size samples in result
        for category in range(num_categories):
            file_samples = np.random.choice([k for k, v in result.iteritems() if v == category], 
                             size=batch_size, replace=False)
            for file_sample in file_samples:
                image_files.append(read_img_file_PIL(file_sample, size=size))
            labels.extend([v for v in itertools.repeat(category, batch_size)])

        # end category loop
        X = np.asarray(image_files, dtype=np.float32)
        # -1 to 1 range
        X = exposure.rescale_intensity(X, out_range=(-1,1))
        y = np.asarray(labels, dtype=np.int32)
        yield X, y
        
def threaded_generator(generator, num_cached=50):
    import Queue
    queue = Queue.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    # define producer (putting items into queue)
    def producer():
        for item in generator:
            queue.put(item)
        queue.put(sentinel)

    # start producer (in a background thread)
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = queue.get()
    while item is not sentinel:
        yield item
        queue.task_done()
        item = queue.get()
        
def load_best_weights(bw_file, net):
    with open(bw_file, 'rb') as reader:
        best_weights = pickle.load(reader)
    net.load_params_from(best_weights)
    
def get_validation_loss(vLoss_file, epoch_number=-1):
    losses = []
    accs = []
    with open(vLoss_file) as reader:
        for line in reader:
            elements = line.split(",")
            epoch = int(elements[0].strip())
            loss = float(elements[1].strip())
            if len(elements) > 2:
                acc = float(elements[2].strip())
                accs.append((epoch, acc))
            losses.append((epoch, loss))
            if epoch == epoch_number:
                break
    return accs, losses
    
def plot_validation_loss(net, vLoss_file, ylim=[1, 5]):
    accs, vLoss = get_validation_loss(vLoss_file)

    train_loss = [row['train_loss'] for row in net.train_history_]
    epochs = [row['epoch'] for row in net.train_history_]
    valid_loss = [row['valid_loss'] for row in net.train_history_]

    fig = plt.figure(figsize=(16, 16))
    plt.ylim(ylim)
    plt.xlabel('epoch')
    plt.ylabel('loss')

    ax1 = fig.add_subplot(111)
    ax1.plot(epochs, train_loss, label='train loss')
    ax1.plot(epochs, valid_loss, label='bootstrap loss')
    ax1.plot(*zip(*vLoss), label="validation loss", c='g')
    if len(accs) > 0:
        ax2 = ax1.twinx()
        ax2.plot(*zip(*accs), label="Accuracy", c='y')

    plt.legend(loc='best')
    plt.show()

        
def regularization_objective(layers, lambda1=0., lambda2=0., *args, **kwargs):
    # default loss
    losses = objective(layers, *args, **kwargs)
    # get layer weights except for the biases
    weights = get_all_params(layers[-1], regularizable=True)
    regularization_term = 0.0
    # sum of abs weights for L1 regularization
    if lambda1 != 0.0:
        sum_abs_weights = sum([abs(w).sum() for w in weights])
        regularization_term += (lambda1 * sum_abs_weights) 
    # sum of squares (sum(theta^2))
    if lambda2 != 0.0:
        sum_squared_weights = (1 / 2.0) * sum([(w ** 2).sum() for w in weights])
        regularization_term += (lambda2 * sum_squared_weights)
    # add weights to regular loss
    losses += regularization_term
    return losses

def vis_square(d, padsize=1, padval=0, size=(20, 20)):
    data = np.copy(d)
    data = (data - data.min()) / (data.max() - data.min())

    print("Original layer shape: {}".format(data.shape))
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    print("Last shape: {}".format(data.shape))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:]).squeeze()
    print("Visualization shape {}:".format(data.shape))
    plt.figure(figsize=size)
    plt.imshow(data, cmap='gray', interpolation='nearest')
    return data
