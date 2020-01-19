'''
Abstract class for creating individual classifiers. I hope it suffices with these four.

Original: Spencer Chang

Modifications:

'''
class AbstractClassifier(object):

    def __init__(self, param_list):
        '''
        Initialize weights and whatever needed variables in this method.
        @param param_list: parameters necessary for classifier
        '''
        raise NotImplementedError


    def train(self, x, y, epochs=10, batch_size=16, lr_rate=0.001, val_x=None, val_y=None):
        '''
        Trains the classifier using input data and labels, if supervised is desired.
        @param x: Input Data
        @param y: Labels corresponding to the input data
        @param epochs: Number of epochs to run training
        @param batch_size: Batch sizes for iterations w/in an epoch
        @param lr_rate: Learning rate. May or may not be used in this function
        @returns: list containing final training accuracy and error
        '''
        raise NotImplementedError


    def predict(self, x) -> list:
        '''
        Predict the results of the classifier on a given set of data
        @param x: Input Data to be classified
        @returns: list of predicted labels (this corresponds to given input data)
        '''
        raise NotImplementedError


    def evaluate(self, x, y):
        '''
        Runs classifier on input data and outputs statistics such as accuracy and error
        @param x: Input data
        @param y: Labels corresponding to input data
        @returns: list containing evaluated accuracy and error
        '''
        raise NotImplementedError