# import pikachu_lib
import AbstractClassifier
import numpy as np

class ChangClassifier(AbstractClassifier.AbstractClassifier):

    def __init__(self, param_list, name='Classifier'):
        self.weights = param_list
        self.name = name

    def train(self, x, y, epochs=10):
        return []

    def predict(self, x) -> list:
        return []


    def evaluate(self, x, y):
        return []
