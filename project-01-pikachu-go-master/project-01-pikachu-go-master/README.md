# Project 1 - Team Pikachu Go

## Description
This respository holds the project for Machine Learning at UF (EEL 5840), which had to do with character recognition.

## Team Members
* Kaiyang Han
* Fuyuan Zhang
* Spencer Chang
* Zhewei Song

## Approach
KNN - Easy data set covering letters a and b.

MLP - Hard data set covering letters a, b, c, d, h, i, j, and k.

## How to Run (For TAs and Prof. Zare)
(*Deprecated First Step as of 12/1/2019: it should be possible to train and test code just by following steps 1 and 2)


1. **Training the Models** (`train.py`) - All the models should be trained by the turn-in time of this project. If, however, they are - for some reason - not present or not trained (there exists no `KNNclassifier.pkl` or `MLPclassifier.pkl`), please follow these instructions. No CLI arguments are required.
    - Examples:
        - Only model training (easy and hard): `python train.py`
        - Only model validation (depends on commented/uncommented code): `python train.py --val`
1. **Testing the Models** (`test.py`) - Given that all models have been trained on their respective data sets, this file should pretty much run by itself, when given a data set to run on using its extracted features.
    - CLI Examples: 
        - Easy Data Set: `python test.py KNNclassifier.pkl <path to .pkl data object>`
        - Hard Data Set: `python test.py --hard MLPclassifier.pkl <path to .pkl data object>`
