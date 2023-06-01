import pickle
import numpy
import seaborn as sns
import matplotlib.pyplot as plt
def save(nfile, object):
    with open(nfile, 'wb') as file:
        pickle.dump(object, file)


def load(nfile):
    with open(nfile, 'rb') as file:
        return pickle.load(file)
