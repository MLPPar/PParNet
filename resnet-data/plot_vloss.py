#!/usr/bin/env python3

import csv
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np

WALL = 'Wall time'
STEP = 'Step'
VALUE = 'Value'

plt.style.use('ggplot')

def key(rule):
    if rule == 'SGDMomentum': return 'momentum'
    elif rule == 'SGDNesterovMomentum': return 'nesterov'
    else: return rule.lower()

def file_name(rule, rate=None):
    if rate is not None:
        return "{}_loss_{}.csv".format(key(rule), rate)
    else:
        return "{}_decay_loss.csv".format(key(rule))

def rate_legend(name):
    rule = key(name)
    val = name.split('_')[-1].split('.')[0]
    return rule + ' ' + val[:1] + '.' + val[1:]

def best_rates():
    return {
        "Adadelta" : "001",
        "Adagrad" : "001",
        "Adam" : "00001",
        "Adamax" : "0001",
        "AMSgrad" : "00001",
        "Eve" : "00001",
        "RMSprop" : "00001",
        "SGD" : "001",
        # "SGDMomentum" : "001",
        "SGDNesterovMomentum" : "001",
    }

def plot_vloss():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    legends = []

    rates = best_rates()
    for rule in rates:
        frame = pd.read_csv(file_name(rule))
        ax.plot(frame[STEP][::5], frame[VALUE][::5])
        legends.append(rule)
    ax.legend(legends)
    ax.set_ylabel('Validation Loss')
    ax.set_xlabel('Training Step')
    plt.show()
    # plt.savefig('resnet_loss.pdf')
    
if __name__ == "__main__":
    plot_vloss()
