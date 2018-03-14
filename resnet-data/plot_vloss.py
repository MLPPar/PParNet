#!/usr/bin/env python3

import csv
import matplotlib.pyplot as plt
import pandas as pd
import sys

WALL = 'Wall time'
STEP = 'Step'
VALUE = 'Value'

plt.style.use('ggplot')

def file_name(rule, rate=None):
    if rate is not None:
        return "{}_loss_{}.csv".format(rule, rate)
    else:
        return "{}_vloss.csv".format(rule)

def rate_legend(name):
    rule = name.split('_')[0]
    val = name.split('_')[-1].split('.')[0]
    return rule + ' ' + val[:1] + '.' + val[1:]

def best_rates():
    return {
        # "adadelta" : "001",
        "adagrad" : "001",
        "adam" : "00001",
        # "adamax" : "0001",
        "eve" : "00001",
        # "momentum" : "001",
        # "nesterov" : "001",
        "rmsprop" : "00001",
        "sgd" : "001",
    }

def plot_vloss():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    legends = []

    rates = best_rates()
    for rule in rates:
        frame = pd.read_csv(file_name(rule))
        ax.plot(frame[STEP], frame[VALUE])
        legends.append(rule)
    ax.legend(legends)
    ax.set_ylabel('Validation Loss')
    ax.set_xlabel('Training Step')
    plt.savefig('resnet_loss.png')
    
if __name__ == "__main__":
    plot_vloss()
