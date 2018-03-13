#!/usr/bin/env python3

import csv
import matplotlib.pyplot as plt
import sys

WALL = 'Wall time'
STEP = 'Step'
VALUE = 'Value'

def learning_rates():
    return ("001", "0001", "00001")

def metrics():
    return ("acc", "loss")

def rules():
    return (
        "adadelta", "adagrad", "adam", "adamax", "eve",
        "momentum", "nesterov", "rmsprop", "sgd"
    )

def variants(name, metric):
    return (
        "{}_{}_{}.csv".format(name, metric, rate)
        for rate in learning_rates()
    )

def rate_legend(name):
    rule = name.split('_')[0]
    val = name.split('_')[-1].split('.')[0]
    return rule + ' ' + val[:1] + '.' + val[1:]

def best_rates():
    return {
        "adadelta" : "001",
        "adagrad" : "001",
        "adam" : "00001",
        "adamax" : "0001",
        "eve" : "00001",
        "momentum" : "001",
        "nesterov" : "001",
        "rmsprop" : "00001",
        "sgd" : "001",
    }
    
def plot_rules():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    legends = []

    bests = best_rates()
    for r in sys.argv[1:]:
        name = "{}_loss_{}.csv".format(r, bests[r])
        with open(name) as csvfile:
            read = csv.DictReader(csvfile)
            xs = []
            ys = []
            for row in read:
                xs.append(float(row[STEP]))
                ys.append(float(row[VALUE]))
            ax.plot(xs, ys, label=name)
            legends.append(rate_legend(name))
    ax.legend(legends)
    plt.show()

def plot_rates():
    if len(sys.argv) < 2:
        print("Need a learning rule!")
        exit(1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    legends = []
    for name in variants(sys.argv[1], "loss"):
        with open(name) as csvfile:
            read = csv.DictReader(csvfile)
            xs = []
            ys = []
            for row in read:
                xs.append(float(row[STEP]))
                ys.append(float(row[VALUE]))
            ax.plot(xs, ys, label=name)
            legends.append(rate_legend(name))
    ax.legend(legends)
    plt.show()

if __name__ == "__main__":
    plot_rules()
