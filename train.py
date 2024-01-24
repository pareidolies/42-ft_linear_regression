import csv
import sys
import math
import time

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.animation as animation

from alive_progress import alive_bar

from predict import predictPrice, predictPriceNorm
from utils import normalizeElem, denormalizeElem

learningRate = 0.0001
iterations = 500000

### STEP 1: GET DATA FROM CSV

def getData():
    try:
        data = pd.read_csv('data.csv')

        mileages = np.array(data['km'])
        prices = np.array(data['price'])

    except:
        sys.exit('Csv file not found')

    return(mileages, prices)

def getThetas():
    data = pd.read_csv('thetas.csv')

    t0 = np.array(data['t0'])
    t1 = np.array(data['t1'])
    cost = np.array(data['cost'])

    return(t0, t1, cost)

def createCsv():
    with open('thetas.csv', 'w') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvWriter.writerow(['t0', 't1', 'cost'])

### STEP 2: PROCESS FEATURE SCALING

def	normalizeData(mileages, prices):
    x = []
    y = []
    minM = min(mileages)
    maxM = max(mileages)
    for mileage in mileages:
        x.append((mileage - minM) / (maxM - minM))

    minP = min(prices)
    maxP = max(prices)
    for price in prices:
        y.append((price - minP) / (maxP - minP))

    return (x, y)

### STEP 3: TRAIN WITH GRADIENT DESCENT

def	storeThetas(t0, t1, cost, file):
	with open(file, 'a') as csvfile:
		csvWriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		csvWriter.writerow([t0, t1, cost])

def costFunction(t0, t1, x, y):
    m = len(y)
    cost = 0
    thetas = [t0, t1]

    for nbr in range(m):
        cost += (predictPrice(x[nbr], thetas) - y[nbr]) ** 2
    cost /= (2 * m)
    return (cost)

def runGradientDescent(x, y, thetas, learningRate):
    m = len(y)
    newThetas = [0, 0]

    # perform gradient descent for each theta
    for i in range(len(thetas)):

        # compute partial derivative of cost function according to ith theta
        cost_prime = 0
        for nbr in range(m):
            if (i == 0):
                cost_prime += (predictPrice(x[nbr], thetas) - y[nbr])
            else:
                cost_prime += (predictPrice(x[nbr], thetas) - y[nbr]) * x[nbr]
        cost_prime /= m

        # update theta with derivate of error function multiplied by learning rate
        newThetas[i] = thetas[i] - learningRate * cost_prime
    
    cost = costFunction(newThetas[0], newThetas[1], x, y)

    storeThetas(newThetas[0], newThetas[1], cost, 'thetas.csv')

    return newThetas

### STEP 4: PLOT RESULT

def createFigure():
    fig = plt.figure(1)
    ax = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 2, 3, projection="3d")
    plt.subplots_adjust(top=0.9, bottom=0.1)
    return (fig, ax, ax2)

def animatePlots(frames, mileages, prices, t0, t1, cost, ax, ax2):

    ax.clear()

    # csv values
    ax.plot(mileages, prices, "bs") 

    # prediction line
    lineX = [float(min(mileages)), float(max(mileages))]
    lineY = []
    for elem in lineX:
        result = t1[frames * 1000] * normalizeElem(mileages, elem) + t0[frames * 1000]
        lineY.append(denormalizeElem(prices, result)) 

    ax.plot(lineX, lineY, 'r-') # add predicted prices as crosses

    # cost function
    ax2.plot(t0[frames * 1000], t1[frames * 1000], cost[frames * 1000], "b.")

def plotLinearRegressionGrid(mileages, prices, ax):

    ax.set_title("Car price estimation depending on mileage", fontweight='bold')
    ax.set_xlabel('km', fontweight='bold')
    ax.set_ylabel('price', fontweight='bold')
    ax.set_xlim(min(mileages) - 10000, max(mileages) + 10000)
    ax.set_ylim(min(prices) - 1000, max(prices) + 1000)
    ax.grid(True)

def plotCostFunction3d(x, y, cost, ax2):

    ax2.set_title("Cost function", fontweight='bold')
    ax2.set_xlabel('t0', fontweight='bold')
    ax2.set_ylabel('t1', fontweight='bold')
    ax2.set_zlabel('cost', fontweight='bold')
    ax2.set_xlim(-1.0, 1.0)
    ax2.set_ylim(-1.0, 1.0)
    ax2.set_zlim(min(cost), 1.0)

    t0_vals = np.linspace(-1.0, 1.0, 100)
    t1_vals = np.linspace(-1.0, 1.0, 100)
    mesh_t0, mesh_t1 = np.meshgrid(t0_vals, t1_vals)

    cost_vals = costFunction(mesh_t0, mesh_t1, x, y)

    ax2.plot_surface(mesh_t0, mesh_t1, cost_vals, cmap='viridis') # improve dégradé

### MAIN

def main():

    thetas = [0, 0]

    # manage data
    mileages, prices = getData()
    x, y = normalizeData(mileages, prices)
    createCsv()

    # train
    with alive_bar(iterations) as bar:
        for i in range (iterations):
            thetas = runGradientDescent(x, y, thetas, learningRate)
            bar()
    t0, t1, cost = getThetas()

    # graphs
    fig, ax, ax2 = createFigure()
    plotLinearRegressionGrid(mileages, prices, ax)
    plotCostFunction3d(x, y, cost, ax2)

    # animation
    ani = animation.FuncAnimation(fig=fig, func=animatePlots, fargs=(mileages, prices, t0, t1, cost, ax, ax2), frames= int(iterations / 1000), interval=1, blit = True, repeat=False)
    plt.show()

    # clean end

if	__name__ == '__main__':
    main()
