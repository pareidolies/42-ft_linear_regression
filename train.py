import csv
import sys
import os
import math
import time

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.animation as animation

from colorama import Fore, Back, Style
from alive_progress import alive_bar

from predict import predictPrice, predictPriceNorm
from utils import normalizeElem, denormalizeElem
from utils import getData, getThetas, createCsv

learningRate = 0.1
iterations = 1000

### STEP 1: PROCESS FEATURE SCALING

def	normalizeData(mileages, prices):

    try:
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
    except:
        sys.exit('Csv value error')

    return (x, y)

### STEP 2: TRAIN WITH GRADIENT DESCENT

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

        # update theta by removing derivate of error function multiplied by learning rate (tmpθ)
        newThetas[i] = thetas[i] - learningRate * cost_prime
    
    cost = costFunction(newThetas[0], newThetas[1], x, y)

    storeThetas(newThetas[0], newThetas[1], cost, 'thetas.csv')

    return newThetas

### STEP 3: PLOT RESULT

def createFigure():
    fig = plt.figure("FT_LINEAR_REGRESSION", figsize=(12.8, 9.6))
    ax = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 2, 3, projection="3d")
    ax3 = plt.subplot(2, 2, 4)
    plt.subplots_adjust(top=0.9, bottom=0.1)
    return (fig, ax, ax2, ax3)

def animatePlots(frames, mileages, prices, t0, t1, cost, ax, ax2, ax3):

    ax.clear()
    plotLinearRegressionGrid(mileages, prices, ax)

    # csv values
    ax.plot(mileages, prices, "gs") 

    # prediction line
    lineX = [float(min(mileages)), float(max(mileages))]
    lineY = []
    for elem in lineX:
        result = t1[frames] * normalizeElem(mileages, elem) + t0[frames]
        lineY.append(denormalizeElem(prices, result)) 

    ax.plot(lineX, lineY, 'b-')
    
    # gap between predicted value and actual value
    predicted_values = []
    for elem in mileages:
        result = t1[frames] * normalizeElem(mileages, elem) + t0[frames]
        predicted_values.append(denormalizeElem(prices,result))
    for i in range(len(mileages)):
        ax.plot(mileages[i], predicted_values[i], 'bx')
        ax.plot([mileages[i], mileages[i]], [predicted_values[i], prices[i]], 'k--')

    # cost
    ax2.plot(t0[frames], t1[frames], cost[frames], "k.")

    printParameters(frames, t0, t1, ax3)

def printParameters(frames, t0, t1, ax3):
    ax3.clear()
    ax3.axis('off')

    ax3.text(0, 0.4, 'epochs: ', fontsize = 12, fontweight='bold')
    ax3.text(0.5, 0.4, str(frames + 1), fontsize = 12)

    ax3.text(0, 0.3, 'learning rate: ', fontsize = 12, fontweight='bold')
    ax3.text(0.5, 0.3, str(learningRate), fontsize = 12)

    ax3.text(0, 0.2, 'normalized: ', fontsize = 12, fontweight='bold')
    ax3.text(0.5, 0.2, 'yes', fontsize = 12)

    ax3.text(0, 0.1, 'theta0: ', fontsize = 12, fontweight='bold')
    ax3.text(0.5, 0.1, str(t0[frames]), fontsize = 12)

    ax3.text(0, 0, 'theta1: ' , fontsize = 12, fontweight='bold')
    ax3.text(0.5, 0, str(t1[frames]), fontsize = 12)

    if (frames == iterations - 1):
        print('Animation ' + Fore.GREEN + '[FINISHED]' + Style.RESET_ALL)

def plotLinearRegressionGrid(mileages, prices, ax):

    ax.set_title("Car price estimation depending on mileage", fontweight='bold')
    ax.set_xlabel('mileage (km)', fontweight='bold')
    ax.set_ylabel('price ($)', fontweight='bold')
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

    t0_vals = np.linspace(-0.75, 1.0, 100)
    t1_vals = np.linspace(-1.0, 1.0, 100)
    mesh_t0, mesh_t1 = np.meshgrid(t0_vals, t1_vals)

    cost_vals = costFunction(mesh_t0, mesh_t1, x, y)

    ax2.plot_surface(mesh_t0, mesh_t1, cost_vals, cmap='viridis')

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
    thetas = runGradientDescent(x, y, thetas, learningRate)
    t0, t1, cost = getThetas()

    # prompt
    print('Linear Regression ' + Fore.GREEN + '[READY]' + Style.RESET_ALL)
    time.sleep(0.1)
    print('Cost Function ' + Fore.GREEN + '[READY]' + Style.RESET_ALL)
    time.sleep(0.1)
    print('Graphs ' + Fore.GREEN + '[READY]' + Style.RESET_ALL)
    time.sleep(0.1)
    print('Press Enter to continue' )
    while 1:
        key = input()
        if (key == ''):
            break

    print('Animation ' + Fore.YELLOW + '[PLAYING]' + Style.RESET_ALL)

    # graphs
    fig, ax, ax2, ax3 = createFigure()
    fig.patch.set_facecolor('#FFFFF0')
    ax.set_facecolor('#FFFFF0')
    ax2.set_facecolor('#FFFFF0')
    plotLinearRegressionGrid(mileages, prices, ax)

    # animation

    try:
        ani = animation.FuncAnimation(fig=fig, func=animatePlots, fargs=(mileages, prices, t0, t1, cost, ax, ax2, ax3), frames= iterations, interval=1, blit = True, repeat=False)
        plotCostFunction3d(x, y, cost, ax2)

        plt.show()
    except:
        sys.exit('Csv value error')

    print(Fore.MAGENTA + 'Good Bye!' + Style.RESET_ALL)

if	__name__ == '__main__':
    main()
