import csv
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from predict import predictPrice, predictPriceNorm
from utils import normalizeElem, denormalizeElem
import matplotlib.animation as animation

learningRate = 0.0001
iterations = 500000

# STEP 1: GET DATA FROM CSV

def getData():
    # check if file exists
    data = pd.read_csv('data.csv')

    mileages = np.array(data['km'])
    prices = np.array(data['price'])

    return(mileages, prices)

def getThetas():
    data = pd.read_csv('thetas.csv')

    t0 = np.array(data['t0'])
    t1 = np.array(data['t1'])
    cost = np.array(data['cost'])

    return(t0, t1, cost)

# STEP 2: PROCESS FEATURE SCALING

# shall we also withdraw outliers ?

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

# how to denormalize theta ?

# STEP 3: TRAIN WITH GRADIENT DESCENT

def	storeThetas(t0, t1, cost, file):
	with open(file, 'a') as csvfile:
		csvWriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		csvWriter.writerow([t0, t1, cost])

def runGradientDescent(x, y, thetas, learningRate):
    m = len(y)
    newThetas = [0, 0]

    # Perform gradient descent for each theta
    for i in range(len(thetas)):

        # Compute partial derivative of cost function according to ith theta
        costPrime = 0
        for nbr in range(m):
            if (i == 0):
                costPrime += (predictPrice(x[nbr], thetas) - y[nbr])
            else:
                costPrime += (predictPrice(x[nbr], thetas) - y[nbr]) * x[nbr]
        costPrime /= m

        # Update theta with derivate of error function multiplied by learning rate
        newThetas[i] = thetas[i] - learningRate * costPrime
    
    # BONUS : compute cost
    cost = 0
    for nbr in range(m):
        cost += (predictPrice(x[nbr], newThetas) - y[nbr]) ** 2
    cost /= (2 * m)

    storeThetas(newThetas[0], newThetas[1], cost, 'thetas.csv')

    return newThetas

def costFunction(t0, t1, x, y):
    m = len(y)
    cost = 0
    thetas = [t0, t1]

    for nbr in range(m):
        cost += (predictPrice(x[nbr], thetas) - y[nbr]) ** 2
    cost /= (2 * m)
    return (cost)

# STEP 4: PLOT RESULT

def plotLinearRegression(frames, mileages, prices, t0, t1, cost, ax, ax2):

    ax.clear()

    ax.set_title("Car price estimation depending on mileage", fontweight='bold')
    ax.set_xlabel('km', fontweight='bold')
    ax.set_ylabel('price', fontweight='bold')
    ax.set_xlim(min(mileages) - 10000, max(mileages) + 10000)
    ax.set_ylim(min(prices) - 1000, max(prices) + 1000)
    ax.grid(True)

    ax.plot(mileages, prices, "bs")

    lineX = [float(min(mileages)), float(max(mileages))]
    lineY = []
    for elem in lineX:
        result = t1[frames * 1000] * normalizeElem(mileages, elem) + t0[frames * 1000]
        lineY.append(denormalizeElem(prices, result)) 

    ax.plot(lineX, lineY, 'r-')
    # add predicted prices as crosses

    # BONUS : plot cost function (don't clear, add last point in red, draw overall shape)
    ax2.set_title("Cost function", fontweight='bold')
    ax2.set_xlabel('t0', fontweight='bold')
    ax2.set_ylabel('t1', fontweight='bold')
    ax2.set_zlabel('cost', fontweight='bold')
    ax2.set_xlim(-1.0, 1.0)
    ax2.set_ylim(-1.0, 1.0)
    ax2.set_zlim(min(cost), 1.0)

    ax2.plot(t0[frames * 1000], t1[frames * 1000], cost[frames * 1000], "b.")

# MAIN

def main():
    thetas = [0, 0]

    mileages, prices = getData()
    x, y = normalizeData(mileages, prices)

    # prepare thetas.csv
    with open('thetas.csv', 'w') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvWriter.writerow(['t0', 't1', 'cost'])

    # train
    for i in range (iterations):
        thetas = runGradientDescent(x, y, thetas, learningRate)

    t0, t1, cost = getThetas()

    # graphs
    fig = plt.figure(1)
    ax = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 2, 3, projection="3d")
    plt.subplots_adjust(top=0.9, bottom=0.1) 

    # PLAY BUTTONS

    # ax_button_1 = plt.axes([0.14, 0.02, 0.15, 0.04], facecolor='black')

    # button_1 = Button(ax_button_1, '1', color='white', hovercolor='cornflowerblue')

    # def on_button_click(it):
    #     iterations = it

    # button_1.on_clicked(1)


    # 3d cost function

    t0_vals = np.linspace(-1.0, 1.0, 100)
    t1_vals = np.linspace(-1.0, 1.0, 100)
    mesh_t0, mesh_t1 = np.meshgrid(t0_vals, t1_vals)

    cost_vals = costFunction(mesh_t0, mesh_t1, x, y)

    ax2.plot_surface(mesh_t0, mesh_t1, cost_vals, cmap='viridis')

    # animation
    ani = animation.FuncAnimation(fig=fig, func=plotLinearRegression, fargs=(mileages, prices, t0, t1, cost, ax, ax2), frames= int(iterations / 1000), interval=2, blit = True, repeat=False)
    
    plt.show()

    # clean end

if	__name__ == '__main__':
    main()
