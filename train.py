import csv
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from predict import predictPrice
import matplotlib.animation as animation

learningRate = 0.0001
iterations = 1000000

# STEP 1: GET DATA FROM CSV

def getData():
    data = pd.read_csv('data.csv')

    mileages = np.array(data['km'])
    prices = np.array(data['price'])

    return(mileages, prices)

def getThetas():
    data = pd.read_csv('thetas.csv')

    t0 = np.array(data['t0'])
    t1 = np.array(data['t1'])

    return(t0, t1)

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

def	normalizeElem(list, elem):
    return ((elem - min(list)) / (max(list) - min(list)))

def	denormalizeElem(list, elem):
    return ((elem * (max(list) - min(list))) + min(list))

#def denormalizeTheta()

# STEP 3: TRAIN WITH GRADIENT DESCENT

def	storeThetas(t0, t1, file):
	with open(file, 'a') as csvfile:
		csvWriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		csvWriter.writerow([t0, t1])

def runGradientDescent(x, y, thetas, learningRate):
    m = len(y)
    newThetas = [0, 0]

    # Perform gradient descent for each theta
    for i in range(len(thetas)):

        # Compute partial derivative of cost function according to ith theta
        cost = 0
        for nbr in range(m):
            if (i == 0):
                cost += (predictPrice(x[nbr], thetas) - y[nbr])
            else:
                cost += (predictPrice(x[nbr], thetas) - y[nbr]) * x[nbr]
        cost /= m

        # Update theta with derivate of error function multiplied by learning rate
        newThetas[i] = thetas[i] - learningRate * cost
    
    storeThetas(newThetas[0], newThetas[1], 'thetas.csv')

    return newThetas

# STEP 4: PLOT RESULT

def plotLinearRegression(frames, mileages, prices, t0, t1, ax):

    ax.clear()

    ax.set_title("Car price estimation depending on mileage", fontweight='bold')
    ax.set_xlabel('km', fontweight='bold')
    ax.set_ylabel('price', fontweight='bold')
    ax.grid(True)

    ax.plot(mileages, prices, "bs")

    lineX = [float(min(mileages)), float(max(mileages))]
    lineY = []
    for elem in lineX:
        result = t1[frames * 1000] * normalizeElem(mileages, elem) + t0[frames * 1000]
        lineY.append(denormalizeElem(prices, result)) 

    ax.plot(lineX, lineY, 'r-')
    # add predicted prices as crossed

# def plotCostFunction():

# STEP 5: ANALYZE ACCURACY OF MODEL

# r2 / mse / mae

# MAIN

def main():
    thetas = [0, 0]

    mileages, prices = getData()
    x, y = normalizeData(mileages, prices)

    # prepare thetas.csv
    with open('thetas.csv', 'w') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvWriter.writerow(['t0', 't1'])

    # train
    for i in range (iterations):
        thetas = runGradientDescent(x, y, thetas, learningRate)

    t0, t1 = getThetas()

    # graph
    fig, ax = plt.subplots()
    plt.subplots_adjust(top=0.9, bottom=0.2) 

    # # BUTTONS

    # ax_button_1 = plt.axes([0.14, 0.02, 0.15, 0.04], facecolor='black')
    # ax_button_10 = plt.axes([0.34, 0.02, 0.15, 0.04], facecolor='black')
    # ax_button_1000 = plt.axes([0.54, 0.02, 0.15, 0.04], facecolor='black')
    # ax_button_1000000 = plt.axes([0.74, 0.02, 0.15, 0.04], facecolor='black')

    # button_1 = Button(ax_button_1, '1', color='white', hovercolor='cornflowerblue')
    # button_10 = Button(ax_button_10, '10', color='white', hovercolor='cornflowerblue')
    # button_1000 = Button(ax_button_1000, '1000', color='white', hovercolor='cornflowerblue')
    # button_1000000 = Button(ax_button_1000000, '1 000 000', color='white', hovercolor='cornflowerblue')

    # def on_button_click(it):
    #     iterations = it
    #     #rerun gradient descent

    # button_1.on_clicked(1)
    # button_10.on_clicked(10)
    # button_1000.on_clicked(1000)
    # button_1000000.on_clicked(1000000)

    # play button

    # animation
    ani = animation.FuncAnimation(fig=fig, func=plotLinearRegression, fargs=(mileages, prices, t0, t1, ax), frames=1000, interval=2, repeat=False)
    plt.show()

    # plotLinearRegression(thetas, mileages, prices)

if	__name__ == '__main__':
    main()
