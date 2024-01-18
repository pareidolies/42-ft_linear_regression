import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from predict import predictPrice

iterations = 100

# STEP 1: GET DATA FROM CSV

def getData():
    data = pd.read_csv('data.csv')

    mileages = np.array(data['km'])
    prices = np.array(data['price'])

    return(mileages, prices)

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

# STEP 3: TRAIN WITH GRADIENT DESCENT

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
    
    return newThetas

# STEP 4: PLOT RESULT

def plotLinearRegression(thetas, mileages, prices):
    fig, ax = plt.subplots()

    plt.subplots_adjust(top=0.9, bottom=0.2) 

    ax.set_title("Car price estimation depending on mileage", fontweight='bold')
    ax.set_xlabel('km', fontweight='bold')
    ax.set_ylabel('price', fontweight='bold')
    ax.grid(True)

    lineX = [float(min(mileages)), float(max(mileages))]
    lineY = []
    for elem in lineX:
        result = thetas[1] * normalizeElem(mileages, elem) + thetas[0]
        lineY.append(denormalizeElem(prices, result)) 

    ax.plot(mileages, prices, "bs")
    ax.plot(lineX, lineY, 'r-')
    # add predicted prices as crossed

    # BUTTONS

    ax_button_1 = plt.axes([0.14, 0.02, 0.15, 0.04], facecolor='black')
    ax_button_10 = plt.axes([0.34, 0.02, 0.15, 0.04], facecolor='black')
    ax_button_1000 = plt.axes([0.54, 0.02, 0.15, 0.04], facecolor='black')
    ax_button_1000000 = plt.axes([0.74, 0.02, 0.15, 0.04], facecolor='black')

    button_1 = Button(ax_button_1, '1', color='white', hovercolor='cornflowerblue')
    button_10 = Button(ax_button_10, '10', color='white', hovercolor='cornflowerblue')
    button_1000 = Button(ax_button_1000, '1000', color='white', hovercolor='cornflowerblue')
    button_1000000 = Button(ax_button_1000000, '1 000 000', color='white', hovercolor='cornflowerblue')

    def on_button_click(it):
        iterations = it
        #rerun gradient descent

    button_1.on_clicked(1)
    button_10.on_clicked(10)
    button_1000.on_clicked(1000)
    button_1000000.on_clicked(1000000)

    plt.show()

# def plotCostFunction():


# STEP 5: ANALYZE ACCURACY OF MODEL




# MAIN

def main():
    learningRate = 0.0001
    thetas = [0, 0]

    mileages, prices = getData()

    x, y = normalizeData(mileages, prices)

    for i in range(iterations):
        thetas = runGradientDescent(x, y, thetas, learningRate)

    # print(thetas[0])
    # print(thetas[1])

    # for elem in mileages:
    #     result = thetas[1] * normalizeElem(mileages, elem) + thetas[0]
    #     print(elem, ": ", round(denormalizeElem(prices, result)))

    plotLinearRegression(thetas, mileages, prices)

    # STORE ALL THETAS IN CSV

if	__name__ == '__main__':
    main()
