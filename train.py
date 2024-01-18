import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from predict import predictPrice

# STEP 1: GET DATA


# STEP 2: FEATURE SCALING

# shall we withdraw outliers ?

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

# STEP 3: GRADIENT DESCENT

def runGradientDescent(x, y, thetas, learningRate):
    m = len(y)
    newThetas = [0, 0]

    # Perform gradient descent for each theta
    for i in range(len(thetas)):

        # Compute partial derivative of error function according to ith theta
        errors = 0
        for nbr in range(m):
            if (i == 0):
                errors += (predictPrice(x[nbr], thetas) - y[nbr])
            else:
                errors += (predictPrice(x[nbr], thetas) - y[nbr]) * x[nbr]
        errors /= m

        # Update theta with derivate of error function multiplied by learning rate
        newThetas[i] = thetas[i] - learningRate * errors
    
    return newThetas

# STEP 4: PLOT

def plotLinearRegression(thetas, mileages, prices):
    plt.title("Car price estimation depending on mileage")
    plt.xlabel('km')
    plt.ylabel('price')
    plt.grid(True)

    lineX = [float(min(mileages)), float(max(mileages))]
    lineY = []
    for elem in lineX:
    	result = thetas[1] * normalizeElem(mileages, elem) + thetas[0]
    	lineY.append(denormalizeElem(prices, result)) 

    plt.plot(mileages,prices, "bs", lineX, lineY, 'r-')
    plt.show()


def main():
    learningRate = 0.0001
    #1 10 1000 100000 
    iterations = 1000000
    thetas = [0, 0]

    data = pd.read_csv('data.csv')

    mileages = np.array(data['km'])
    prices = np.array(data['price'])

    x, y = normalizeData(mileages, prices)

    for i in range(iterations):
        thetas = runGradientDescent(x, y, thetas, learningRate)

    print(thetas[0])
    print(thetas[1])

    for elem in mileages:
        result = thetas[1] * normalizeElem(mileages, elem) + thetas[0]
        print(elem, ": ", round(denormalizeElem(prices, result)))

    plotLinearRegression(thetas, mileages, prices)


if	__name__ == '__main__':
    main()
