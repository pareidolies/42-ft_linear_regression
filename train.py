import pandas as pd
import numpy as np
import math

from predict import predictPrice

# STEP 1: GET DATA


# STEP 2: FEATURE SCALING

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

def main():
    learningRate = 0.0001
    #1 10 1000 100000 
    iterations = 1000
    thetas = [0, 0]

    data = pd.read_csv('data.csv')

    mileages = np.array(data['km'])
    prices = np.array(data['price'])

    x, y = normalizeData(mileages, prices)

    for i in range(iterations): 
        thetas = runGradientDescent(x, y, thetas, learningRate)
    print(thetas[0])
    print(thetas[1])


if	__name__ == '__main__':
    main()