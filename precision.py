import math
import numpy as np
from train import getData
from predict import predictPriceNorm

def getLastThetas(file):
    with open(file, 'r') as f:
        thetas = [0.0, 0.0]
        lines = f.readlines()
        if lines:
            lastLine = lines[-1].strip()
            thetas[0] = float(lastLine.split(',')[0])
            thetas[1] = float(lastLine.split(',')[1])
        return thetas
        
def mse(y, y_hat):
	return sum((y_hat - y) ** 2) / y.size

def rmse(y, y_hat):
	ret = mse(y, y_hat)
	return np.sqrt(ret)

def mae(y, y_hat):
	return sum(abs(y_hat - y)) / y.size

def r2score(y, y_hat):
	return 1 - (sum((y_hat - y) ** 2) / sum((y - y.mean()) ** 2))

def main():
    thetas = getLastThetas('thetas.csv')

    x, y = getData()

    y_hat = []
    for i in range (len(x)):
        y_hat.append(predictPriceNorm(x[i], thetas, x, y))

    my_mse = mse(y, y_hat)
    print("mse: ", my_mse)
    my_rmse = rmse(y, y_hat)
    print("rmse: ", my_rmse)
    my_mae = mae(y, y_hat)
    print("mae: ", my_mae)
    my_r2score = r2score(y, y_hat)
    print("r2score: ", my_r2score)

if __name__ == "__main__":
    main()

