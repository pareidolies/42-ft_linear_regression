import math
import numpy as np
from train import getData
from predict import predictPriceNorm
from colorama import Fore, Back, Style
from utils import getLastThetas

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

    print("Model accuracy indicators: ")
    my_mse = mse(y, y_hat)
    print("mse: " + Fore.MAGENTA + str(my_mse) + Style.RESET_ALL)
    my_rmse = rmse(y, y_hat)
    print("rmse: " + Fore.MAGENTA + str(my_rmse) + Style.RESET_ALL)
    my_mae = mae(y, y_hat)
    print("mae: " + Fore.MAGENTA + str(my_mae) + Style.RESET_ALL)
    my_r2score = r2score(y, y_hat)
    print("r2score: " + Fore.MAGENTA + str(my_r2score) + Style.RESET_ALL)
    print(Fore.YELLOW + "ie " + str(round(my_r2score * 100)) + "% of variation in price is predictable from km with our model" + Style.RESET_ALL)

if __name__ == "__main__":
    main()

