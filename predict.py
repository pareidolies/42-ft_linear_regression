import sys
import os

import numpy as np
from colorama import Fore, Back, Style

from utils import normalizeElem, denormalizeElem, getLastThetas, getData

def predictPrice(mileage, thetas):
    return thetas[0] + thetas[1] * mileage

def predictPriceNorm(mileage, thetas, mileages, prices):
    price = thetas[1] * normalizeElem(mileages, mileage) + thetas[0]
    return (denormalizeElem(prices, price))

def getMileage():
    while 1:
        print("What is the mileage of the car of interest?")
        try:
            mileage = input()
        except EOFError:
            sys.exit('EOF error')
        except:
            sys.exit('Input error')
        mileage = int(mileage)
        if mileage >= 0:
            break
        else:
            print('Mileage should be positive')
    return (mileage)

def main():
    mileages, prices = getData()

    # if before training, set thetas to 0
    file = 'thetas.csv'
    if(os.path.exists(file) and os.path.isfile(file)): 
        thetas = getLastThetas('thetas.csv')
    else:
        thetas = [0,0]

    mileage = getMileage()

    estimation = predictPriceNorm(mileage, thetas, mileages, prices)
    print('The estimated price of a car with ' + str(mileage) + ' km is ' + Fore.BLUE + str(round(estimation, 2)) + Style.RESET_ALL)

    if (mileage in mileages):
        print('The reported price in our database is ' + Fore.GREEN + str(prices[np.where(mileages == mileage)[0][0]]) + Style.RESET_ALL)
        
if __name__ == "__main__":
    main()
