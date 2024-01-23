import sys
from utils import normalizeElem, denormalizeElem

def predictPrice(mileage, thetas):
    return thetas[0] + thetas[1] * mileage

def predictPriceNorm(mileage, thetas, mileages, prices):
    price = thetas[1] * normalizeElem(mileages, mileage) + thetas[0]
    return (denormalizeElem(prices, price))
    # get last computed theta from thetas.csv

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
    mileage = getMileage()
    print(mileage)

if __name__ == "__main__":
    main()
