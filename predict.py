import sys

def predictPrice(mileage, thetas):
    return thetas[0] + thetas[1] * mileage
    # get last computed theta from thetas.csv

#def normPredictPrice(mileage, thetas):
#    return thetas[0] + thetas[1] * mileage

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
