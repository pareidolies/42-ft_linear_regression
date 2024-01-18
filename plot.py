import matplotlib.pyplot as plt


def plotLinearRegression(thetas, mileages, prices):
    plt.title("Car price estimation depending on mileage")
    plt.xlabel('km')
    plt.ylabel('price')
    plt.grid(True)

    lineX = [float(min(mileages)), float(max(mileages))]
    lineY = []
    for elem in lineX:
    	result = thetas[1] * normalizeElem(mileages, elem) + thetas[0]
    	lineY.append(denormalizeElem(prices, elem)) 

    plt.plot(mileages,prices, "bs", lineX, lineY, 'r-')
    plt.show()
