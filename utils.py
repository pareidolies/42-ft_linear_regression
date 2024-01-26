import csv
import os
import sys

import pandas as pd
import numpy as np

## NORMALIZATION

def	normalizeElem(list, elem):
    return ((elem - min(list)) / (max(list) - min(list)))

def	denormalizeElem(list, elem):
    return ((elem * (max(list) - min(list))) + min(list))

## CSV MANAGEMENT

def getLastThetas(file):
    with open(file, 'r') as f:
        thetas = [0.0, 0.0]
        lines = f.readlines()
        if lines:
            lastLine = lines[-1].strip()
            thetas[0] = float(lastLine.split(',')[0])
            thetas[1] = float(lastLine.split(',')[1])
        return thetas

def getData():
    try:
        data = pd.read_csv('data.csv')

        mileages = np.array(data['km'])
        prices = np.array(data['price'])

    except:
        sys.exit('Csv file not found')

    return(mileages, prices)

def getThetas():
    data = pd.read_csv('thetas.csv')

    t0 = np.array(data['t0'])
    t1 = np.array(data['t1'])
    cost = np.array(data['cost'])

    return(t0, t1, cost)

def createCsv():
    file = 'thetas.csv'
    if(os.path.exists(file) and os.path.isfile(file)): 
        os.remove(file)
    with open('thetas.csv', 'w') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvWriter.writerow(['t0', 't1', 'cost'])
