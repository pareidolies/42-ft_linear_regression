import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('data.csv')

x = np.array(data['km'])
y = np.array(data['price'])


plt.title("Car price estimation depending on mileage")
plt.xlabel('km')
plt.ylabel('price')
plt.grid(True)

plt.plot(x,y, "bs")
plt.show()
