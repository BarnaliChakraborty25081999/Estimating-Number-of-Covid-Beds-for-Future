import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv('covidbeds.csv')

x = data['days']
y = data['beds']

linear_reg = LinearRegression()
linear_reg.fit(data[['days']],data.beds)

ploynomial_reg = PolynomialFeatures(degree=3)
real_x_poly = ploynomial_reg.fit_transform(data[['days']])
ploynomial_reg.fit(real_x_poly,y)
linear_reg2 = LinearRegression()
linear_reg2.fit(real_x_poly,y)
plt.scatter(x,y,color = 'red')

plt.plot(data[['days']],linear_reg2.predict(ploynomial_reg.fit_transform(data[['days']])),color = 'blue')
plt.title('polynomial model')
plt.xlabel('no of days')
plt.ylabel('no of beds')
plt.show()

linear_reg2.predict(ploynomial_reg.fit_transform([[150]]))
