import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import csv


X = [] #empty list for stock date values
Y = [] #empty list for stock opening price values

#creating price prediction method to plot the models and print data
def price_prediction(dates, prices, x): #prediction method that accepts date, price, and prediction parameters
   dates = np.reshape(dates,(len(dates), 1)) #conversion to matrix of n x 1 for dates

   #defining the support vector regression models
   svr_lin = SVR(kernel= 'linear', C= 1000)#C is the penalty parameter
   svr_poly = SVR(kernel= 'poly', C= 1000, degree= 2)#degree is the power you are taking to
   svr_rbf = SVR(kernel= 'rbf', C= 1000, gamma= .1) #defining Radial Basis Function, C and gamma should be exponentially spaced
   svr_rbf.fit(dates, prices) # fitting the data points in the models
   svr_lin.fit(dates, prices)
   svr_poly.fit(dates, prices)

   #plotting the data using matplotlib library
   plt.scatter(dates, prices, color= 'black', label= 'Data') # plotting the initial datapoints
   plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') # plotting the line made by the RBF kernel
   plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model') # plotting the line made by linear kernel
   plt.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model') # plotting the line made by polynomial kernel
   plt.xlabel('Stock Date by Day of Month')
   plt.ylabel('Daily Opening Price')
   plt.title('Support Vector Regression')
   plt.legend()
   plt.show()

   rbf_prediction = svr_rbf.predict(x)[0]#storing the models as variables and printing the results
   lin_prediction = svr_lin.predict(x)[0]
   poly_prediction = svr_poly.predict(x)[0]
   print('RBF Price Prediction: ', rbf_prediction)#printing the stock prediction results
   print('Linear Price Prediction: ', lin_prediction)
   print('Polynomial Price Prediction: ', poly_prediction)


   return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]#returning prediction from each model in case we want to store


#function to read the stock data csv files
def extract_data(filename):
    with open(filename, 'r') as csvfile:
        StockDataReader = csv.reader(csvfile)#storing csv file from reader in variable
        next(StockDataReader)	# skipping the first column headers in the excel file
        for row in StockDataReader:
            X.append(data[i][0:-1])
            y.append(data[i][-1])#converting price to float so it is more precise
    return


extract_data(r'D:\TUGAS\Arsip Tugas Kuliah\SEMESTER 6\PPCD\detektif-tomat-2-master\feature\feature.csv') # calling extract_data method by passing the csv file to it

price_prediction(X, Y, 20)
