# -*- coding: utf-8 -*-
import pandas as pd
from pandas._libs.tslibs import timestamps
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import mysql.connector as connection
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime
import joblib
from sklearn.metrics import mean_squared_error


try:
    mydb = connection.connect(host="localhost", database = 'sensor_data',user="root", passwd="",use_pure=True)
    query = "Select * FROM sensor_data WHERE thingName = 'NIVÅ011' AND data_type='waterLevelMmAdjustedRH2000';"
    
    result_dataFrame = pd.read_sql(query,mydb)
    
    #convert
    result_dataFrame['timestamp'] = pd.to_datetime(result_dataFrame['timestamp']).dt.date
    result_dataFrame['from'] = pd.to_datetime(result_dataFrame['from']).dt.date
   
    df=result_dataFrame[["thingName","timestamp","value","smhi_rain","from"]]
    #shape
    df = df.drop(columns=['thingName','timestamp','from'])
    # x, y with sklearn and convert to numpy.ndarry
    x = df[['smhi_rain']].to_numpy()
    y = df[['value']].to_numpy()
    # change the data type to avoid array problem
    x = x.astype(np.float64,copy=False)
    y = y.astype(np.float64,copy=False)
    #split data
    x_train, x_test,y_train,y_test = train_test_split(x,y,test_size =0.2, random_state=0)
    regressor = LinearRegression()
    regressor.fit(x_train,y_train)
    #Coefficienrt of determination (R2)
    r_sq = regressor.score(x_train,y_train)
    print('coefficient of determination:', r_sq)
    #
    y_pred = regressor.predict(x_test)
    # The mean squared error
    print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
    # The coefficient  
    print('Coefficients: \n', regressor.coef_)
    # The intercept
    print('Intercept: \n', regressor.intercept_)
    #plot out put
    
    plt.scatter(x_train, y_train, color = "red", label="Data point")
    plt.plot(x_train, regressor.predict(x_train), color = "green", label="Linear Regression")
    plt.title("Water Level vs Rain (Training set)")
    plt.xlabel("Rain")
    plt.ylabel("Water Level")
    plt.legend()
    plt.show()
    #
    rain=[[1.0]]
    ##
    predicted_waterLevel = regressor.predict(rain)
    print(predicted_waterLevel)
    if(predicted_waterLevel > -0.5):
        print("Will flood soon")
    else:
        print("Water level is OK")
    dir='C:\\wamp64\\www\\Flood_project\\linear_regrassion\\output\\'
    filename ='final_model_2.sav'
    joblib.dump(regressor, dir + filename)
    ##
    ## database cloding 
    mydb.close() #close the connection
except Exception as e:
    mydb.close()
    print(str(e))





