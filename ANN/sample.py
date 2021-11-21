# -*- coding: utf-8 -*-
import pandas as pd
from pandas._libs.tslibs import timestamps
import datetime 
from datetime import datetime 
import time
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pymysql
from  sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from datetime import datetime
import joblib
from sklearn.metrics import mean_squared_error,confusion_matrix, accuracy_score
from mpl_toolkits.mplot3d import Axes3D
Axes3D = Axes3D  # pycharm auto import
#Import the Keras libraries and packages
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense,Dropout
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import json
import sys
from sklearn.preprocessing import MinMaxScaler
from numpy import array , hstack
from tensorflow.keras.layers import Dense,LSTM,Activation

#us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

# Gloabla variable
predicted_waterLevel=""
level_word=[]
result = []
things=""
thing=""
j=""

def model():
#print("Start..!")
#waterlevel range 1 
    mydb=pymysql.connect(host='localhost',port=int(3307),user='root',password='',db='flood_prediction') 
    #
def ml():
        thing = j
        
        cursor = mydb.cursor()
        query1="""SELECT cast(timestamp as char) as Date, cast(thingName as char) as waterLevel, ROW_NUMBER() OVER(ORDER BY id) row_num, cast(value as char) as water_value FROM sensor_data WHERE thingName=%s AND data_type='waterLevelMmAdjustedRH2000'"""
        result_q1 = cursor.execute(query1,(thing,))
        result_q1 = cursor.fetchall()       
        result_q1 = pd.DataFrame(result_q1)
        result_q1.columns = ['Date','waterLevel','row_num', 'water_value']
        
        query2="Select cast(thingName as char) as seaLevel,ROW_NUMBER() OVER (ORDER BY id) row_num, cast(value as char) as sea_value FROM sensor_data WHERE thingName BETWEEN 'NIVÅ015' AND 'NIVÅ016' AND data_type='waterLevelMmAdjustedRH2000';"
        df2=pd.read_sql(query2,mydb)
        
        query3="Select  cast(thingName as char) as groundLevel, ROW_NUMBER() OVER (ORDER BY id) row_num, cast(value as char) as ground_value,smhi_rain  FROM sensor_data WHERE data_type='waterLevel';"
        df3=pd.read_sql(query3,mydb)
        
        # read values
        #print(df1)
        #print(df2)
        #print(df3)
        
        #print("thing name in ml()",thing)
        #Separate dates for future plotting and insertion in db
        result["Date"] = pd.to_datetime(result["Date"])
        
        old_date = result["Date"].tail(10)
        
        result_q1["Date"] = pd.to_datetime(result_q1["Date"])
        train_dates = result_q1["Date"]  

        #print(train_dates)
        #print(train_dates.tail(15))#Check last few dates. 
        # read values
        #print(df1)
        #print(df2)
        #print(df3)
        #waterlevel range 1
        df1_1= result_q1[["row_num","waterLevel","water_value"]]
        df2_2=pd.DataFrame(df2[["row_num","seaLevel","sea_value"]])
        df3_3=pd.DataFrame(df3[["row_num","groundLevel","ground_value","smhi_rain"]])
        #print reading result
        #print(df1_1)
        #print(df2_2)
        #print(df3_3)
        # concatniting data
        #dataframe=[df3_3,df2_2,df1_1]
        #df1.merge(df2_2,how='left', left_on='Column1', right_on='ColumnA')
        df=pd.merge(df3_3,df1_1, on='row_num')
        df_last=pd.merge(df,df2_2, on='row_num')
        #print(df_last)

        x_1 = df_last['smhi_rain']
        x_2 = df_last['sea_value']
        x_3 = df_last['ground_value']
        y = df_last['water_value']

        x_1 = x_1.values
        x_2 = x_2.values
        x_3 = x_3.values
        y = y.values

        # Step 1 : convert to [rows, columns] structure
        x_1 = x_1.reshape((len(x_1), 1))
        x_2 = x_2.reshape((len(x_2), 1))
        x_3 = x_3.reshape((len(x_3), 1))
        y = y.reshape((len(y), 1))

        print ("x_1.shape" , x_1.shape) 
        print ("x_2.shape" , x_2.shape)
        print ("x_3.shape" , x_3.shape)  
        print ("y.shape" , y.shape)

        # Step 2 : normalization 
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_1_scaled = scaler.fit_transform(x_1)
        x_2_scaled = scaler.fit_transform(x_2)
        x_3_scaled = scaler.fit_transform(x_3)
        y_scaled = scaler.fit_transform(y)

        # Step 3 : horizontally stack columns
        dataset_stacked = hstack((x_1_scaled, x_2_scaled,x_3_scaled, y_scaled))
        print ("dataset_stacked.shape" , dataset_stacked.shape)

        # split a multivariate sequence into samples
        def split_sequences(sequences, n_steps_in, n_steps_out):
            X, y = list(), list()
            for i in range(len(sequences)):
                # find the end of this pattern
                end_ix = i + n_steps_in
                out_end_ix = end_ix + n_steps_out-1
                # check if we are beyond the dataset
                if out_end_ix > len(sequences):
                    break
                # gather input and output parts of the pattern
                seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
                X.append(seq_x)
                y.append(seq_y)
            return array(X), array(y)

        # choose a number of time steps #change this accordingly
        n_steps_in, n_steps_out = 60 , 10 

        # covert into input/output
        X, y = split_sequences(dataset_stacked, n_steps_in, n_steps_out)

        print ("X.shape" , X.shape) 
        print ("y.shape" , y.shape) 

        split = 4817*85
        train_X , train_y = X[:split, :] , y[:split, :]
        test_X , test_y = X[split:, :] , y[split:, :]

        n_features = train_X.shape[2]

        print ("train_X.shape" , train_X.shape) 
        print ("train_y.shape" , train_y.shape) 
        print ("test_X.shape" , test_X.shape) 
        print ("test_y.shape" , test_y.shape) 
        print ("n_features" , n_features)

        #optimizer learning rate
        opt = keras.optimizers.Adam(learning_rate=0.01)

        # define model
        model = Sequential()
        model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
        model.add(LSTM(50, activation='relu'))
        model.add(Dense(n_steps_out))
        model.add(Activation('linear'))
        model.compile(loss='mse' , optimizer=opt , metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error','accuracy'])

        # Fit network
        history = model.fit(train_X , train_y , epochs=5 , steps_per_epoch=25 , verbose=1 ,validation_data=(test_X, test_y) ,shuffle=False)



new_arry = []
new_arry= ['NIVÅ001','NIVÅ002','NIVÅ003','NIVÅ004','NIVÅ005','NIVÅ006','NIVÅ007','NIVÅ008','NIVÅ009','NIVÅ010','NIVÅ011','NIVÅ012','NIVÅ013','NIVÅ014',
"NIVÅ015","NIVÅ016","NIVÅ017","NIVÅ018","NIVÅ020","NIVÅ021","NIVÅ022","NIVÅ023","NIVÅ024","NIVÅ025","NIVÅ026","NIVÅ027","NIVÅ028","NIVÅ029",
"NIVÅ030","NIVÅ031","NIVÅ032","NIVÅ033"]
#print("data:",data)'
    
for j in new_arry:
        
    thing = j
    print("thing name:",j)
    #print("data lenght:",len(result_dataFrame['thingName']))
    mydb=pymysql.connect(host='localhost',port=int(3307),user='root',passwd='',db='flood_prediction') 
    cursor = mydb.cursor()
    query = """SELECT DATE(timestamp) as Date, cast(thingName as char) as waterLevel, ROW_NUMBER() OVER(ORDER BY id) row_num, cast(value as char) as water_value FROM sensor_data WHERE thingName=%s AND data_type='waterLevelMmAdjustedRH2000' GROUP BY DATE(timestamp)  ORDER BY Date DESC LIMIT 10"""
    #
    result = cursor.execute(query,(thing,))
    result = cursor.fetchall()
    #print(result)
    #
    #cast(thingName as char) as waterLevel, ROW_NUMBER() OVER(ORDER BY id) row_num, cast(value as char) as water_value
    #convert
    #result=pd.DataFrame(result[["row_num","waterLevel","water_value"]])
        
    result = pd.DataFrame(result)
   
    #print(result)
    #
    #print(result.head())
    #
    result.columns = ['Date','waterLevel','row_num', 'water_value']
    #
    #print(result)
    #
        
    #print("this result for one thing:", df1_1)
    #run predic function
    #print("thing name:",j)     
    ml()
    sys.exit
    #
    #print("Thing fetching finished")
    
    ## database cloding 
mydb.commit()
cursor.close()
mydb.close() #close the connection
print("Database closed!")