# -*- coding: utf-8 -*-
import pandas as pd
from pandas._libs.tslibs import timestamps
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import mysql.connector as connection
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from datetime import datetime
import joblib
from sklearn.metrics import mean_squared_error,confusion_matrix, accuracy_score
from mpl_toolkits.mplot3d import Axes3D
Axes3D = Axes3D  # pycharm auto import
#Import the Keras libraries and packages
import tensorflow as tf
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Dense

try:
    mydb = connection.connect(host="localhost",port=int(3306), database = 'sensor_data',user="root", passwd="",use_pure=True)
    #waterlevel range 1
    query1 = "Select cast(thingName as char) as waterLevel, ROW_NUMBER() OVER (ORDER BY id) row_num , cast(value as char) as water_value  FROM sensor_data WHERE thingName BETWEEN 'NIVÅ001' AND 'NIVÅ010' AND data_type='waterLevelMmAdjustedRH2000' ;"
    df1 = pd.read_sql(query1,mydb)
    #print(df1)
    #sea level
    query2 = "Select cast(thingName as char) as seaLevel,ROW_NUMBER() OVER (ORDER BY id) row_num, cast(value as char) as sea_value FROM sensor_data WHERE thingName BETWEEN 'NIVÅ015' AND 'NIVÅ016' AND data_type='waterLevelMmAdjustedRH2000' ;"
    df2 = pd.read_sql(query2,mydb)
    #print(df2)
    #ground water
    query3 = "Select  cast(thingName as char) as groundLevel, ROW_NUMBER() OVER (ORDER BY id) row_num, cast(value as char) as ground_value,smhi_rain  FROM sensor_data WHERE data_type='waterLevel' ;"
    df3 = pd.read_sql(query3,mydb)
    #print(df3)
    #print(df1)
    #print(df2)
    #print(df3)
    # read values
    df1_1=pd.DataFrame(df1[["row_num","waterLevel","water_value"]])
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
    # x, y with sklearn convert to nump.ndarray
    x = df_last[["smhi_rain","sea_value","ground_value"]].to_numpy()# here we have 3 variables for multiple regression. 
    y = df_last[["water_value"]].to_numpy() 
    print(x)
    print(y)
    # change the data type to avoid array problem
    x = x.astype(np.float64,copy=False)
    y = y.astype(np.float64,copy=False)
    # splitting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
    #print(x_train)
    #print(y_train)
    # Initialize the Artificial Neural Network
    classifier = Sequential()
    #Add the input layer and the first hidden layer
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 3))
    #Add the second hidden layer
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    #Add the output layer
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'sigmoid'))
    #Train the ANN
    #Compile the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    #Fit the ANN to the Training set
    classifier.fit(x_train, y_train, batch_size = 10, nb_epoch = 100)
    #Predict the Test Set Results-
    y_pred = classifier.predict(x_test)
    y_pred 
    #Make the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    accuracy_score(y_test,y_pred)

    # print(y_pred)
    
    # test values 
    #
    values=[[8.0,40,2.79]]
    ##
    predicted_waterLevel = classifier.predict(values)
    print(predicted_waterLevel)
    if(predicted_waterLevel > 1200):
        print("Will flood soon")
    else:
        print("Water level is OK")
    dir='C:\\wamp64\\www\\Flood_project\\ANN\\output\\'
    filename ='ANN_model_1.sav'
    joblib.dump(classifier, dir + filename)

    #save result in database
    

   #evaluation the model
    #r2_score(y_test,y_pred)
    #plot resut
    #plt.figure(figsize=(15,10))
    #plt.scatter(y_test,y_pred)
    #plt.xlabel("Acual")
    #plt.ylabel("predicted")
    #plt.title("Acual vs. Predicted")
    #plt.show()
    #predict values
   # pred_y_df=df.dataframe({'Actual value':y_test,'Predicted value':y_pred, 'Difference':y_test-y_pred})
   # pred_y_df[0:20]

      ## database cloding 
    mydb.close() #close the connection
except Exception as e:
    mydb.close()
    print(str(e))