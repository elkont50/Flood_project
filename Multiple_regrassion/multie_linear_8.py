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
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
Axes3D = Axes3D  # pycharm auto import
try:
    mydb = connection.connect(host="localhost", database = 'sensor_data',user="root", passwd="",use_pure=True)
    #waterlevel range 1
    query1 = "Select cast(thingName as char) as waterLevel, ROW_NUMBER() OVER (ORDER BY id) row_num , cast(value as char) as water_value  FROM sensor_data WHERE thingName ='NIVÅ028' AND data_type='waterLevelMmAdjustedRH2000' ;"
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
    #print(x)
    #print(y)
    # change the data type to avoid array problem
    x = x.astype(np.float64,copy=False)
    y = y.astype(np.float64,copy=False)
    # splitting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
    # creating an object of LinearRegression class
    ml = LinearRegression()
    # fitting the training data
    ml.fit(x_train,y_train)
    #predict the test result
    y_pred=ml.predict(x_test)
   # print(y_pred)
    #Coefficienrt of determination (R2)
    r_sq = ml.score(x_train,y_train)
    print('coefficient of determination:', r_sq)
    #
    y_pred = ml.predict(x_test)
    # The mean squared error
    print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
    # The coefficient  
    print('Coefficients: \n', ml.coef_)
    # The intercept
    print('Intercept: \n', ml.intercept_)
    # test values 
    #
    values=[[0.5,30,0.79]]
    ##
    predicted_waterLevel = ml.predict(values)
    print(predicted_waterLevel)
    if(predicted_waterLevel > 900):
        print("Will flood soon")
    else:
        print("Water level is OK")
    dir='C:\\wamp64\\www\\Flood_project\\multiple_regrassion\\output\\'
    filename ='Multi_model_8.sav'
    joblib.dump(ml, dir + filename)
   #evaluation the model
    #r2_score(y_test,y_pred)
    #plot the result 
    fig1= plt.figure(figsize=(10,10))
    ax = fig1.add_subplot(111, projection='3d')
    df_last["smhi_rain"]=pd.to_numeric(df_last["smhi_rain"], downcast="float")
    x1 = df_last["smhi_rain"]
    df_last["water_value"]=pd.to_numeric(df_last["water_value"], downcast="float")
    y1=df_last["water_value"]
    df_last["sea_value"]=pd.to_numeric(df_last["sea_value"], downcast="float")
    x2=df_last["sea_value"]
    
    
    #x_surf=np.linspace(df_last.smhi_rain.min(), df_last.smhi_rain.max(), 100)
    #y_surf=np.linspace(df_last.water_value.min(), df_last.water_value.max(), 100)
    #x_surf, y_surf =np.meshgrid(x_surf, y_surf)# generatet the meash
    #z_surf=np.sqrt(x_surf + y_surf)
    #ax.plot_srface(x_surf,y_surf,z_surf)
    
    ax.scatter(x1,x2,y1, c=(x1-x2)-y1, marker='x')
    ax.set_xlabel('Rain')
    ax.set_ylabel('Water Level')
    ax.set_zlabel('Sea Level')
    ax.axis('equal')
    ax.axis('tight')
    plt.show()
    #predict values
   # pred_y_df=df.dataframe({'Actual value':y_test,'Predicted value':y_pred, 'Difference':y_test-y_pred})
   # pred_y_df[0:20]

      ## database cloding 
    mydb.close() #close the connection
except Exception as e:
    mydb.close()
    print(str(e))