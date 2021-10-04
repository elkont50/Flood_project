# -*- coding: utf-8 -*-
from typing import overload
import pandas as pd
from pandas._libs.tslibs import timestamps
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import mysql.connector as connection
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import datetime 
import time
import joblib
from sklearn.metrics import mean_squared_error

# Gloabla variable
predicted_waterLevel=""
level_word=""
result = []
things=""
thing=""
j=""


def linear_model():
    try:
      mydb = connection.connect(host="localhost",port=3307, database = 'fp_test',user="root", passwd="",use_pure=True)
      query = "Select * FROM sensor_data WHERE thingName BETWEEN 'NIVÅ001' AND 'NIVÅ033' AND data_type='waterLevelMmAdjustedRH2000';"
      result_dataFrame = pd.read_sql(query,mydb)
      #convert
      result_dataFrame['timestamp'] = pd.to_datetime(result_dataFrame['timestamp']).dt.date
      result_dataFrame['from'] = pd.to_datetime(result_dataFrame['from']).dt.date
      df= result_dataFrame[["id","thingName","timestamp","value","smhi_rain","from"]]         
      #print(df.tail(7))
      # prediction function
      def ml():
        thing = j 
        
        new_df= result[[2,11,10,12,17]]
        #print(new_df)
        new_df= new_df.drop(columns=[2,11,17])
        #print(df)
        #print(df.shape)
        # x, y with sklearn and convert to numpy.ndarry
        x = new_df[[12]].to_numpy()
        y = new_df[[10]].to_numpy()
        # change the data type to avoid array problem
        x = x.astype(np.float64,copy=False)
        y = y.astype(np.float64,copy=False)
        #print("x:", x)
        #print("y:",y)
        x_train, x_test,y_train,y_test = train_test_split(x,y,test_size =0.2, random_state=0)
        regressor = LinearRegression()
        regressor.fit(x_train,y_train)
        #Coefficienrt of determination (R2)
        r_sq = regressor.score(x_train,y_train)
        print('coefficient of determination:', r_sq)
        #convert to pass to db
        r_sq =float(r_sq)
        #
        y_pred = regressor.predict(x_test)

        # The mean squared error
        r2 ='Mean squared error: %.2f' % mean_squared_error(y_test, y_pred) 
        print('Mean squared error: %.2f'
              % mean_squared_error(y_test, y_pred))
        # The coefficient         
        print('Coefficients: \n', regressor.coef_)
        # The intercept
        intercept = regressor.intercept_.astype('float64')
        intercept = float(intercept)
        print(intercept)
        print('Intercept: \n', regressor.intercept_)
        print("-------------------------")
        #plot
        
        plt.scatter(x_train, y_train, color = "red", label="Data point")
        plt.plot(x_train, regressor.predict(x_train), color = "green", label="Linear Regression")
        plt.title("Water Level vs Rain (Training set)")
        plt.xlabel("Rain")
        plt.ylabel("Water Level")
        plt.legend()
        plt.show()

        #last rain value
        rain = result_dataFrame[["smhi_rain"]].tail(1)        
        #print(rain)
        ##
        predicted_waterLevel = regressor.predict(rain)
        #print('Water Level:\n',predicted_waterLevel)
        #print(j)
        if(("NIVÅ001">= j and j <= "NIVÅ014") and (predicted_waterLevel >= 800)):
                level_1 ="Will flood soon"
                level_word = level_1
        else:
                level_1 ="Water level is OK"
                level_word = level_1
        if((('NIVÅ017'>= j and j <= 'NIVÅ028')) and (predicted_waterLevel >= 800)):  
                level_1 ="Will flood soon"
                level_word = level_1
        else:
                level_1 ="Water level is OK"
                level_word = level_1        
        if((('NIVÅ029'>= j and j <='NIVÅ032') and (predicted_waterLevel >= -500))):
                level_1 ="Will flood soon"
                level_word = level_1
        else:
                level_1 ="Water level is OK"
                level_word = level_1        
        if((('NIVÅ015'== j or j =='NIVÅ016') and (predicted_waterLevel >= -500))):  
                level_1 ="Will flood soon"
                level_word = level_1
        else:
                level_1 ="Water level is OK"
                level_word = level_1        
        if((('NIVÅ033' == j) and (predicted_waterLevel >= 4000))):  
                level_1 ="Will flood soon"
                level_word = level_1
        else:
                level_1 ="Water level is OK"
                level_word = level_1

        dir='C:\\wamp64\\www\\Flood_project\\linear_regrassion\\output\\files\\'
        #x=1
        filename = 'final_model_'+ thing +'.sav'
        #x+=1
        joblib.dump(regressor, dir + filename)
        ##   
        #last value is the things 24 hour befor
        old_value = new_df[10].tail(1)
        old_value = float(old_value)
               # print("old_value:",old_value)    
        things = thing
               #old_value=-176.0
        predicted_waterLevel.item(0,0)
        predicted_waterLevel= float(predicted_waterLevel)      
        print("-------------------------")
        print("things:",thing)
        print("old_value:",old_value)
        print("pridic _value:",predicted_waterLevel)
        print("water_level:",level_word)
        print("---------'****-------------")
        # Preparing SQL query to INSERT a record into the database
        # time 
        ts = time.time()
        timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        print(timestamp)
        cursor.execute ("""
            INSERT INTO ml_result
                (thingName,old_value,predicted_level,comment,r2,r_sq,intercept)
                VALUES(%s,%s,%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE  
                timestamp=%s,old_value=%s,predicted_level=%s,comment=%s,r2=%s,r_sq=%s,intercept=%s
                """, (things,old_value,predicted_waterLevel,level_word,r2,r_sq,intercept,timestamp,old_value,predicted_waterLevel,level_word,r2,r_sq,intercept)) 
        print("model Done!") 
              
        # for loop to loop sensor data by sensor name
        #"NIVÅ019",
        #Arry hold the things name to do foor loop throught it, we nne to featch each sensor data that way we used this way
      new_arry = []
      new_arry= ['NIVÅ001','NIVÅ002','NIVÅ003','NIVÅ004','NIVÅ005','NIVÅ006','NIVÅ007','NIVÅ008','NIVÅ009','NIVÅ010','NIVÅ011','NIVÅ012','NIVÅ013','NIVÅ014',
            "NIVÅ015","NIVÅ016","NIVÅ017","NIVÅ018","NIVÅ020","NIVÅ021","NIVÅ022","NIVÅ023","NIVÅ024","NIVÅ025","NIVÅ026","NIVÅ027","NIVÅ028","NIVÅ029",
            "NIVÅ030","NIVÅ031","NIVÅ032","NIVÅ033"]
            #print("data:",data)'
    
      for j in new_arry:
        
        thing = j
        cursor = mydb.cursor()
        query = """SELECT * FROM sensor_data WHERE thingName=%s AND data_type='waterLevelMmAdjustedRH2000'"""
        #
        record = cursor.execute(query,(thing,))
        result = cursor.fetchall()
        #convert
        result = pd.DataFrame(result)
        #
        new_df= result[[2,11,10,12,17]]
        #run predic function
        ml()
        #
      print("Thing fetching finished")  
    except Exception as e:
    # Rolling back in case of error
      mydb.rollback()
      mydb.close()
      print(str(e))

 
