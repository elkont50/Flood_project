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
import datetime 
import time
import joblib
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
Axes3D = Axes3D  # pycharm auto import

# Gloabla variable
predicted_waterLevel=""
level_word=""
result = []
things=""
thing=""
j=""

def model():
    print("Start..!")
    try:
           mydb = connection.connect(host="localhost", database = 'sensor_data',user="root", passwd="",use_pure=True)
           #sea level
           query2 = "Select cast(thingName as char) as seaLevel,ROW_NUMBER() OVER(ORDER BY id) row_num, cast(value as char) as sea_value FROM sensor_data WHERE thingName BETWEEN 'NIVÅ015' AND 'NIVÅ016' AND data_type='waterLevelMmAdjustedRH2000';"
           df2 = pd.read_sql(query2,mydb)
           #ground water
           query3 = "Select  cast(thingName as char) as groundLevel,ROW_NUMBER() OVER(ORDER BY id) row_num, cast(value as char) as ground_value,smhi_rain  FROM sensor_data WHERE data_type='waterLevel';"
           df3 = pd.read_sql(query3,mydb)
           def ml():
             thing = j 
           # read values
           #waterlevel range 1
             df1_1= result[["row_num","waterLevel","water_value"]]
             df2_2=pd.DataFrame(df2[["row_num","seaLevel","sea_value"]])
             df3_3=pd.DataFrame(df3[["row_num","groundLevel","ground_value","smhi_rain"]])
             # concatniting data
             df=pd.merge(df3_3,df1_1, on="row_num")
             df_last=pd.merge(df,df2_2,on="row_num")
             # x, y with sklearn convert to nump.ndarray
             x = df_last[["smhi_rain","sea_value","ground_value"]].to_numpy()# here we have 3 variables for multiple regression. 
             y = df_last[["water_value"]].to_numpy() 
             # change the data type to avoid array problem
             x = x.astype(np.float64,copy=False)
             y = y.astype(np.float64,copy=False)
             # splitting the data
             x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
             # creating an object of LinearRegression class
             ml = LinearRegression()
             # fitting the training data
             ml.fit(x_train,y_train)
             #predict the test result
             y_pred=ml.predict(x_test)
             #
             #Coefficienrt of determination (R2)
             r_sq = ml.score(x_train,y_train)
             print('coefficient of determination:', r_sq)
              #convert to pass to db
             r_sq =float(r_sq)
             #
             y_pred = ml.predict(x_test)
            # The mean squared error
             r2 ='Mean squared error: %.2f' % mean_squared_error(y_test, y_pred) 
             print('Mean squared error: %.2f'
              % mean_squared_error(y_test, y_pred))
             # The coefficient  
             print('Coefficients: \n', ml.coef_)
             # The intercept
             # The intercept
             intercept = ml.intercept_.astype('float64')
             intercept = float(intercept)
             print(intercept)
             print('Intercept: \n', ml.intercept_)
             #
             values=[[8.0,40,2.79]]
             ##
             predicted_waterLevel = ml.predict(values)
             print(predicted_waterLevel)
             if(predicted_waterLevel > 1200):
                level_1 ="Will flood soon"
                level_word = level_1
             else:
                level_1 ="Water level is OK"
                level_word = level_1
             dir='C:\\wamp64\\www\\Flood_project\\multiple_regrassion\\output\\files'
             filename ='Multi_'+ thing +'.sav'
             joblib.dump(ml, dir + filename)
             # #last value is the things 24 hour befor
             old_value = result["water_value"].tail(1)
             print(old_value)
             old_value = float(old_value)
             #predicted_waterLevel =predicted_waterLevel.value
             things = thing
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
               INSERT INTO multiple_result
                (thingName,old_value,predicted_level,comment,r2,r_sq,intercept)
                VALUES(%s,%s,%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE  
                timestamp=%s,old_value=%s,predicted_level=%s,comment=%s,r2=%s,r_sq=%s,intercept=%s
                """, (things,old_value,predicted_waterLevel,level_word,r2,r_sq,intercept,timestamp,old_value,predicted_waterLevel,level_word,r2,r_sq,intercept)) 
     
           print("model Done!") 
           #19 is missed 
           new_arry = []
           new_arry= ['NIVÅ001','NIVÅ002','NIVÅ003','NIVÅ004','NIVÅ005','NIVÅ006','NIVÅ007','NIVÅ008','NIVÅ009','NIVÅ010','NIVÅ011','NIVÅ012','NIVÅ013','NIVÅ014',
            "NIVÅ015","NIVÅ016","NIVÅ017","NIVÅ018","NIVÅ020","NIVÅ021","NIVÅ022","NIVÅ023","NIVÅ024","NIVÅ025","NIVÅ026","NIVÅ027","NIVÅ028","NIVÅ029",
            "NIVÅ030","NIVÅ031","NIVÅ032","NIVÅ033"]
            
    
           for j in new_arry:
        
            thing = j
            #print(j)
            cursor = mydb.cursor()
            query = """SELECT cast(thingName as char) as waterLevel, ROW_NUMBER() OVER(ORDER BY id) row_num, cast(value as char) as water_value FROM sensor_data WHERE thingName=%s AND data_type='waterLevelMmAdjustedRH2000'"""
            result = cursor.execute(query,(thing,))
            result = cursor.fetchall()
            #convert
            result = pd.DataFrame(result)
            #
            result.columns = ['waterLevel','row_num', 'water_value']
            #print("this result for one thing:", df1_1)
            #run predic function
            ml()
            #
            print("Thing fetching finished")
        #close the connection
    except Exception as e:
        mydb.close()
        print(str(e))