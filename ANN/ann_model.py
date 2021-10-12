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
    mydb=pymysql.connect(host='localhost',port=int(3307),user='root',password='',db='fp_test') 
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

        # x, y with sklearn convert to nump.ndarray
        df_last = df_last[["water_value","smhi_rain","sea_value","ground_value"]].to_numpy()# here we have 4 variables for multiple regression. 
        
        #y = df_last[["water_value"]].to_numpy() 
        #df_last=pd.merge(x,y, on="row_num")
        #print(x)
        #print(df_last)
        # normalize the dataset
        scaler = StandardScaler()
        scaler=scaler.fit(df_last)
        df_x=scaler.transform(df_last)
        #As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features. 
        #In this example, the n_features is 4. We will make timesteps = 15 (past days data used for training). 

        #Empty lists to be populated using formatted training data
        x_train = []
        y_train = []

        n_future = 1   # Number of days we want to look into the future based on the past days.
        n_past = 15  # Number of past days we want to use to predict the future.

        #Reformat input data into a shape: (n_samples x timesteps x n_features)
        #the df_x has a shape (3208, 4)
        #3208 refers to the number of data points and 4 refers to the columns (multi-variables).
        for i in range(n_past, len(df_x) - n_future +1):
                x_train.append(df_x[i - n_past:i, 0:df_x.shape[1]])
                y_train.append(df_x[i + n_future - 1:i + n_future, 0])

        x_train,y_train = np.array(x_train), np.array(y_train)

        #print(x_train)
        #print(y_train)
        #print('x_train shape == {}.'.format(x_train.shape))
        #print('y_train shape == {}.'.format(y_train.shape))
       
        # define the Autoencoder model

        model = Sequential()
        model.add(LSTM(64, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
        model.add(LSTM(32, activation='relu', return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(y_train.shape[1]))

        model.compile(optimizer='adam', loss='mse',metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'])
        model.summary()
        # fit the model
        history = model.fit(x_train, y_train, epochs=5, batch_size=16, validation_split=0.1, verbose=1)
       
        # print(y_pred)

        #plt.plot(history.history['loss'], label='Training loss')
        #plt.plot(history.history['val_loss'], label='Validation loss')
        #plt.legend()
        # plot metrics
        #plt.plot(history.history['mean_squared_error'])
        #plt.plot(history.history['mean_absolute_error'])
        #plt.plot(history.history['mean_absolute_percentage_error'])
        #plt.show()
        #
        n_future=10
        #n_past = 16
        #n_days_for_prediction=15
        #print(train_dates)
        #train_dates=pd.to_datetime(pd.Series(df1["Date"]))
        #print(train_dates)
        #list(train_dates)[-1]
        forecast_period_dates = pd.date_range(start=datetime.now(), periods=n_future, freq='1D').tolist()
        #print(forecast_period_dates)
        #predict the next 7 days
        #print(x_train)
        forecast = model.predict(x_train[-n_future:])
        #print(forecast)
        forecast_copies =np.repeat(forecast, df_last.shape[1],axis=-1)
        y_pred_future = scaler.inverse_transform(forecast_copies)[:,0]
        #print(forecast_copies)
        #print(y_pred_future)
        # Convert timestamp to date
        forecast_dates = []
        for time_i in forecast_period_dates:
             forecast_dates.append(time_i.date())
        df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Water_Level_last':y_pred_future})
        df_forecast["Date"]=pd.to_datetime(df_forecast["Date"])
       
        #print(df_forecast)
        #data preper for insertion in db
        
        Date_last = result["Date"]
        Date_last = Date_last.tail(10)
       
        #    
        old_value = result["water_value"]
        old_value = old_value.tail(10)
        print(old_value)
        
        #test values
        predicted_waterLevel = df_forecast["Water_Level_last"] 
        #print("Water Level for next 10 days :\n",predicted_waterLevel) 
        for i in predicted_waterLevel:
            if(i >= 800): 
                level_1 ="Will flood soon"
                level_word.append(level_1) 
        else:
                level_1 ="Water level is OK"
                level_word.append(level_1) 
                
        dir='C:\\Users\\ASUS\\ANN\\output\\files'
        filename ='ANN_model_'+ thing +'.sav'
        joblib.dump(predicted_waterLevel, dir + filename)
        things = thing
        
        new_date=df_forecast["Date"]
        
        print("-------------------------")
        print("things:",thing)
        print("old_date:",old_date.to_numpy())
        print("old_value:",old_value.to_numpy())
        print("pridic_value:",predicted_waterLevel.to_numpy())
        print("forcast_date",new_date.to_numpy() )
        print("water_level:",level_word)
        print("---------'****-------------")
        
        # time 
        #timestamp= datetime.datetime.now()
        #ts = time.time()
        #timestamp = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        #print(timestamp)
        
        old_date_json=json.dumps(old_date.to_numpy().tolist())
        old_value_json=json.dumps(old_value.to_numpy().tolist())
        pridic_value_json=json.dumps(predicted_waterLevel.to_numpy().tolist())
        forcast_date_json=json.dumps(new_date.to_numpy().tolist())
        water_level_json=json.dumps(level_word)
        loss=json.dumps(history.history['loss'])
        val_loss=json.dumps(history.history['val_loss'])
        mean_squared_error=json.dumps(history.history['mean_squared_error']) #
        mean_absolute_error=json.dumps(history.history['mean_absolute_error'])
        mean_absolute_percentage_error=json.dumps(history.history['mean_absolute_percentage_error'])
        
        cursor = mydb.cursor()
        cursor.execute ("""INSERT INTO ann_result  SET thingName=%s,old_date=%s,old_value=%s,predicted_level=%s,forcast_date=%s,comment=%s,loss=%s,val_loss=%s,mean_squared_error=%s,mean_absolute_error=%s,mean_absolute_percentage_error=%s """, (thing,old_date_json,old_value_json,pridic_value_json,forcast_date_json,water_level_json,loss,val_loss,mean_squared_error,mean_absolute_error,mean_absolute_percentage_error)) 
     
        #if j =="NIVÅ014":
        #exit()
        print("model Done!") 

new_arry = []
new_arry= ['NIVÅ001','NIVÅ002','NIVÅ003','NIVÅ004','NIVÅ005','NIVÅ006','NIVÅ007','NIVÅ008','NIVÅ009','NIVÅ010','NIVÅ011','NIVÅ012','NIVÅ013','NIVÅ014',
"NIVÅ015","NIVÅ016","NIVÅ017","NIVÅ018","NIVÅ020","NIVÅ021","NIVÅ022","NIVÅ023","NIVÅ024","NIVÅ025","NIVÅ026","NIVÅ027","NIVÅ028","NIVÅ029",
"NIVÅ030","NIVÅ031","NIVÅ032","NIVÅ033"]
#print("data:",data)'
    
for j in new_arry:
        
    thing = j
    print("thing name:",j)
    #print("data lenght:",len(result_dataFrame['thingName']))
    mydb=pymysql.connect(host='localhost',port=int(3307),user='root',passwd='',db='fp_test') 
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
    #
    #print("Thing fetching finished")
    
    ## database cloding 
mydb.commit()
cursor.close()
mydb.close() #close the connection
print("Database closed!")