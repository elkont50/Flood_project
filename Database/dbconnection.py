# -*- coding: utf-8 -*-
import mysql.connector

try:
    connection = mysql.connector.connect(host='localhost', database='sensor_data', user='root', password='')
    sql_select_Query = "SELECT * FROM sensor_data limit 1"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    # get all records
    records = cursor.fetchall()
    print("Total number of rows in table: ", cursor.rowcount)

    print("\nPrinting each row")
    for row in records:
        print("Id = ", row[0], )
        print("thingId = ", row[1])
        print("thingName = ", row[2])
        print("thingModel = ", row[3])
        print("thingType = ", row[4])
        print("thingLabel = ", row[5])
        print("parentID = ", row[6])
        print("thingCreated = ", row[7])
        print("data_type = ", row[8])
        print("value = ", row[9])
        print("timestamp = ", row[10])
        print("smhi_rain = ", row[11])
        print("smh_wind = ", row[12])
        print("smhi_temp = ", row[13])
        print("from = ", row[14])
        print("to = ", row[15])
        print("created_at = ", row[16])
        print("updated_at = ", row[17])


except mysql.connector.Error as e:
    print("Error reading data from MySQL table", e)
finally:
    if connection.is_connected():
        connection.close()
        cursor.close()
        print("******")
        print("*************")
        print("MySQL connection is closed")