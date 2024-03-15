import json
import os.path
import psycopg2
import csv

from datetime import datetime
from dateutil import parser
from os import listdir
import pandas as pd

# TimescaleDB confing
host = "192.168.1.249"
port = "5432"
username = "postgres"
password = "password"

connection = "postgres://{}:{}@{}:{}".format(username, password, host, port)

if __name__ == '__main__':
    with psycopg2.connect(connection) as timescaledb_connection:
        cursor = timescaledb_connection.cursor()
        main_data_path = '../data/sensor/L-TOWN_Real/'

        excel_file = pd.ExcelFile(main_data_path + '2018_SCADA.xlsx')
        flows_sheet = pd.read_excel(excel_file, 'Flows (m3_h)')

        for i, row in flows_sheet.iterrows():
            for column in flows_sheet.columns[1:]:
                cursor.execute("INSERT INTO ltown (time, value, NAME, sensor_type) values (%s, %s, %s, %s)", (
                    row['Timestamp'],
                    row[column],
                    "s_" + column,
                    "flow"
                ))
        timescaledb_connection.commit()
        print("Flow values are imported")

        demands_sheet = pd.read_excel(excel_file, 'Demands (L_h)')
        for i, row in demands_sheet.iterrows():
            for column in demands_sheet.columns[1:]:
                cursor.execute("INSERT INTO ltown (time, value, NAME, sensor_type) values (%s, %s, %s, %s)", (
                    row['Timestamp'],
                    row[column],
                    "s_" + column,
                    "demand"
                ))
        timescaledb_connection.commit()
        print("Demand values are imported")

        pressures_sheet = pd.read_excel(excel_file, 'Pressures (m)')
        for i, row in pressures_sheet.iterrows():
            for column in pressures_sheet.columns[1:]:
                cursor.execute("INSERT INTO ltown (time, value, NAME, sensor_type) values (%s, %s, %s, %s)", (
                    row['Timestamp'],
                    row[column],
                    "s_" + column,
                    "pressure"
                ))

        timescaledb_connection.commit()
        print("Pressure values are imported")
