import json
import os.path
import psycopg2
import csv

from datetime import datetime
from dateutil import parser
from os import listdir
import pandas as pd

# TimescaleDB confing
host = "145.3.76.61"
port = "5432"
username = "postgres"
password = "password"

connection = "postgres://{}:{}@{}:{}".format(username, password, host, port)

if __name__ == '__main__':
    with psycopg2.connect(connection) as timescaledb_connection:
        cursor = timescaledb_connection.cursor()
        with open('../data/sensor/LBNL_FDD_Dataset_FCU/FCU_OADMPRLeak_20.csv', newline='') as csvfile:
            csv_content = csv.reader(csvfile, delimiter=' ', quotechar='|')
            headers = next(csv_content)[0].split(",")
            for row in csv_content:
                measurements = row[1].split(',')
                timestamp = row[0] + " " + measurements[0]
                measurements = measurements[1:]
                for index in range(len(measurements)):
                    cursor.execute("INSERT INTO lbnl_fdd (time, value, NAME, sensor_type) values (%s, %s, %s, %s)", (
                        timestamp,
                        measurements[index],
                        "s_" + headers[index + 1],
                        headers[index + 1]
                    ))

        timescaledb_connection.commit()
