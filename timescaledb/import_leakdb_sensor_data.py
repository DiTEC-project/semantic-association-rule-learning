import json
import os.path
import psycopg2
import csv

from datetime import datetime
from dateutil import parser
from os import listdir

# TimescaleDB confing
host = "145.109.95.190"
port = "5432"
username = "postgres"
password = "password"

connection = "postgres://{}:{}@{}:{}".format(username, password, host, port)

if __name__ == '__main__':
    with psycopg2.connect(connection) as timescaledb_connection:
        cursor = timescaledb_connection.cursor()
        main_data_path = '../data/sensor/LeakDB_Hanoi_CMH_Scenario-1/'

        # read timestamps
        timestamps = []
        with open(main_data_path + 'Timestamps.csv') as file:
            timestamps_csv = csv.reader(file, delimiter=',', quotechar='|')
            # ignore header row
            next(timestamps_csv)
            for row in timestamps_csv:
                timestamps.append(row[1])

        # read demands (junctions)
        demands = []
        demand_files = listdir(main_data_path + 'Demands')
        for demand_file_name in demand_files:
            junction_id = "Junction_" + str(demand_file_name.replace('Node_', '').replace('.csv', ''))
            with open(main_data_path + 'Demands/' + demand_file_name) as file:
                demands_csv = csv.reader(file, delimiter=',', quotechar='|')
                # ignore header row
                next(demands_csv)
                values = []
                for row in demands_csv:
                    values.append({'timestamp': timestamps[int(row[0]) - 1], 'value': row[1]})
            demands.append({'id': junction_id, 'values': values})

        # read flows (pipes)
        flows = []
        flow_files = listdir(main_data_path + 'Flows')
        for flow_file_name in flow_files:
            pipe_id = "Pipe_" + str(flow_file_name.replace('Link_', '').replace('.csv', ''))
            with open(main_data_path + 'Flows/' + flow_file_name) as file:
                flows_csv = csv.reader(file, delimiter=',', quotechar='|')
                # ignore header row
                next(flows_csv)
                values = []
                for row in flows_csv:
                    values.append({'timestamp': timestamps[int(row[0]) - 1], 'value': row[1]})
            flows.append({'id': pipe_id, 'values': values})

        # read flows (junctions)
        pressures = []
        pressure_files = listdir(main_data_path + 'Pressures')
        for pressure_file_name in pressure_files:
            junction_id = "Junction_" + str(pressure_file_name.replace('Node_', '').replace('.csv', ''))
            with open(main_data_path + 'Pressures/' + pressure_file_name) as file:
                pressures_csv = csv.reader(file, delimiter=',', quotechar='|')
                # ignore header row
                next(pressures_csv)
                values = []
                for row in pressures_csv:
                    values.append({'timestamp': timestamps[int(row[0]) - 1], 'value': row[1]})
            pressures.append({'id': junction_id, 'values': values})

        # store the data in timescaledb
        with psycopg2.connect(connection) as timescaledb_connection:
            cursor = timescaledb_connection.cursor()

            for demand in demands:
                object_id = demand['id']
                for measurement in demand['values']:
                    cursor.execute("INSERT INTO leakdb (time, value, NAME, sensor_type) values (%s, %s, %s, %s)", (
                        parser.parse(measurement['timestamp']).isoformat(),
                        float(measurement['value']),
                        "s_" + object_id,
                        "demand"
                    ))

            for flow in flows:
                object_id = flow['id']
                for measurement in flow['values']:
                    cursor.execute("INSERT INTO leakdb (time, value, NAME, sensor_type) values (%s, %s, %s, %s)", (
                        parser.parse(measurement['timestamp']).isoformat(),
                        float(measurement['value']),
                        "s_" + object_id,
                        "flow"
                    ))

            for pressure in pressures:
                object_id = pressure['id']
                for measurement in pressure['values']:
                    cursor.execute("INSERT INTO leakdb (time, value, NAME, sensor_type) values (%s, %s, %s, %s)", (
                        parser.parse(measurement['timestamp']).isoformat(),
                        float(measurement['value']),
                        "s_" + object_id,
                        "pressure"
                    ))

            timescaledb_connection.commit()
