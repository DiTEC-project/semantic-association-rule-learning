import json
import os.path
import psycopg2

from datetime import datetime
from dateutil import parser

# TimescaleDB confing
host = "192.168.1.249"
port = "5432"
username = "postgres"
password = "password"

connection = "postgres://{}:{}@{}:{}".format(username, password, host, port)

sensor_data_file_path = os.path.join(
    os.path.dirname(__file__),
    '../data/sensor/recorded.json'
)

manual_match_file_path = os.path.join(
    os.path.dirname(__file__),
    '../data/sensor/manual-match.json'
)

with open(manual_match_file_path) as manual_match_file:
    manual_match_data = json.load(manual_match_file)


def find_object_id_and_type(sensor_id_text):
    for key in manual_match_data:
        if key == sensor_id_text:
            return manual_match_data[key]["id"], manual_match_data[key]["type"]
    return None


with psycopg2.connect(connection) as timescaledb_connection:
    cursor = timescaledb_connection.cursor()
    with open(sensor_data_file_path) as sensor_data_file:
        sensor_data = json.load(sensor_data_file)
        count = 0
        for attr in sensor_data:
            object_id, object_type = find_object_id_and_type(attr)
            if object_id is None:
                continue
            for time in sensor_data[attr]:
                count = count + 1
                print(count)
                cursor.execute("INSERT INTO SensorData (time, value, NAME, sensor_type) values (%s, %s, %s, %s)", (
                    parser.parse(time).isoformat(),
                    sensor_data[attr][time],
                    "s_" + object_id,
                    object_type
                ))

            timescaledb_connection.commit()
