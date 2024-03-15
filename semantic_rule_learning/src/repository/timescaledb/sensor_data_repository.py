import os
import random

import psycopg2
from psycopg2.extensions import AsIs


class SensorDataRepository:
    """
    This class contains TimescaleDB specific database operations about managing sensor data
    """

    def __init__(self):
        self.connection = "postgres://{}:{}@{}:{}/{}" \
            .format(os.getenv("TIMESCALEDB_USER"),
                    os.getenv("TIMESCALEDB_PASSWORD"),
                    os.getenv("TIMESCALEDB_HOST"),
                    os.getenv("TIMESCALEDB_PORT"),
                    os.getenv("TIMESCALEDB_DB"))
        self.table_name = os.getenv("TIMESCALEDB_TABLE")

    def get_all_data(self):
        with psycopg2.connect(self.connection) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM %s", (AsIs(self.table_name),))
                result = cur.fetchall()
                return result

    def get_data_by_sensor(self, object_id):
        with psycopg2.connect(self.connection) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM %s where name = %s", (AsIs(self.table_name), object_id,))
                result = cur.fetchall()
                return result

    def get_grouped_data_by_time(self, time_interval_in_minutes: int, precision: int = 0):
        # also filter the data due to space and time complexity of Naive SemRL
        sensor_name_list = []
        for row in self.get_unique_sensor_names():
            sensor_name_list.append(row[0])
        sensor_name_list = random.sample(sensor_name_list, 7)
        # print(sensor_name_list)
        with psycopg2.connect(self.connection) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT time_bucket('%(minutes)s minutes', time) AS time_interval, "
                            "round(cast(avg(value) as numeric), %(precision)s) as average, "
                            "CONCAT('_name_', name, '_end__type_', sensor_type, '_end_') id FROM %(table_name)s s "
                            "where name = ANY(%(sensor_name_list)s) GROUP BY time_interval, name, sensor_type "
                            "ORDER BY time_interval, id",
                            {'minutes': time_interval_in_minutes, 'precision': precision,
                             'sensor_name_list': sensor_name_list, 'table_name': AsIs(self.table_name)})
                result = cur.fetchall()
                return result

    def get_unique_sensor_ids(self):
        with psycopg2.connect(self.connection) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT distinct name, sensor_type from %s", (AsIs(self.table_name),))
                result = cur.fetchall()
                return result

    def get_unique_sensor_names(self):
        with psycopg2.connect(self.connection) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT distinct name from %s", (AsIs(self.table_name),))
                result = cur.fetchall()
                return result

    def get_unique_sensor_types(self):
        with psycopg2.connect(self.connection) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT distinct sensor_type from %s", (AsIs(self.table_name),))
                result = cur.fetchall()
                return result
