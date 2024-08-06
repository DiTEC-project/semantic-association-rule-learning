import os

import psycopg2
from psycopg2.extensions import AsIs
from src.repository.graphdb.node_repository import NodeRepository


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
        self.node_repository = NodeRepository()

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

    def get_grouped_data_by_time(self, time_interval_in_minutes: int, precision: int = 0, subsample: int = 0):
        # also filter the data due to space and time complexity of Naive SemRL
        sensor_name_list = []
        for row in self.get_unique_sensor_names():
            sensor_name_list.append(row[0])
        if subsample > 0:
            sensor_name_list = self.node_repository.get_random_sensor_subgraph(subsample)

        # this is necessary to fill time gaps, e.g. if we don't have a measurement from a sensor at a specific
        # time frame, then we will put a 0
        time_intervals = self.get_mix_max_time()
        with psycopg2.connect(self.connection) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT time_bucket_gapfill('%(minutes)s minutes', time) AS time_interval, "
                            "CASE WHEN avg(value) IS NULL THEN 0 ELSE round(cast(avg(value) as numeric), 0) END as average, "
                            "CONCAT('_name_', name, '_end__type_', sensor_type, '_end_') id FROM %(table_name)s s "
                            "where name = ANY(%(sensor_name_list)s) AND "
                            "time >= %(start_interval)s AND time <= %(end_interval)s "
                            "GROUP BY time_interval, name, sensor_type "
                            "ORDER BY time_interval, id",
                            {'minutes': time_interval_in_minutes, 'precision': precision,
                             'sensor_name_list': sensor_name_list, 'table_name': AsIs(self.table_name),
                             'start_interval': time_intervals[0], 'end_interval': time_intervals[1]})
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

    def get_mix_max_time(self):
        with psycopg2.connect(self.connection) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT min(time) as min, max(time) as max from %s", (AsIs(self.table_name),))
                result = cur.fetchall()
                return result[0]
