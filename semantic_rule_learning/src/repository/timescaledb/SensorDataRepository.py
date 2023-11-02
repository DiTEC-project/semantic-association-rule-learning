import os

import psycopg2


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

    def get_all_data(self):
        with psycopg2.connect(self.connection) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM sensordata")
                result = cur.fetchall()
                return result

    def get_data_by_sensor(self, object_id):
        with psycopg2.connect(self.connection) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM sensordata where id = %s", (object_id,))
                result = cur.fetchall()
                return result

    def get_grouped_data_by_time(self, time_interval_in_minutes: int, precision: int = 0):
        with psycopg2.connect(self.connection) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT time_bucket('%s minutes', time) AS time_interval, "
                            "round(cast(avg(value) as numeric), %s) as average, id "
                            "FROM sensordata s "
                            "GROUP BY time_interval, id "
                            "ORDER BY time_interval", (time_interval_in_minutes, precision))
                result = cur.fetchall()
                return result

    def get_unique_sensor_ids(self):
        with psycopg2.connect(self.connection) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT distinct id, sensor_type from sensordata")
                result = cur.fetchall()
                return result
