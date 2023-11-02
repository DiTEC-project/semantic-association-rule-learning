CREATE TABLE IF NOT EXISTS SensorData (
    time TIMESTAMP NOT NULL,
    value DOUBLE PRECISION NULL,
    sensor_type TEXT NOT NULL,
    ID TEXT NOT NULL
);

select create_hypertable('SensorData', 'time');