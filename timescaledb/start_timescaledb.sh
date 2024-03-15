#!/bin/bash

docker run -d --name timescaledb -p 5432:5432 -e POSTGRES_PASSWORD=password --restart always timescale/timescaledb:latest-pg15

docker cp db_setup.sql timescaledb:/opt/db_setup.sql

sleep 5

docker exec -it timescaledb bash -c 'cd /opt && value=`cat db_setup.sql|tr -d "\n"|cat` && psql -U postgres -c "$value"'
