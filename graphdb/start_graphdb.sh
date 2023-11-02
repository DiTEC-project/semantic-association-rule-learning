#!/bin/bash

docker run \
  --name graphdb -d --restart always \
  --publish=7474:7474 --publish=7687:7687 \
  --volume=$HOME/neo4j/data:/data \
  --env NEO4J_AUTH=neo4j/password \
  neo4j:latest

sleep 5

# todo: authentication
docker exec -it graphdb bash -c 'mv labs/apoc* plugins/ && neo4j restart'
