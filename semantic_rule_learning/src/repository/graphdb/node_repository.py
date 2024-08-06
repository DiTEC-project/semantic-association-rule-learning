import json

from src.repository.graphdb.base_repository import BaseRepository
from src.util.graph_util import *


class NodeRepository(BaseRepository):
    """
    This class contains Neo4j specific database operations about managing nodes
    """

    def get_all_nodes(self):
        with self.driver.session() as session:
            query = "MATCH (n:Node) RETURN n"
            nodes = json.dumps(session.run(query).data())
            session.close()
            return nodes

    def get_all_nodes_with_relations(self):
        with self.driver.session() as session:
            query = "MATCH (s)-[r]-(d) RETURN s,r,d"
            result = session.run(query).data()
            session.close()
            return result

    def add_sensor(self, object_id, sensor_type):
        with self.driver.session() as session:
            query = "match (n {name: $id})\n" \
                    "MERGE (s:Sensor {name: $sensor_id, type: 'Sensor', measurement_aspect: $type})-[:Placed_In]->(n)"
            session.run(query, {
                "id": object_id.replace('s_', '', 1).replace('demand', '').replace('pressure', '').replace('flow', ''),
                "sensor_id": object_id,
                "type": sensor_type
            })
            session.close()

    def get_random_sensor_subgraph(self, sensor_node_count):
        """
        (Random subsampling of the KG) Get a subgraph that has "neighboring_sensor_count" amount of sensors
        Starts from "sensor_name" node and gradually searches 1st, 2nd, 3rd ... neighbors to find
        "neighboring_sensor_count" amount of sensors in total
        """
        sensor_name_list = []

        with self.driver.session() as session:
            path_length = 2
            paths = []
            while len(paths) < sensor_node_count or path_length > 40:
                query = "MATCH (a:Sensor)-[]-(t) " + \
                        "with a.name as randomSensor, rand() as r " + \
                        "order by r limit 1 " + \
                        "MATCH (n {name: randomSensor}) " + \
                        "OPTIONAL MATCH p=(n)-[*1.." + str(path_length) + "]-(neighbor) " + \
                        "WHERE neighbor: Sensor " + \
                        "with p, collect(distinct neighbor) AS neighbors " + \
                        "return neighbors, length(p) " + \
                        "order by length(p) asc"
                paths = []
                results = session.run(query).data()
                for row in results:
                    for neighbor in row['neighbors']:
                        if neighbor not in paths:
                            paths.append(neighbor)
                path_length += 1
            for sensor_node in paths[:sensor_node_count]:
                sensor_name_list.append(sensor_node['name'])

        return sensor_name_list
