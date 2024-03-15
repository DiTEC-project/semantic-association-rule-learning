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
            return json.dumps(session.run(query).data())

    def get_all_nodes_with_relations(self):
        with self.driver.session() as session:
            query = "MATCH (s)-[r]-(d) RETURN s,r,d"
            return session.run(query).data()

    def add_sensor(self, object_id, sensor_type):
        with self.driver.session() as session:
            query = "match (n {name: $id})\n" \
                    "MERGE (s:Sensor {name: $sensor_id, type: 'Sensor', measurement_aspect: $type})-[:Placed_In]->(n)"
            session.run(query, {
                "id": object_id.replace('s_', '', 1).replace('demand', '').replace('pressure', '').replace('flow', ''),
                "sensor_id": object_id,
                "type": sensor_type
            })

    def get_all_neighbors(self, object_id, n_neighbors):
        neighbors = []
        if type(n_neighbors) != int or n_neighbors < 0 or n_neighbors > 1000:
            print("Neighbor count to include in rule learning is forced set to 2.")
            n_neighbors = 2

        with self.driver.session() as session:
            query = "MATCH p=(s {id: $id})-[*" + str(n_neighbors) + ".." + str(n_neighbors) + "]-() RETURN p"
            paths = session.run(query, {"id": object_id}).data()
            for path in paths:
                source_type = get_node_type(path["p"][0])
                link_type = path["p"][1]
                destination_type = get_node_type(path["p"][2])
                neighbors.append({
                    'source_type': source_type,
                    'link_type': link_type,
                    'destination_type': destination_type
                })

        return neighbors
