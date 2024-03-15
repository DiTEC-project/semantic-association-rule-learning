import json
from src.util.json_util import *

from src.repository.BaseRepository import BaseRepository


class NodeRepository(BaseRepository):
    def add_nodes(self, nodes):
        """
        add nodes for water network dataset
        """
        for node in nodes:
            node = linearize(node)
            node["id"] = node["name"]
            node["type"] = node["node_type"]
            node["name"] = node["node_type"] + "_" + node["name"]
            with self.driver.session() as session:
                query = 'WITH apoc.convert.fromJsonMap($node) AS document CREATE(p:Node) SET p = document'
                session.run(query, {
                    "node": json.dumps(node)
                })

    def create_node(self, label, properties):
        with self.driver.session() as session:
            query = "merge (n:" + str(label) + ") set n = $properties"
            session.run(query, {
                'properties': properties
            })

    def mark_sensors(self, node_name, new_name, label):
        with self.driver.session() as session:
            query = "match (n {name: $name}) set n.type = $type, n.name = $new_name, n:" + label
            session.run(query, {
                'name': node_name,
                'type': new_name.replace('_', ''),
                'new_name': 's_' + new_name
            })
