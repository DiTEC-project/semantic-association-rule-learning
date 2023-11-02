import json
from src.util.json_util import *

from src.repository.BaseRepository import BaseRepository


class NodeRepository(BaseRepository):
    """
    This class contains Neo4j specific database operations about managing nodes
    """

    def add_nodes(self, nodes):
        """
        add given list of nodes to the graph database
        :param nodes: a list of nodes where each node contains at least following attributes: id, type, name.
            The rest of the attributes of each node is added as they are.
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
