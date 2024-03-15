import json
from src.util.json_util import *

from src.repository.BaseRepository import BaseRepository


class LinkRepository(BaseRepository):
    def add_links(self, links):
        """
        add links for the water network dataset
        """
        for link in links:
            link = linearize(link)
            link["id"] = link["name"]
            link["type"] = link["link_type"]
            link["name"] = link["link_type"] + "_" + link["name"]
            with self.driver.session() as session:
                query = "WITH apoc.convert.fromJsonMap($link) AS document CREATE(p:Link) SET p = document"
                session.run(query, {
                    "link": json.dumps(link)
                })

                query = "match (n1:Node {id: $source})\n" \
                        "match (n2:Node {id: $destination})\n" \
                        "match (l:Link {id: $pipe_name})\n" \
                        "CREATE (n1)-[:Link]->(l)\n" \
                        "CREATE (l)-[:Link]->(n2)\n"
                session.run(query, {
                    "source": link["start_node_name"],
                    "destination": link["end_node_name"],
                    "pipe_name": link["id"]
                })

    def create_edge(self, edge_label, source_node_name, destination_node_name):
        with self.driver.session() as session:
            query = "match (n1 {name: $source}), (n2 {name: $destination})\n" \
                    "merge (n1)-[r:" + edge_label + "]->(n2)"
            session.run(query, {
                'source': source_node_name,
                'destination': destination_node_name
            })
