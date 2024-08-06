import json
from src.util.json_util import *

from src.repository.BaseRepository import BaseRepository


class LinkRepository(BaseRepository):
    def add_links(self, links):
        """
        add links for the water network dataset
        """
        with self.driver.session() as session:
            for link in links:
                link = linearize(link)
                link["id"] = link["name"]
                link["type"] = link["link_type"]
                link["name"] = link["link_type"] + "_" + link["name"]
                query = "WITH apoc.convert.fromJsonMap($link) AS document CREATE(p:" + link[
                    "type"] + ") SET p = document"
                session.run(query, {
                    "link": json.dumps(link)
                })

                if link["start_node_name"] == link["id"] or link["end_node_name"] == link["id"]:
                    query = "match (n1 {id: $source})\n" \
                            "match (n2 {id: $destination})\n" \
                            "MERGE (n1)-[:connectedTo]->(n2)\n"
                    session.run(query, {
                        "destination": link["end_node_name"],
                        "source": link["start_node_name"]
                    })
                else:
                    query = "match (n1 {id: $source})\n" \
                            "match (n2 {id: $destination})\n" \
                            "match (l:" + link["type"] + " {id: $pipe_name})\n" \
                                                         "MERGE (n1)-[:connectedTo]->(l)\n" \
                                                         "MERGE (l)-[:connectedTo]->(n2)\n"
                    session.run(query, {
                        "destination": link["end_node_name"],
                        "source": link["start_node_name"],
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
