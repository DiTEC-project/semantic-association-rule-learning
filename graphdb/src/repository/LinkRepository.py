import json
from src.util.json_util import *

from src.repository.BaseRepository import BaseRepository


class LinkRepository(BaseRepository):
    """
    This class contains Neo4j specific database operations about managing links in between nodes
    """

    def add_links(self, links):
        """
        create given links in the database
        :param links: a list of links where each link has the following attributes:
            id, link_type, name, start_node_name, end_node_name.
            Start and end node names refer to existing nodes in the graph
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
