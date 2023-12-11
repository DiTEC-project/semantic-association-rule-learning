from neo4j import GraphDatabase
import os


class BaseRepository:
    """
    This class contains common database operations such as connect, disconnect or run a query
    """

    def __init__(self):
        """
        Connect to the neo4j database using the connection parameters in .env file
        """
        # get db credentials from environment variables
        url = os.getenv("NEO4J_URL")
        user = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")

        self.driver = GraphDatabase.driver(url, auth=(user, password))

    def close(self):
        """
        Close database connection
        :return:
        """
        self.driver.close()

    def run_query(self, query, parameters):
        """
        Run a given graph db query with the given parameters
        :param query: neo4j db query
        :param parameters: parameters of the query above
        :return:
        """
        result = []
        with self.driver.session() as session:
            response = session.run(query=query, parameters=parameters)
            for line in response:
                result.append(line)

        return result

    def clean_up_db(self):
        """
        Delete all nodes and edges in the graph db
        :return:
        """
        with self.driver.session() as session:
            session.run("match(s)-[t]-(r) delete t,s,r")
            session.run("match(s) delete s")
