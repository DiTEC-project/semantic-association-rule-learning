from dotenv import load_dotenv

from src.algorithm.ae_semrl import AESemRL
from src.repository.graphdb.node_repository import NodeRepository
from src.preprocessing.semantic_enrichment import *

load_dotenv()

if __name__ == "__main__":
    # knowledge graph
    node_repository = NodeRepository()
    knowledge_graph_neo4j = node_repository.get_all_nodes_with_relations()
    knowledge_graph_networkx = neo4j_to_networkx(knowledge_graph_neo4j)

    # sensor data
    sensor_data_repository = SensorDataRepository()
    # create nodes on the KG for sensors, if they don't exist already
    unique_sensor_ids = sensor_data_repository.get_unique_sensor_ids()
    for sensor in unique_sensor_ids:
        node_repository.add_sensor(sensor[0], sensor[1])

    sensor_data = sensor_data_repository.get_grouped_data_by_time(1440)
    transactions = timeseries_to_transactions(sensor_data)

    ae_semrl = AESemRL(knowledge_graph_networkx, transactions)
    ae_semrl.train()
    association_rules = ae_semrl.generate_rules()
