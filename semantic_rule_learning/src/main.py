from dotenv import load_dotenv

from src.algorithm.ae_semrl_cat import AESemRLCat
from src.repository.graphdb.node_repository import NodeRepository
from src.preprocessing.semantic_enrichment import *
from src.preprocessing.base_preprocessing import *
from src.algorithm.naive_semrl import NaiveSemRL

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

    # filter the kg properties, but keep the name as an identifier of the nodes which won't be used in the learning
    knowledge_graph = filter_knowledge_graph_props(knowledge_graph_networkx,
                                                   categorical_attributes + numerical_attributes + ["name"])

    # discretize numerical attributes in the knowledge graph
    knowledge_graph = discretize_numerical_attributes(knowledge_graph, numerical_attributes, num_bins=5)

    naive_semrl = NaiveSemRL(min_support=0.8, min_confidence=0.8, num_bins=5)
    full_assoc_rules = naive_semrl.learn_semantic_association_rules(knowledge_graph_networkx, transactions)

    ae_semrl = AESemRLCat(knowledge_graph_networkx, transactions, num_bins=5)
    ae_semrl.train()
    association_rules = ae_semrl.generate_rules(similarity_threshold=0.7)
