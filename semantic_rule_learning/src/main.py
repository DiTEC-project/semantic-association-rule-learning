import pandas as pd

from dotenv import load_dotenv

from src.algorithm.NaiveSemRL import NaiveSemRL
from src.util.ConverterUtil import *
from src.repository.timescaledb.SensorDataRepository import SensorDataRepository
from src.repository.graphdb.NodeRepository import NodeRepository
from src.preprocessing.BasePreprocessing import *

load_dotenv()

if __name__ == "__main__":
    # knowledge graph
    node_repository = NodeRepository()
    knowledge_graph_neo4j = node_repository.get_all_nodes_with_relations()
    knowledge_graph_networkx = neo4j_to_networkx(knowledge_graph_neo4j)

    # sensor data
    sensor_data_repository = SensorDataRepository()
    unique_sensor_ids = sensor_data_repository.get_unique_sensor_ids()

    # create nodes
    for sensor in unique_sensor_ids:
        node_repository.add_sensor(sensor[0], sensor[1])

    rule_learning = NaiveSemRL(min_support=0.2, min_confidence=0.9)

    # discretization by averaging values per 60 minutes
    sensor_data = sensor_data_repository.get_grouped_data_by_time(1440)

    transactions = timeseries_to_transactions(sensor_data)
    assc_rules = rule_learning.learn_semantic_association_rules(knowledge_graph_networkx, transactions)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
      print(assc_rules)
