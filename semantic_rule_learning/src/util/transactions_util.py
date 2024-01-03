"""
Includes utility functions for processing transaction(s)
"""
from src.util.graph_util import *


def get_transactions_by_subgraph(transactions, subgraph):
    """
    Filter given transactions by nodes in the given subgraph
    :param transactions:
    :param subgraph:
    :return: A list of transactions for the nodes in the given subgraph only
    """
    subset = []
    for transaction in transactions:
        new_transaction = []
        for item in transaction:
            sensor_id = item.split("_", 1)[1].split('__')[0]

            for edge_list in subgraph:
                for edge in edge_list:
                    if sensor_id == edge['source']['properties']['id'] or sensor_id == \
                            edge['destination']['properties']['id']:
                        new_transaction.append(item)
        subset.append(list(dict.fromkeys(new_transaction)))

    return subset


def calculate_discrete_boundaries(knowledge_graph, transactions, num_bins):
    """
    calculate discrete boundaries for each type of sensor data in the transaction set
    :param knowledge_graph: knowledge graph in NetworkX form
    :param transactions: transactions in the form of list of lists
    :param num_bins: number of bins to discretize the transactions in
    """
    sensor_types = get_unique_values(knowledge_graph, "measurement_aspect")

    sensor_values_per_type = {}
    for sensor_type in sensor_types:
        sensor_values_per_type[sensor_type] = []

    for transaction in transactions:
        for item in transaction:
            measurement = item.split("_", 1)[0]
            sensor_id = item.split("_", 1)[1].split('__')[0]
            node = knowledge_graph.nodes[sensor_id]
            sensor_values_per_type[node["properties"]["measurement_aspect"]].append(float(measurement))

    boundary_map = {'label': {}}
    for sensor_type in sensor_values_per_type:
        # sort values in increasing order
        sorted_values = np.sort(sensor_values_per_type[sensor_type])
        # define boundaries based on num_bins
        boundaries = np.interp(np.linspace(0, len(sorted_values), num_bins + 1),
                               np.arange(len(sorted_values)),
                               sorted_values)
        boundary_map[sensor_type] = boundaries
        boundary_map['label'][sensor_type] = []
        for index in range(len(boundaries) - 1):
            boundary_map['label'][sensor_type].append(
                "sensor_" + sensor_type + "_" + str(boundaries[index]) + "_" + str(boundaries[index + 1]))

    return boundary_map
