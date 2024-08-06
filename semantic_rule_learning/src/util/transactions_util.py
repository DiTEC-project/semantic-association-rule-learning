"""
Includes utility functions for processing transaction(s)
"""
from src.util.graph_util import *
from src.repository.timescaledb.sensor_data_repository import SensorDataRepository
from src.util.discretization_util import *


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


def calculate_discrete_boundaries(transactions, num_bins):
    """
    calculate discrete boundaries for each type of sensor data in the transaction set
    :param transactions: transactions in the form of list of lists
    :param num_bins: number of bins to discretize the transactions in
    """
    sensor_data_repository = SensorDataRepository()
    sensor_types = sensor_data_repository.get_unique_sensor_types()

    sensor_values_per_type = {}
    for sensor_type in sensor_types:
        sensor_values_per_type[sensor_type[0]] = []

    for transaction in transactions:
        for item in transaction:
            measurement = item.split("_", 1)[0]
            sensor_type = item.split("_type_", 1)[1].split('_end_')[0]
            sensor_values_per_type[sensor_type].append(float(measurement))

    boundary_map = {'label': {}}
    for sensor_type in sensor_values_per_type:
        # sort values in increasing order
        if len(sensor_values_per_type[sensor_type]) == 0:
            continue
        boundaries = equal_frequency_discretization(sensor_values_per_type[sensor_type], num_bins)

        boundary_map[sensor_type] = boundaries
        boundary_map['label'][sensor_type] = []
        for index in range(len(boundaries) - 1):
            boundary_map['label'][sensor_type].append(
                "sensor_type_" + sensor_type + "_end__range_" + str(boundaries[index]) + "_" + str(
                    boundaries[index + 1]) + "_end_")

    return boundary_map
