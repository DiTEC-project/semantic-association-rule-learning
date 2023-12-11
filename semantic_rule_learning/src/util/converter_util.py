from collections import defaultdict
import networkx as nx


def timeseries_to_transactions(sensor_data):
    """
    Convert timescaledb output to transactions that can be processed by an ARM algorithm
    :param sensor_data: in the form of timescaledb objects
    :return: list of transactions, list of sensor ids for items in the transactions
    """
    # group sensor data based on timestamp
    groups = defaultdict(list)
    transaction_list = []
    for item in sensor_data:
        if item[1] is None or item[2] is None:
            continue
        groups[item[0]].append(item)

    # add items with the same timestamp in the same list
    for timestamp in groups:
        measurement_list = []
        for measurement in groups[timestamp]:
            measurement_list.append(str(measurement[1]) + "_" + measurement[2])
        transaction_list.append(measurement_list)

    return transaction_list


def neo4j_to_networkx(neo4j_graph):
    """
    Convert graph data that is in the form of Neo4j objects to more common NetworkX graph format
    :param neo4j_graph:
    :return: Same graph data in NetworkX format
    """
    networkx_graph = nx.MultiDiGraph()

    for row in neo4j_graph:
        source_id = row['s']['name'] if 'name' in row['s'] else row['s']['id']
        dest_id = row['d']['name'] if 'name' in row['d'] else row['d']['id']
        networkx_graph.add_node(source_id, labels=row['s']['type'], properties=row['s'])
        networkx_graph.add_node(dest_id, labels=row['d']['type'], properties=row['d'])
        networkx_graph.add_edge(source_id, dest_id, type=row['s']['type'] + "_" + row['d']['type'])

    return networkx_graph
