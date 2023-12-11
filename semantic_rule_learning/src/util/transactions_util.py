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
            sensor_id = item.split("_", 1)[1]

            for edge_list in subgraph:
                for edge in edge_list:
                    if sensor_id == edge['source']['properties']['id'] or sensor_id == \
                            edge['destination']['properties']['id']:
                        new_transaction.append(item)
        subset.append(list(dict.fromkeys(new_transaction)))

    return subset
