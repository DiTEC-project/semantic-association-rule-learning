from src.util.graph_util import *
from src.util.converter_util import *
from src.repository.timescaledb.sensor_data_repository import SensorDataRepository


def naive_semrl_enrich_transactions(knowledge_graph, disc_time_period):
    """
    Get grouped transactions from the timeseries database and enrich transactions that contains only sensor data with
    semantics from the knowledge graph. This enrichment is specific to the Naive SemRL approach, because all the values
    are in the string format, so that they can be one-hot encoded for the FP-Growth.
    :param knowledge_graph: knowledge graph in NetworkX format
    :param disc_time_period: discretization time period in minutes, the sensor data will be aggregated based on this
    :return:
    """
    sensor_data_repository = SensorDataRepository()
    # discretization by averaging values per 60 minutes
    sensor_data = sensor_data_repository.get_grouped_data_by_time(disc_time_period)
    disc_hist_time_series = timeseries_to_transactions(sensor_data)

    enriched_transactions = []
    for transaction in disc_hist_time_series:
        new_transaction = []
        # new_transaction += transaction
        for item in transaction:
            sensor_id = item.split("_", 1)[1]
            measurement = item.split("_", 1)[0]
            node = knowledge_graph.nodes[sensor_id.replace('s_', '', 1)]
            current_node_attributes = [('s_' + str(value)) for value in node['properties'].values()]
            neighbors = get_first_neighbor_with_relations(knowledge_graph, node)
            topology = get_topology(node, neighbors)
            # neighbors_attributes = get_attributes([neighbors])

            new_transaction.append(measurement)
            for attribute in topology + current_node_attributes:
                new_transaction.append(attribute)
        enriched_transactions.append(new_transaction)

    return enriched_transactions


def ae_semrl_enrich_transactions(knowledge_graph, transaction_dataset):
    """
    enrich given transactions that contains only sensor data with semantics from the knowledge graph
    this enrichment is specific to the Autoencoder-based approach
    :param knowledge_graph: knowledge graph in NetworkX format
    :param transaction_dataset: sensor measurements as list of transactions
    :return:
    """
