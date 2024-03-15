from src.util.graph_util import *
from src.util.converter_util import *
from src.repository.timescaledb.sensor_data_repository import SensorDataRepository
from src.preprocessing.base_preprocessing import *


def enrich_transactions_fpgrowth(knowledge_graph, disc_hist_time_series, num_bins):
    """
    Get grouped transactions from the timeseries database and enrich transactions that contains only sensor data with
    semantics from the knowledge graph. This enrichment is specific to the Naive SemRL (FP-Growth) approach,
    because all the values are in the string format, so that they can be one-hot encoded for the FP-Growth.
    :param knowledge_graph: knowledge graph in NetworkX format
    :param disc_hist_time_series: discrete time-series sensor data
    :param num_bins: number of bins to discretize sensor values into
    :return:
    """
    # calculate boundaries for the ranges of sensor values, per sensor type
    boundaries = calculate_discrete_boundaries(disc_hist_time_series, num_bins)
    enriched_transactions = []
    for transaction in disc_hist_time_series:
        new_transaction = []
        # new_transaction += transaction
        for item in transaction:
            sensor_id = item.split("_name_", 1)[1].split('_end_')[0]
            measurement = float(item.split("_", 1)[0])
            sensor_type = item.split("_type_")[1].split('_end_')[0]
            node = knowledge_graph.nodes[list(knowledge_graph.neighbors(sensor_id))[0]]
            current_node_attributes = [('s_key_' + key + "_end__value_" + str(node['properties'][key]) + "_end_") for
                                       key in
                                       node['properties'].keys() if key != 'name']

            for value_index in range(len(boundaries[sensor_type]) - 1):
                if boundaries[sensor_type][value_index] <= measurement <= boundaries[sensor_type][value_index + 1]:
                    measurement = str(boundaries[sensor_type][value_index]) + "_" + \
                                  str(boundaries[sensor_type][value_index + 1])
                    break

            # neighbors = get_first_neighbor_with_relations(knowledge_graph, node)
            # topology = get_topology(node, neighbors)
            # neighbors_attributes = get_attributes([neighbors])

            new_transaction.append("sensor_type_" + sensor_type + "_end__range_" + measurement + "_end_")
            for attribute in current_node_attributes:
                if not attribute.startswith('s_name'):
                    new_transaction.append(
                        "sensor_type_" + sensor_type + "_end__range_" + measurement + "_end__attribute_" + attribute + "_end_")
        enriched_transactions.append(new_transaction)

    return enriched_transactions


def enrich_transactions_hho(knowledge_graph, time_series):
    """
    Get grouped transactions from the timeseries database and enrich transactions that contains only sensor data with
    semantics from the knowledge graph. This enrichment is specific to the Naive SemRL (HHO) approach
    :param knowledge_graph: knowledge graph in NetworkX format
    :param time_series: time series sensor data
    :return:
    """
    enriched_transactions = []
    column_names = []
    for item in time_series[0]:
        sensor_id = item.split("_name_", 1)[1].split('_end_')[0]
        sensor_type = item.split("_type_")[1].split('_end_')[0]
        node = knowledge_graph.nodes[list(knowledge_graph.neighbors(sensor_id))[0]]
        current_node_attributes = [(sensor_id + '--' + key) for key in node['properties'].keys() if key != 'name']
        column_names.append(sensor_id + "--" + sensor_type)
        column_names += current_node_attributes

    for transaction in time_series:
        new_transaction = []
        for item in transaction:
            sensor_id = item.split("_name_", 1)[1].split('_end_')[0]
            measurement = float(item.split("_", 1)[0])
            node = knowledge_graph.nodes[list(knowledge_graph.neighbors(sensor_id))[0]]
            current_node_attributes = [node['properties'][key] for key in node['properties'].keys() if key != 'name']

            # neighbors = get_first_neighbor_with_relations(knowledge_graph, node)
            # topology = get_topology(node, neighbors)
            # neighbors_attributes = get_attributes([neighbors])

            new_transaction.append(measurement)
            for attribute in current_node_attributes:
                new_transaction.append(attribute)
        enriched_transactions.append(new_transaction)

    enriched_transactions.insert(0, column_names)
    return enriched_transactions
