from src.util.graph_util import *
from src.util.converter_util import *
from src.repository.timescaledb.sensor_data_repository import SensorDataRepository
from src.preprocessing.base_preprocessing import *


def naive_semrl_enrich_transactions(knowledge_graph, disc_hist_time_series, num_bins):
    """
    Get grouped transactions from the timeseries database and enrich transactions that contains only sensor data with
    semantics from the knowledge graph. This enrichment is specific to the Naive SemRL approach, because all the values
    are in the string format, so that they can be one-hot encoded for the FP-Growth.
    :param knowledge_graph: knowledge graph in NetworkX format
    :param disc_hist_time_series: discrete time-series sensor data
    :param num_bins: number of bins to discretize sensor values into
    :return:
    """
    # calculate boundaries for the ranges of sensor values, per sensor type
    boundaries = calculate_discrete_boundaries(knowledge_graph, disc_hist_time_series, num_bins)

    enriched_transactions = []
    for transaction in disc_hist_time_series:
        new_transaction = []
        # new_transaction += transaction
        for item in transaction:
            sensor_id = item.split("_", 1)[1].split('__')[0]
            measurement = float(item.split("_", 1)[0])
            sensor_type = item.split("__")[1]
            node = knowledge_graph.nodes[sensor_id.replace('s_', '', 1)]
            current_node_attributes = [('s_' + key + "_" + str(node['properties'][key])) for key in
                                       node['properties'].keys()]

            for value_index in range(len(boundaries[sensor_type]) - 1):
                if boundaries[sensor_type][value_index] <= measurement <= boundaries[sensor_type][value_index + 1]:
                    measurement = str(boundaries[sensor_type][value_index]) + "__" + \
                                  str(boundaries[sensor_type][value_index + 1])
                    break

            # neighbors = get_first_neighbor_with_relations(knowledge_graph, node)
            # topology = get_topology(node, neighbors)
            # neighbors_attributes = get_attributes([neighbors])

            for attribute in current_node_attributes:
                if not attribute.startswith('s_name'):
                    new_transaction.append("sensor_" + sensor_type + "_" + measurement + "___" + attribute)
            enriched_transactions.append(new_transaction)

    return enriched_transactions
