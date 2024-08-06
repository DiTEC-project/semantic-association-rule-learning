"""
This Python script includes functions related to semantic enrichment of sensor data
"""
import pandas as pd
from src.preprocessing.base_preprocessing import *
from src.util.graph_util import get_unique_values
from src.util.transactions_util import calculate_discrete_boundaries
from src.util.vector_util import create_vector_rep_node, create_vector_rep_measurement


def enrich_transactions_naivesemrl(knowledge_graph, disc_hist_time_series, num_bins):
    """
    Get grouped transactions from the timeseries database and enrich transactions that contains only sensor data with
    semantics from the knowledge graph. This enrichment is specific to the Naive SemRL approach,
    because all the values are in the string format, so that they can be one-hot encoded for the exhaustive methods.
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


def enrich_transactions_tsnarm(knowledge_graph, time_series):
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


def transactions_without_semantics(disc_hist_time_series, num_bins):
    """
    Get grouped transactions from the timeseries database
    :param disc_hist_time_series: discrete time-series sensor data
    :param num_bins: number of bins to discretize sensor values into
    :return:
    """
    # calculate boundaries for the ranges of sensor values, per sensor type
    boundaries = calculate_discrete_boundaries(disc_hist_time_series, num_bins)
    enriched_transactions = []
    for transaction in disc_hist_time_series:
        new_transaction = []
        for item in transaction:
            sensor_id = item.split("_name_", 1)[1].split('_end_')[0]
            measurement = float(item.split("_", 1)[0])
            sensor_type = item.split("_type_")[1].split('_end_')[0]

            for value_index in range(len(boundaries[sensor_type]) - 1):
                if boundaries[sensor_type][value_index] <= measurement <= boundaries[sensor_type][value_index + 1]:
                    measurement = str(boundaries[sensor_type][value_index]) + "_" + \
                                  str(boundaries[sensor_type][value_index + 1])
                    break

            new_transaction.append("sensor_type_" + sensor_type + "_end__range_" + measurement +
                                   "_end__attribute__key_id_end__value_" + sensor_id + "_end_")
        enriched_transactions.append(new_transaction)

    return enriched_transactions


def semantic_enrichment_our_ae_based_arm(knowledge_graph, transactions, num_bins, num_neighbors):
    """
    discretize all numerical data and apply one-hot encoding to both categorical and discrete numerical data
    :param knowledge_graph: knowledge graph in NetworkX format
    :param transactions: discrete timeseries data from sensors in the form of list of transactions
    :param num_bins: number of bins to discretize the numerical values into categories
    :param num_neighbors:
    :return: list of one-hot encoded vectors representing categorical and discrete numerical data
    """
    # calculate boundaries for the ranges of sensor values, per sensor type
    boundaries = calculate_discrete_boundaries(transactions, num_bins)

    unique_values_per_attribute = {}
    # apply one-hot encoding on the categorical attributes (as well as numerical as they are discrete from now on)
    for node_id in knowledge_graph.nodes:
        new_props = {}
        for attribute in categorical_attributes + numerical_attributes:
            # unique values for an attribute can also be taken from the ontology underlying the KG, if defined
            if attribute in unique_values_per_attribute:
                unique_values = unique_values_per_attribute[attribute]
            else:
                unique_values = get_unique_values(knowledge_graph, attribute)
                unique_values_per_attribute[attribute] = unique_values
            # one-hot encoding on the attribute
            # mark categorical attributes with the "cat_" prefix
            if attribute in knowledge_graph.nodes[node_id]['properties']:
                for value in unique_values:
                    new_props[attribute + "_" + value] = 0
                new_props[attribute + "_" + knowledge_graph.nodes[node_id]['properties'][attribute]] = 1
        # the "name" attribute is listed, just to use it as an identifier, when finding the neighbors of each node,
        # and it won't be included in the learning process
        new_props['name'] = knowledge_graph.nodes[node_id]['properties']['name']
        knowledge_graph.nodes[node_id]['properties'] = new_props

    # create vector representations of sensor values, numerical and categorical value from the KG
    vector_list = []
    vector_tracker_list = []
    input_vector_category_indices = []

    for transaction in transactions:
        vector_tracker = []
        vector = []
        feature_tracker = []
        feature_tracker_start_index = 0
        for index in range(len(transaction)):
            item = transaction[index]
            sensor_id = item.split("_name_", 1)[1].split('_end_')[0]
            measurement = float(item.split("_", 1)[0])
            sensor_type = item.split("_type_")[1].split('_end_')[0]
            node = knowledge_graph.nodes[list(knowledge_graph.neighbors(sensor_id))[0]]

            postfix = "_item_" + str(index)

            # neighbors = get_neighbors(knowledge_graph, node, num_neighbors)
            # for neighbor_degree in neighbors.keys():
            #     for neighbor_index in range(len(neighbors[neighbor_degree])):
            #         values, indices = create_vector_rep_node(neighbors[neighbor_degree][neighbor_index],
            #                                                  "--" + str(neighbor_degree) + "--" + str(
            #                                                      neighbor_index) + "--" + postfix)
            #         vector += values
            #         vector_tracker += indices
            #         feature_tracker.append(
            #             {'start': feature_tracker_start_index, 'end': feature_tracker_start_index + len(values)})
            #         feature_tracker_start_index += len(values)

            values, indices = create_vector_rep_measurement(measurement, boundaries, sensor_type, postfix)
            vector += values
            vector_tracker += indices
            feature_tracker.append(
                {'start': feature_tracker_start_index, 'end': feature_tracker_start_index + len(values)})
            feature_tracker_start_index += len(values)

            values, indices = create_vector_rep_node(node, postfix)
            vector += values
            vector_tracker += indices
            feature_tracker.append(
                {'start': feature_tracker_start_index, 'end': feature_tracker_start_index + len(values)})
            feature_tracker_start_index += len(values)

        vector_list.append(vector)
        vector_tracker_list.append(vector_tracker)
        input_vector_category_indices.append(feature_tracker)

    return {
        'vector_list': vector_list,
        'vector_tracker_list': vector_tracker_list,
        'category_indices': input_vector_category_indices
    }


def enrich_transactions_arm_ae(knowledge_graph, transactions, num_bins, num_neighbors):
    input_vectors = semantic_enrichment_our_ae_based_arm(knowledge_graph, transactions, num_bins, num_neighbors)
    transactions = []
    for vector in input_vectors["vector_list"]:
        transactions.append([True if number != 0 else False for number in vector])
    return pd.DataFrame(transactions, columns=input_vectors['vector_tracker_list'][0])
