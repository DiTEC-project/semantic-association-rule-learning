"""
Copyright (C) 2023 University of Amsterdam
@author Erkan Karabulut – e.karabulut@uva.nl
@version 1.0
Preprocessing functions that are common across all algorithms implemented
"""
import itertools

from src.util.graph_util import *
from src.util.vector_util import *
from src.util.json_util import *
from src.util.transactions_util import *

# list of categorical and numerical attributes to consider (not all attributes are taken into account for now)
categorical_attributes = ['type', 'measurement_aspect']
numerical_attributes = ['diameter', 'length']

# to be used when calculating permutation of input data for neighbor properties
MAX_NEIGHBOR_COUNT = 4


def get_combinations_of_attributes(object_id, sensor_measurement, list_of_all_attributes):
    """
    Calculate combinations of given neighboring nodes' attributes together with the object itself and its sensor
    measurement. So far (26.07.2023), it only considers type (label) of each node and edge, and not their attributes
    :param object_id:
    :param sensor_measurement:
    :param list_of_all_attributes: in the form of ['Pipe', 'Junction', 'Sensor', ...]
    :return: list of tuples with all possible combinations
    """
    base_tuple = (object_id, sensor_measurement)
    combination_list = []

    for L in range(len(list_of_all_attributes) + 1):
        for subset in itertools.combinations(list_of_all_attributes, L):
            # combination_list.append(frozenset(subset + base_tuple))
            combination_list.append(object_id + "___" + str(sensor_measurement) + "___" + str(subset))
    return combination_list


def filter_knowledge_graph_props(knowledge_graph, list_of_props_to_keep=None):
    """
    remove the properties of nodes in the knowledge graph, except the given list of properties to keep
    :param knowledge_graph: knowledge graph in NetworkX format
    :param list_of_props_to_keep: list of names of props to keep
    """
    # linearize the KG properties, meaning that turn properties with the type list and object into plain key-value pairs
    for node_id in knowledge_graph.nodes:
        knowledge_graph.nodes[node_id]['properties'] = linearize(
            knowledge_graph.nodes[node_id]['properties'])

    # create a subset of the KG that includes only the categorical + numerical attributes defined above
    if list_of_props_to_keep is None:
        list_of_props_to_keep = categorical_attributes + numerical_attributes
    for node_id in knowledge_graph.nodes:
        new_props = {}
        for attribute in list_of_props_to_keep:
            if attribute in knowledge_graph.nodes[node_id]['properties']:
                new_props[attribute] = knowledge_graph.nodes[node_id]['properties'][attribute]
        knowledge_graph.nodes[node_id]['properties'] = new_props

    return knowledge_graph


def get_transactions_as_cat_vectors(knowledge_graph, transactions, num_bins):
    """
    discretize all numerical data and apply one-hot encoding to both categorical and discrete numerical data
    :param knowledge_graph: knowledge graph in NetworkX format
    :param transactions: discrete timeseries data from sensors in the form of list of transactions
    :param num_bins: number of bins to discretize the numerical values into categories
    :return: list of one-hot encoded vectors representing categorical and discrete numerical data
    """
    # calculate boundaries for the ranges of sensor values, per sensor type
    boundaries = calculate_discrete_boundaries(knowledge_graph, transactions, num_bins)

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

    # create a vector "tracker" object that keeps track of which value goes in which order
    # this will be used to track input order to our AE model in the next step(s)
    vector_item_order = []
    for node_type in unique_values_per_attribute["type"]:
        for random_node_id in knowledge_graph.nodes:
            if "type_" + node_type in knowledge_graph.nodes[random_node_id]['properties'] and \
                    knowledge_graph.nodes[random_node_id]['properties']["type_" + node_type] == 1:
                for key in knowledge_graph.nodes[random_node_id]['properties'].keys():
                    if key not in vector_item_order and key != 'name':
                        vector_item_order.append(key)
                break

    # create vector representations of sensor values, numerical and categorical value from the KG
    vector_list = []
    vector_tracker_list = []

    for transaction in transactions:
        for item in transaction:
            vector_tracker = []

            sensor_id = item.split("_", 1)[1].split('__')[0]
            sensor_type = item.split("_", 1)[1].split('__')[1]
            measurement = float(item.split("_", 1)[0])
            node = knowledge_graph.nodes[sensor_id.replace('s_', '', 1)]
            neighbors = get_first_neighbor_with_relations(knowledge_graph, node)

            # todo: topology is not encoded explicitly, is it necessary? if yes, how do we do that?
            vector, indices = create_vector_rep_measurement(measurement, boundaries, sensor_type)
            vector_tracker += indices

            values, indices = create_vector_rep_node(node, vector_item_order)
            vector += values
            vector_tracker += indices

            # todo: assumed max 4 neighbors and calculate permutations based on these 4 neighbors, this assumption
            #  will be changed in the future

            vector_list.append(vector)
            vector_tracker_list.append(vector_tracker)

    return {
        'vector_list': vector_list,
        'vector_tracker_list': vector_tracker_list
    }


def get_transactions_as_vectors(knowledge_graph, timeseries_dataset):
    """
    create a common numerical representation in a vector format, for the given timeseries_dataset and the
    knowledge graph
    :param knowledge_graph: knowledge graph in NetworkX format
    :param timeseries_dataset: timeseries data from sensors in the form of list of transactions
    """
    # filter the kg properties, but keep the name as an identifier of the nodes which won't be used in the learning
    knowledge_graph = filter_knowledge_graph_props(knowledge_graph,
                                                   categorical_attributes + numerical_attributes + ["name"])

    unique_values_per_attribute = {}
    # apply one-hot encoding on the categorical attributes
    for node_id in knowledge_graph.nodes:
        new_props = {}
        # filter numerical attributes
        for attribute in numerical_attributes:
            if attribute in knowledge_graph.nodes[node_id]['properties']:
                new_props[attribute] = float(knowledge_graph.nodes[node_id]['properties'][attribute])

        # filter categorical attributes and one-hot encode them
        for attribute in categorical_attributes:
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
                    new_props["cat_" + value] = 0
                new_props["cat_" + knowledge_graph.nodes[node_id]['properties'][attribute]] = 1
        # the "name" attribute is listed, just to use it as an identifier, when finding the neighbors of each node,
        # and it won't be included in the learning process
        new_props['name'] = knowledge_graph.nodes[node_id]['properties']['name']
        knowledge_graph.nodes[node_id]['properties'] = new_props

    # vector representation of categorical and numerical attributes in the knowledge graph
    vector_list_categorical = []
    vector_list_numerical = []
    # vector representation of the sensor data
    vector_list_sensor_data = []
    # in the 'ae_main', input data is fed into the autoencoder in the order of sensor data + numerical data +
    # categorical the array below, tracks the values of each by indexing them
    vector_list_tracker = []
    for transaction in timeseries_dataset:
        vector_categorical = []
        vector_numerical = []
        vector_sensor_data = []
        for item in transaction:
            sensor_id = item.split("_", 1)[1].split('__')[0]
            measurement = item.split("_", 1)[0]
            node = knowledge_graph.nodes[sensor_id.replace('s_', '', 1)]
            neighbors = get_first_neighbor_with_relations(knowledge_graph, node)

            vector_sensor_data.append(float(measurement))
            vector_list_tracker.append(sensor_id)
            for key in node['properties'].keys():
                if key != 'name':
                    vector_list_tracker.append(sensor_id + "---" + key)
                    if key.startswith('cat_'):
                        vector_categorical.append(node['properties'][key])
                    elif not key.startswith('cat_'):
                        vector_numerical.append(node['properties'][key])
            for neighbor in neighbors:
                for key in neighbor['neighbor']['properties'].keys():
                    if key != 'name':
                        vector_list_tracker.append(sensor_id + "---" + key)
                        if key.startswith('cat_'):
                            vector_categorical.append(neighbor['neighbor']['properties'][key])
                        elif not key.startswith('cat_'):
                            vector_numerical.append(neighbor['neighbor']['properties'][key])
        vector_list_sensor_data.append(vector_sensor_data)
        vector_list_categorical.append(vector_categorical)
        vector_list_numerical.append(vector_numerical)

    return {
        'categorical': vector_list_categorical,
        'numerical': vector_list_numerical,
        'sensor': vector_list_sensor_data,
        'id_tracker': vector_list_tracker
    }
