"""
Copyright (C) 2023 University of Amsterdam
@author Erkan Karabulut – e.karabulut@uva.nl
@version 1.0
Preprocessing functions that are common across all algorithms implemented
"""
from src.util.graph_util import *
from src.util.json_util import *


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


def filter_knowledge_graph_props(knowledge_graph, list_of_props_to_keep):
    """
    remove the properties of nodes in the knowledge graph, except the given list of properties to keep
    :param knowledge_graph: knowledge graph in NetworkX format
    :param list_of_props_to_keep: list of names of props to keep
    """
    # create a subset of the KG that includes only the categorical + numerical attributes defined above
    for node_id in knowledge_graph.nodes:
        new_props = {}
        for attribute in list_of_props_to_keep:
            if attribute in knowledge_graph.nodes[node_id]['properties']:
                new_props[attribute] = knowledge_graph.nodes[node_id]['properties'][attribute]

    return knowledge_graph


def get_transactions_as_vectors(knowledge_graph, timeseries_dataset):
    """
    create a common numerical representation in a vector format, for the given timeseries_dataset and the
    knowledge graph
    :param knowledge_graph: knowledge graph in NetworkX format
    :param timeseries_dataset: timeseries data from sensors in the form of list of transactions
    """
    # linearize the KG properties, meaning that turn properties with the type list and object into plain key-value pairs
    for node_id in knowledge_graph.nodes:
        knowledge_graph.nodes[node_id]['properties'] = linearize(
            knowledge_graph.nodes[node_id]['properties'])

    # list of categorical and numerical attributes to consider (not all attributes are taken into account for now)
    categorical_attributes = ['type', 'measurement_aspect']
    numerical_attributes = ['diameter', 'elevation', 'id']

    # filter the kg properties, but keep the name as an identifier of the nodes which won't be used in the learning
    knowledge_graph = filter_knowledge_graph_props(
        knowledge_graph, categorical_attributes + numerical_attributes + ["name"])

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
    for transaction in timeseries_dataset:
        vector_categorical = []
        vector_numerical = []
        vector_sensor_data = []
        for item in transaction:
            sensor_id = item.split("_", 1)[1]
            measurement = item.split("_", 1)[0]
            node = knowledge_graph.nodes[sensor_id.replace('s_', '', 1)]
            neighbors = get_first_neighbor_with_relations(knowledge_graph, node)

            vector_sensor_data.append(float(measurement))
            vector_categorical += [node['properties'][key] for key in
                                   [key for key in node['properties'].keys() if
                                    key != 'name' and key.startswith("cat_")]]
            vector_numerical += [node['properties'][key] for key in
                                 [key for key in node['properties'].keys() if
                                  key != 'name' and not key.startswith("cat_")]]
            for neighbor in neighbors:
                vector_categorical += [neighbor['neighbor']['properties'][key] for key in
                                       [key for key in neighbor['neighbor']['properties'].keys() if
                                        key != 'name' and key.startswith("cat_") for neighbor in neighbors]]
                vector_numerical += [neighbor['neighbor']['properties'][key] for key in
                                     [key for key in neighbor['neighbor']['properties'].keys() if
                                      key != 'name' and not key.startswith("cat_") for neighbor in neighbors]]
        vector_list_sensor_data.append(vector_sensor_data)
        vector_list_categorical.append(vector_categorical)
        vector_list_numerical.append(vector_numerical)

    return {
        'categorical': vector_list_categorical,
        'numerical': vector_list_numerical,
        'sensor': vector_list_sensor_data
    }
