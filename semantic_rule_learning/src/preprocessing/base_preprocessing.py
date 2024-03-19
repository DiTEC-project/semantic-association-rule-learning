from src.util.vector_util import *
from src.util.json_util import *
from src.util.transactions_util import *

# list of categorical and numerical attributes to consider (not all attributes are taken into account for now)
# attribute names for water network datasets
categorical_attributes = ['type', 'measurement_aspect']
numerical_attributes = ['diameter', 'length']

# attribute names for lbnl dataset (only the 'type' attribute exists :( )
# categorical_attributes = ['type']
# numerical_attributes = []


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


def get_transactions_as_cat_vectors(knowledge_graph, transactions, num_bins, num_neighbors):
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

    for transaction in transactions:
        vector_tracker = []
        vector = []
        for index in range(len(transaction)):
            item = transaction[index]
            sensor_id = item.split("_name_", 1)[1].split('_end_')[0]
            measurement = float(item.split("_", 1)[0])
            sensor_type = item.split("_type_")[1].split('_end_')[0]
            node = knowledge_graph.nodes[list(knowledge_graph.neighbors(sensor_id))[0]]

            postfix = "_item_" + str(index)

            neighbors = get_neighbors(knowledge_graph, node, num_neighbors)
            for neighbor_degree in neighbors.keys():
                for neighbor_index in range(len(neighbors[neighbor_degree])):
                    values, indices = create_vector_rep_node(neighbors[neighbor_degree][neighbor_index],
                                                             "--" + str(neighbor_degree) + "--" + str(
                                                                 neighbor_index) + "--" + postfix)
                    vector += values
                    vector_tracker += indices

            values, indices = create_vector_rep_measurement(measurement, boundaries, sensor_type,
                                                            postfix)
            vector += values
            vector_tracker += indices

            values, indices = create_vector_rep_node(node, postfix)
            vector += values
            vector_tracker += indices

        vector_list.append(vector)
        vector_tracker_list.append(vector_tracker)

    # track where in a vector properties of each item starts and end
    item_boundaries = []
    start, end = 0, 0
    for index in range(len(transactions[0])):
        item = transactions[0][index]
        sensor_id = item.split("_name_", 1)[1].split('_end_')[0]
        measurement = float(item.split("_", 1)[0])
        sensor_type = item.split("_type_")[1].split('_end_')[0]
        node = knowledge_graph.nodes[list(knowledge_graph.neighbors(sensor_id))[0]]
        # neighbors = get_first_neighbor_with_relations(knowledge_graph, node)

        # todo: topology is not encoded explicitly, is it necessary? if yes, how do we do that?
        postfix = "_item_" + str(index)
        values, indices = create_vector_rep_measurement(measurement, boundaries, sensor_type,
                                                        postfix)
        end += len(values)
        values, indices = create_vector_rep_node(node, postfix)
        end += len(values)
        item_boundaries.append({'start': start, 'end': end})
        start = end

    return {
        'vector_list': vector_list,
        'vector_tracker_list': vector_tracker_list,
        'item_boundaries': item_boundaries
    }
