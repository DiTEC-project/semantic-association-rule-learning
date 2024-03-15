"""
This class includes NetworkX specific graph db utility functions which are used for further processing
the extracted data from the graph database. All the functions interacting with the db itself directly
are placed under the "repository" package.
"""
import numpy as np


def get_neighbors(graph, node, num_of_neighbors):
    """
    Get first $num_of_neighbors neighbors of a given node and the edges in between them
    :param graph:
    :param node:
    :param num_of_neighbors:
    :return:
    """
    list_of_nodes_to_scan = [node['properties']['name']]
    for pointer in range(num_of_neighbors):
        for pointer2 in range(len(list_of_nodes_to_scan)):
            node_name = list_of_nodes_to_scan[pointer2]
            if not node_name.startswith('s_'):
                list_of_nodes_to_scan += [node_name[1] + "--" + str(pointer) for node_name in
                                          list(graph.edges(node_name, data=True))]

    unique_neighbor_name_list = np.unique(list_of_nodes_to_scan)
    neighbors = {}
    unique_neighbor_name_list = list(unique_neighbor_name_list)
    unique_neighbor_name_list.remove(node['properties']['name'])
    for node_name in unique_neighbor_name_list:
        if not node_name.startswith('s_'):
            split = node_name.split("--")
            neighbor_degree = split[1]
            node_name = split[0]
            if neighbor_degree not in neighbors:
                neighbors[neighbor_degree] = []
            neighbors[neighbor_degree].append(graph.nodes[node_name])
    return neighbors


def get_topology(node, neighbor_list):
    """
    Returns node_type -- edge_type -- neighbor_node_type lists for the given neighbor list
    :param node: a node in the KG
    :param neighbor_list: list of neighbors for the given node in the form of {neighbor, edge_props} list format
    :return: Topology of the given node and its neighbors in the form of list of "source-type_edge-type_destination_type"
     strings
    """
    topology = []
    for edge in neighbor_list:
        relation = node['labels'] + "_" + edge['neighbor']['labels'] + "_" + edge['edge_props']['type']
        index = 2
        if relation in topology:
            temp = relation
            while relation in topology:
                relation = temp + '_' + str(index)
                index += 1
            topology.append(relation)
        else:
            topology.append(relation)

    return topology


def get_attributes(subgraph):
    """
    Returns a list of attributes of the nodes in the subgraph
    :param subgraph: A subgraph in the form of {source, destination, edge_props} list format
    :return:
    """
    attributes = []
    unique_ids = []
    for edge_list in subgraph:
        for edge in edge_list:
            if edge['neighbor']['properties']['id'] not in unique_ids:
                attributes += [('d_' + str(value)) for value in edge['neighbor']['properties'].values()]
                unique_ids.append(edge['neighbor']['properties']['id'])

    return attributes


def get_unique_values(graph, attribute):
    """
    Returns a list of unique values in the graph, for a given attribute
    :param graph: a graph in NetworkX format
    :para attribute: key of an attribute in the graph
    :return: a lit of unique values for the given attribute in the graph
    """
    unique_values = []
    for node_id in graph.nodes:
        if attribute in graph.nodes[node_id]['properties']:
            if graph.nodes[node_id]['properties'][attribute] not in unique_values:
                unique_values.append(graph.nodes[node_id]['properties'][attribute])

    return unique_values


def discretize_numerical_attributes(knowledge_graph, numerical_attribute_list, num_bins):
    """
    discretize the numerical attributes inside the knowledge graph based on equal-frequency binning method
    :param knowledge_graph: a knowledge graph in NetworkX format
    :param numerical_attribute_list: list of numerical property keys in string
    :param num_bins: number of bins to discretize the numerical attributes in
    :return: the same knowledge graph with discrete (range) numeric values instead of continuous values
    """
    # initialize empty lists per numerical values
    numerical_value_map = {}
    for attribute in numerical_attribute_list:
        numerical_value_map[attribute] = []

    # collect all numerical values per attribute
    for node_id in knowledge_graph.nodes:
        for attribute in numerical_attribute_list:
            if attribute in knowledge_graph.nodes[node_id]['properties']:
                numerical_value_map[attribute].append(knowledge_graph.nodes[node_id]['properties'][attribute])

    for numerical_attribute in numerical_value_map:
        # sort values in increasing order
        sorted_values = np.sort(numerical_value_map[numerical_attribute])
        # define boundaries based on num_bins
        boundaries = np.interp(np.linspace(0, len(sorted_values), num_bins + 1),
                               np.arange(len(sorted_values)),
                               sorted_values)

        # assign each numerical value of the numerical_attribute in the knowledge_graph to one of the boundary sets
        for node_id in knowledge_graph.nodes:
            if numerical_attribute in knowledge_graph.nodes[node_id]['properties']:
                attr_value = knowledge_graph.nodes[node_id]['properties'][numerical_attribute]
                for i in range(len(boundaries) - 1):
                    if boundaries[i] <= attr_value <= boundaries[i + 1]:
                        knowledge_graph.nodes[node_id]['properties'][numerical_attribute] = \
                            str(boundaries[i]) + "_" + str(boundaries[i + 1])
                        break
    return knowledge_graph
