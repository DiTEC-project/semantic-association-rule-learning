"""
This class includes NetworkX specific graph db utility functions which are used for further processing
the extracted data from the graph database. All the functions interacting with the db itself directly
are placed under the "repository" package.
"""
import itertools

def get_first_neighbor_with_relations(graph, node):
    """
    Get first neighbors of a given node and the edges in between them
    :param graph:
    :param node:
    :return:
    """
    edges = list(graph.edges(node['properties']['id'], data=True))
    neighbors = []
    for (current_node, neighbor, edge_props) in edges:
        if not graph.nodes[neighbor]['properties']['id'].startswith('s_'):
            neighbors.append({
                'neighbor': graph.nodes[neighbor],
                'edge_props': edge_props
            })
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
        topology.append(
            node['labels'] + "_" + edge['neighbor']['labels'] + "_" +
            edge['edge_props']['type'])

    powerset = []
    for L in range(1, len(topology) + 1):
        for subset in itertools.combinations(topology, L):
            powerset.append(str(subset))

    return powerset


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
