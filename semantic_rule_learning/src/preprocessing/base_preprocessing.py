"""
This Python script includes some common preprocessing functions
"""
from src.util.json_util import *

# list of categorical and numerical attributes to consider (not all attributes are taken into account for now)
# attribute names for water network datasets
categorical_attributes = ['type', 'measurement_aspect']
numerical_attributes = ['diameter', 'length', 'roughness', 'elevation']


# attribute names for lbnl dataset (only the 'type' attribute exists sadly :( )
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
        knowledge_graph.nodes[node_id]['properties'] = linearize(knowledge_graph.nodes[node_id]['properties'])

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
