import numpy as np


def create_vector_rep_node(node, node_vector_attribute_order):
    """
    create a vector representation for the given node based on the node types
    each vector contains space for all possible node types, and only the type which the given node corresponds
    to is filled (this means sparsity in the input), this is done to avoid permutation
    :param node: a node in the knowledge graph in the NetworkX format
    :param node_vector_attribute_order: represents the order of attributes in a vector that contains values for a node
    """
    values = []
    indices = []

    if node is None:
        # return a vector with all zeros
        for attribute in node_vector_attribute_order:
            values.append(0)
            indices.append(attribute)
        return values, indices

    for key in node_vector_attribute_order:
        indices.append(key)
        if key in node['properties'].keys():
            values.append(node['properties'][key])
        else:
            values.append(0)

    return values, indices


def create_vector_rep_node_type(node, node_vector_attribute_order):
    """
    create a vector representation for the node type attribute only
    this will be used to encode topology
    """
    values = []
    indices = []
    # return a vector with all zeros
    for key in node_vector_attribute_order:
        if key.startswith("type_") and key != 'type_Sensor':
            values.append(0)
            indices.append(key)

    if node is None:
        return values, indices

    node_type = [key for key in node['properties'].keys() if
                 key.startswith("type_") and node['properties'][key] == 1][0]
    values[indices.index(node_type)] = 1

    return values, indices


def create_vector_rep_measurement(sensor_measurement, boundaries, sensor_type):
    vector = []
    indices = []
    for type in boundaries:
        if type == 'label':
            continue
        partial_vector = np.zeros(len(boundaries[type]) - 1)
        indices += boundaries['label'][type]
        if type == sensor_type:
            for index in range(len(boundaries[type]) - 1):
                if boundaries[type][index] <= sensor_measurement <= boundaries[type][index + 1]:
                    partial_vector[index] = 1
                    break
        vector += partial_vector.tolist()

    return vector, indices


def get_category_boundaries(vector_category_tracker):
    """
    each input vector consists of one-hot encoded data for multiple categorical input
    at the last step of the AE, a softmax will be applied to each category individually
    therefore, this method returns the indices of input categories
    """
    indices = []
    start = 0
    for index in range(1, len(vector_category_tracker)):
        if vector_category_tracker[start].split('_')[0] != vector_category_tracker[index].split('_')[0]:
            indices.append({'start': start, 'end': index})
            start = index

    indices.append({'start': start, 'end': len(vector_category_tracker)})

    return indices
