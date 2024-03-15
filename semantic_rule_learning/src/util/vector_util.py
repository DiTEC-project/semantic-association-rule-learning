import numpy as np


def create_vector_rep_node(node, postfix):
    """
    create a vector representation for the given node based on the node types
    each vector contains space for all possible node types, and only the type which the given node corresponds
    to is filled (this means sparsity in the input), this is done to avoid permutation
    :param node: a node in the knowledge graph in the NetworkX format
    :param postfix:
    """
    values = []
    indices = []

    for key in node['properties'].keys():
        if key == 'name':
            continue
        indices.append(key + postfix)
        values.append(node['properties'][key])

    return values, indices


def create_vector_rep_measurement(sensor_measurement, boundaries, sensor_type, postfix):
    vector = []
    indices = []
    partial_vector = np.zeros(len(boundaries[sensor_type]) - 1)
    indices += boundaries['label'][sensor_type]
    for index in range(len(indices)):
        indices[index] = indices[index] + postfix
    for index in range(len(boundaries[sensor_type]) - 1):
        if boundaries[sensor_type][index] <= sensor_measurement <= boundaries[sensor_type][index + 1]:
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
    boundaries_per_transaction = []
    for category_vector in vector_category_tracker:
        indices = []
        start = 0
        for index in range(1, len(category_vector)):
            if category_vector[start].split('_')[0] != category_vector[index].split('_')[0]:
                indices.append({'start': start, 'end': index})
                start = index

        indices.append({'start': start, 'end': len(category_vector)})
        boundaries_per_transaction.append(indices)

    return boundaries_per_transaction
