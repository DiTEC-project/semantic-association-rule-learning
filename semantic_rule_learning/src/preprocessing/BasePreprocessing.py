"""
Copyright (C) 2023 University of Amsterdam
@author Erkan Karabulut – e.karabulut@uva.nl
@version 1.0
Preprocessing functions that are common across all algorithms implemented
"""
import itertools
import time

from src.repository.graphdb.NodeRepository import NodeRepository
from src.util.GraphUtil import *


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
