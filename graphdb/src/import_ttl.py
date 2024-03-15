import os
import json
from dotenv import load_dotenv
from rdflib import Graph
from src.repository.NodeRepository import NodeRepository
from src.repository.LinkRepository import LinkRepository

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

if __name__ == '__main__':
    node_repository = NodeRepository()
    edge_repository = LinkRepository()

    graph = Graph()
    graph.parse('../../data/meta/LBNL_FDD_Data_Sets_FCU.ttl', format='n3')

    full_object_names = {}
    for statement in graph:
        subject = str(statement[0]).split("/")[-1].split('#')[-1]
        predicate = str(statement[1]).split("/")[-1].split('#')[-1]
        object = str(statement[2]).split("/")[-1].split('#')[-1]

        if subject != 'fcu':
            object = subject + "_" + object

        node_repository.create_node(subject, {'name': subject, 'type': subject})
        node_repository.create_node(object, {'name': object, 'type': subject})
        edge_repository.create_edge(predicate, subject, object)

    file = open("../../data/meta/lbnl_binding.json")
    binding = json.load(file)
    for key in binding:
        node_repository.mark_sensors(key, binding[key], "Sensor")
    file.close()
