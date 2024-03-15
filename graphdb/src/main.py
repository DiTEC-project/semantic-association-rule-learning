import time

import networkx
from neo4j import GraphDatabase
from util.json_util import *
from dotenv import load_dotenv
from repository.NodeRepository import NodeRepository
from repository.LinkRepository import LinkRepository

import json
import wntr
import os
import networkx as nx

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

if __name__ == '__main__':
    """
    This file reads semantics of the water network related use cases
    """
    file_name = '../../data/meta/LeakDB_Hanoi_CMH_Scenario-1.inp'
    # file_name = 'data/meta/GEN-09 Oosterbeek.inp'
    wn = wntr.network.WaterNetworkModel(file_name)
    wn_json = wn.to_dict()

    node_repository = NodeRepository()
    edge_repository = LinkRepository()

    node_repository.clean_up_db()
    node_repository.add_nodes(wn_json["nodes"])
    edge_repository.add_links(wn_json["links"])

    node_repository.close()
    edge_repository.close()
