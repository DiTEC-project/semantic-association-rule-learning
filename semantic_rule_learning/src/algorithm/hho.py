from niaarm import get_rules, Dataset
from niapy.algorithms.basic import HarrisHawksOptimization

from src.preprocessing.semantic_enrichment import *

import pandas as pd


class HHO:
    """
    An implementation of optimization-based (HarrisHawksOptimization algorithm) ARM using niaarm package
    """
    def __init__(self, population_size=50, max_iterations=30):
        self.population_size = population_size
        self.max_iterations = max_iterations

    def learn_rules(self, knowledge_graph, transactions):
        enriched_transactions = enrich_transactions_hho(knowledge_graph, transactions)
        algo = HarrisHawksOptimization(self.population_size)
        metrics = ['confidence']

        frame = pd.DataFrame(enriched_transactions[1:], columns=enriched_transactions[0])
        dataset = Dataset(frame)
        rules, run_time = get_rules(dataset, algo, metrics, self.max_iterations, logging=False)

        return rules, run_time
