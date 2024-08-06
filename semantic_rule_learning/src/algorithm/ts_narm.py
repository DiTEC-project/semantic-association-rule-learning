from niaarm import get_rules, Dataset

from src.preprocessing.semantic_enrichment import *

import numpy as np
import pandas as pd


class TSNARM:
    """
    An implementation/adaptation of the TS-NARM from Fister et. al using NiaARM and NiaPy
    """

    def __init__(self, optimization_algorithm, max_evaluations=50000):
        self.max_evaluations = max_evaluations
        self.optimization_algorithm = optimization_algorithm

    def learn_rules(self, knowledge_graph, transactions):
        """
        Learn association rules using nature-inspired optimization-based methods from semantically enriched sensor data
        """
        enriched_transactions = enrich_transactions_tsnarm(knowledge_graph, transactions)
        metrics = ['support', 'confidence']

        frame = pd.DataFrame(enriched_transactions[1:], columns=enriched_transactions[0])
        dataset = Dataset(frame)
        rules, run_time = get_rules(dataset=dataset, algorithm=self.optimization_algorithm, metrics=metrics,
                                    max_evals=self.max_evaluations, logging=False)
        if len(rules) == 0:
            return False, False

        data_coverage = self.calculate_coverage(rules, enriched_transactions)
        support, confidence, rule_coverage, zhangs = \
            rules.mean("support"), rules.mean("confidence"), rules.mean("coverage"), rules.mean("zhang")
        # return 0 for training time
        return [len(rules), 0, run_time, support, confidence, rule_coverage, zhangs, data_coverage], rules

    @staticmethod
    def calculate_coverage(rules, dataset):
        """
        calculate coverage of the given rule set on the dataset
        """
        rule_coverage = np.zeros(len(dataset))
        data_headers = dataset[0]
        for index in range(1, len(dataset)):
            transaction = dataset[index]
            for rule in rules:
                covered = True
                for item in rule.antecedent:
                    if item.categories:
                        if transaction[data_headers.index(item.name)] != item.categories[0]:
                            covered = False
                            break
                    else:
                        covered = False
                        value = transaction[data_headers.index(item.name)]
                        if item.min_val <= value <= item.max_val:
                            covered = True
                            break
                if covered:
                    rule_coverage[index] = 1
                    break

        return sum(rule_coverage) / len(dataset)
