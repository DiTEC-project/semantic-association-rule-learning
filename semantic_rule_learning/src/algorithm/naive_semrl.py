"""
Copyright (C) 2023 University of Amsterdam
@author Erkan Karabulut – e.karabulut@uva.nl
@version 1.0
FP-Growth algorithm implementation using mlxtend package
"""

from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from src.preprocessing.semantic_enrichment import *
import pandas as pd


class NaiveSemRL:
    """
    FP-Growth algorithm implementation based on mlxtend package
    """

    def __init__(self, min_support, min_confidence, num_bins):
        """
        Initialize algorithm parameters
        :param min_support:
        :param min_confidence:
        :param num_bins: number of bins to discretize numerical values into
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.num_bins = num_bins

    def fp_growth(self, transactions):
        """
        Extract association rules using fpgrowth algorithm
        :param transactions:
        :return:
        """
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        frq_items = fpgrowth(df, self.min_support, use_colnames=True)
        if len(frq_items) == 0:
            return None

        rules = association_rules(frq_items, metric="confidence", min_threshold=self.min_confidence)

        formatted_rules = []
        for i in range(len(rules["antecedents"])):
            consequent_set = list(rules["consequents"][i])
            antecedents = list(rules["antecedents"][i])
            if not (len(consequent_set) == 1 and consequent_set[0].startswith('sensor') and
                    len(consequent_set[0].split("_")) == 3):
                continue

            antecedent_list = []
            for antecedent in antecedents:
                antecedent_list.append(self.deconstruct_rule_string(antecedent))

            formatted_rules.append(
                {'antecedent': antecedent_list, 'consequent': self.deconstruct_rule_string(consequent_set[0])}
            )

        return formatted_rules

    @staticmethod
    def deconstruct_rule_string(rule_string):
        split_string = rule_string.split("_")
        sensor_type = split_string[1]
        measurement_range = split_string[2]

        rule = {
            'type': sensor_type,
            'measurement_range': measurement_range
        }
        if len(split_string) > 3:
            rule['attribute'] = '_'.join(split_string[3:])

        return rule

    def learn_semantic_association_rules(self, knowledge_graph, transactions):
        """
        Learn semantic association rules from discretized historical time series data and knowledge graph
        :param knowledge_graph: knowledge graph in NetworkX format
        :param transactions: discrete time-series sensor data
        :return:
        """
        enriched_transactions = naive_semrl_enrich_transactions(knowledge_graph, transactions, self.num_bins)
        assoc_rules = self.fp_growth(enriched_transactions)

        return assoc_rules
