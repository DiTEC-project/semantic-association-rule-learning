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

    def __init__(self, min_support, min_confidence):
        """
        Initialize algorithm parameters
        :param min_support:
        :param min_confidence:
        """
        self.min_support = min_support
        self.min_confidence = min_confidence

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
            formatted_rules.append({
                "antecedents": rules["antecedents"][i],
                "consequents": rules["consequents"][i],
            })

        return formatted_rules

    def learn_semantic_association_rules(self, knowledge_graph, disc_time_period):
        """
        Learn semantic association rules from discretized historical time series data and knowledge graph
        :param knowledge_graph: knowledge graph in NetworkX format
        :param disc_time_period: discretization time period in minutes, the sensor data will be aggregated based on this
        :return:
        """
        enriched_transactions = naive_semrl_enrich_transactions(knowledge_graph, disc_time_period)
        assoc_rules = self.fp_growth(enriched_transactions)

        return assoc_rules
