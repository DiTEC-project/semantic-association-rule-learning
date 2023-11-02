"""
Copyright (C) 2023 University of Amsterdam
@author Erkan Karabulut – e.karabulut@uva.nl
@version 1.0
FP-Growth algorithm implementation using mlxtend package
"""

from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from src.util.GraphUtil import *
from src.util.TransactionsUtil import *
from src.preprocessing.BasePreprocessing import *
from itertools import groupby
import pandas as pd


class NaivestSemRL:
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

        # finding the longest and shortest rules
        # longest_rule_index = 0
        # shortest_rule_index = 0
        # longest_rule_length = 0
        # shortest_rule_length = 9999
        # for index in range(len(rules["antecedents"])):
        #     rule_length = len(rules["antecedents"][index]) + len(rules["antecedents"][index])
        #     if rule_length > longest_rule_length:
        #         longest_rule_length = rule_length
        #         longest_rule_index = index
        #     if rule_length < shortest_rule_length:
        #         shortest_rule_length = rule_length
        #         shortest_rule_index = index
        # print("Longest rule: ", rules["antecedents"][longest_rule_index], " -> ", rules["consequents"][longest_rule_index], ". Confidence: ", rules["confidence"][longest_rule_index])
        # print("Shortest rule: ", rules["antecedents"][shortest_rule_index], " -> ", rules["consequents"][shortest_rule_index], ". Confidence: ", rules["confidence"][shortest_rule_index])

        formatted_rules = []
        for i in range(len(rules["antecedents"])):
            formatted_rules.append({
                "antecedents": rules["antecedents"][i],
                "consequents": rules["consequents"][i],
            })

        return formatted_rules

    def learn_semantic_association_rules(self, knowledge_graph, disc_hist_time_series):
        """
        Learn semantic association rules from discretized historical time series data and knowledge graph
        :param knowledge_graph: knowledge graph in NetworkX format
        :param disc_hist_time_series: discretized sensor measurements as list of transactions
        :return:
        """
        # initial_assoc_rules = self.fp_growth(disc_hist_time_series)

        enriched_transactions = []
        for transaction in disc_hist_time_series:
            new_transaction = []
            # new_transaction += transaction
            for item in transaction:
                sensor_id = item.split("_", 1)[1]
                measurement = item.split("_", 1)[0]
                sensor_type = knowledge_graph.nodes[sensor_id]['properties']['type']
                node = knowledge_graph.nodes[sensor_id.replace('s_', '', 1)]
                current_node_attributes = [('s_' + str(value)) for value in node['properties'].values()]
                neighbors = get_first_neighbor_with_relations(knowledge_graph, node)
                topology = get_topology(node, neighbors)
                # neighbors_attributes = get_attributes([neighbors])

                for attribute in topology:
                    new_transaction.append(measurement + "___" + sensor_type + "___" + attribute)
            enriched_transactions.append(new_transaction)

        assoc_rules = self.fp_growth(enriched_transactions)

        return assoc_rules
