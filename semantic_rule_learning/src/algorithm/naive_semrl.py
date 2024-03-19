import time
import pandas as pd

from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from src.preprocessing.semantic_enrichment import *


class NaiveSemRL:
    """
    Implementation Naive SemRL semantic association rule mining from discrete time series data and knowledge graphs
     based on FP-Growth algorithm from mlxtend package
    """

    def __init__(self, min_support, min_confidence, num_bins, max_antecedent):
        """
        Initialize algorithm parameters
        :param min_support:
        :param min_confidence:
        :param num_bins: number of bins to discretize numerical values into
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.num_bins = num_bins
        self.max_antecedent = max_antecedent
        self.rules = []

    def fp_growth(self, knowledge_graph, transactions):
        """
        Learn semantic association rules from discrete historical time series data and knowledge graph using FP-Growth
        :param knowledge_graph: knowledge graph in NetworkX format
        :param transactions: discrete time-series sensor data
        :return:
        """
        enriched_transactions = enrich_transactions_fpgrowth(knowledge_graph, transactions, self.num_bins)

        te = TransactionEncoder()
        te_ary = te.fit(enriched_transactions).transform(enriched_transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        start = time.time()
        frq_items = fpgrowth(df, self.min_support, use_colnames=True)
        if len(frq_items) == 0:
            return [], (time.time() - start)

        self.rules = association_rules(frq_items, metric="confidence", min_threshold=self.min_confidence)
        execution_time = time.time() - start
        # calculate antecedent length for each of the rule, to be used for filtering based on self.max_antecedent
        self.rules["antecedent_len"] = self.rules["antecedents"].apply(lambda x: len(x))
        self.rules["consequent_len"] = self.rules["consequents"].apply(lambda x: len(x))

        # from now on, format each rule in a way that is generic and compatible with the AE SemRL approach
        formatted_rules = []
        for i in range(len(self.rules["antecedents"])):
            # filter rules based on the number of antecedents
            if self.rules['antecedent_len'][i] > self.max_antecedent:
                continue
            # association_rules finds rules with trivial consequences, such as: a->b, a->c, a->b&c.
            # here a->b&c is trivial, any rule with len(consequents) > 1 is trivial.
            if self.rules['consequent_len'][i] > 1:
                continue

            consequent = list(self.rules["consequents"][i])[0]
            antecedents = list(self.rules["antecedents"][i])
            stats = self.calculate_stats(antecedents, consequent, enriched_transactions)

            groups = {}
            unique_item_list = []
            item_index = -1
            for antecedent in antecedents:
                measurement_range = antecedent.split('_range_')[1].split('_end_')[0]
                type = antecedent.split('_type_')[1].split('_end_')[0]
                if type + "-" + measurement_range not in unique_item_list:
                    unique_item_list.append(type + "-" + measurement_range)
                    item_index += 1
                    groups[str(item_index)] = []

                groups[str(item_index)].append(antecedent)

            antecedent_list = []
            for key in groups:
                new_antecedent = {}
                for sub_item in groups[key]:
                    if "_attribute_" in sub_item:
                        attribute = sub_item.split('_attribute_')[1]
                        key = attribute.split('_key_')[1].split('_end_')[0]
                        value = attribute.split('_value_')[1].split('_end_')[0]
                        new_antecedent[key] = value
                    else:
                        new_antecedent['measurement_aspect'] = sub_item.split('_type_')[1].split('_end_')[0]
                        new_antecedent['measurement_range'] = sub_item.split('_range_')[1].split('_end_')[0]

                antecedent_list.append(new_antecedent)

            new_consequent = {}
            measurement_range = consequent.split("_range_")[1].split('_end_')[0]
            type = consequent.split("_type_")[1].split('_end_')[0]
            if type + "-" + measurement_range in unique_item_list:
                postfix = unique_item_list.index(type + "-" + measurement_range)
            else:
                postfix = item_index + 1
            if "_attribute_" in consequent:
                attribute = consequent.split('_attribute_')[1]
                key = attribute.split('_key_')[1].split('_end_')[0]
                value = attribute.split('_value_')[1].split('_end_')[0]
                new_consequent[key] = value
            else:
                new_consequent['measurement_aspect'] = consequent.split('_type_')[1].split('_end_')[0]
                new_consequent['measurement_range'] = consequent.split('_range_')[1].split('_end_')[0]

            # if the items in the consequent corresponds to one of the antecedents, then mark this with consequent_index
            # e.g. if a.feature1 & b.feature1 --> a.feature2, then antecedents = [a.feature1, b.feature1]
            # and consequent_index = 1
            new_rule = {
                'antecedents': antecedent_list,
                'consequent': new_consequent,
                'consequent_index': postfix
            }
            new_rule.update(stats)
            formatted_rules.append(new_rule)

        self.rules = formatted_rules
        return formatted_rules, execution_time

    @staticmethod
    def deconstruct_rule_string(rule_string, postfix):
        split_string = rule_string.split("_")
        sensor_type = split_string[1]
        measurement_range = '_'.join(split_string[2:4])

        rule = {
            'type' + postfix: sensor_type,
            'measurement_range' + postfix: measurement_range
        }
        if len(split_string) > 4:
            rule['attribute' + postfix] = '_'.join(split_string[4:]).replace('s_', '')

        return rule

    def calculate_stats(self, antecedents, consequent, enriched_transactions):
        antecedents_occurrence_count = 0
        consequents_occurrence_count = 0
        co_occurrence_count = 0
        only_antecedence_occurrence_count = 0
        only_consequence_occurrence_count = 0
        for transaction_index in range(len(enriched_transactions)):
            transaction = enriched_transactions[transaction_index]
            antecedent_match = True
            for antecedent in antecedents:
                if antecedent not in transaction:
                    antecedent_match = False
                    break
            if antecedent_match:
                antecedents_occurrence_count += 1
            if consequent in transaction:
                consequents_occurrence_count += 1
                if antecedent_match:
                    co_occurrence_count += 1
                else:
                    only_consequence_occurrence_count += 1
            elif antecedent_match:
                only_antecedence_occurrence_count += 1

        stats = {}
        num_transactions = len(enriched_transactions)

        support_body = antecedents_occurrence_count / num_transactions
        support_head = consequents_occurrence_count / num_transactions
        conf_not_body_and_head = only_consequence_occurrence_count / num_transactions

        stats['support'] = co_occurrence_count / num_transactions
        stats['confidence'] = stats['support'] / support_body
        stats['lift'] = stats['confidence'] / support_head
        stats['leverage'] = stats['support'] - (support_body * support_head)
        stats['zhangs_metric'] = (stats['confidence'] - conf_not_body_and_head) / max(stats['confidence'],
                                                                                      conf_not_body_and_head)
        return stats
