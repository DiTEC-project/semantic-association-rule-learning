import time
import numpy as np
from mlxtend.frequent_patterns import association_rules, fpgrowth, hmine
from mlxtend.preprocessing import TransactionEncoder

from src.preprocessing.semantic_enrichment import *
from src.util.rule_quality import *


class NaiveSemRL:
    """
    Implementation of Naive SemRL from Karabulut et. al (2023), using MLxtend Python package
    """

    def __init__(self, min_support, min_confidence, num_bins, max_antecedent, algorithm):
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
        self.algorithm = algorithm
        self.rules = []

    def mine_rules(self, transactions):
        """
        Learn semantic association rules from discrete sensor data and knowledge graph using exhaustive ARM methods
        :param transactions: semantically enriched sensor data
        :return:
        """
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        start = time.time()

        # mine frequent items
        if self.algorithm == "fpgrowth":
            frq_items = fpgrowth(df, self.min_support, use_colnames=True, max_len=self.max_antecedent + 1)
        else:
            frq_items = hmine(df, self.min_support, use_colnames=True, max_len=self.max_antecedent + 1)
        if len(frq_items) == 0:
            return [], (time.time() - start), 0

        # create association rules
        self.rules = association_rules(frq_items, metric="confidence", min_threshold=self.min_confidence)
        execution_time = time.time() - start

        # from now on, format each rule in a way that is generic and compatible with the other approaches
        formatted_rules = []
        dataset_coverage = np.zeros(len(transactions))
        for i in range(len(self.rules["antecedents"])):
            consequent = list(self.rules["consequents"][i])[0]
            antecedents = list(self.rules["antecedents"][i])
            stats = self.calculate_stats(antecedents, consequent, transactions, dataset_coverage)

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
            new_consequent['measurement_aspect'] = consequent.split('_type_')[1].split('_end_')[0]
            new_consequent['measurement_range'] = consequent.split('_range_')[1].split('_end_')[0]

            # if the items in the consequent corresponds to one of the antecedents, then mark this with consequent_index
            # e.g. if a.feature1 & b.feature1 --> a.feature2, then antecedents = [a.feature1, b.feature1]
            # and consequent_index = 1
            new_rule = {'antecedents': antecedent_list, 'consequent': new_consequent, 'consequent_index': postfix,
                        "nonformatted_antecedents": self.rules["antecedents"][i],
                        "nonformatted_consequents": self.rules["consequents"][i]}
            new_rule.update(stats)
            formatted_rules.append(new_rule)

        self.rules = formatted_rules
        return formatted_rules, execution_time, dataset_coverage.sum() / len(transactions)

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

    @staticmethod
    def calculate_stats(antecedents, consequent, enriched_transactions, dataset_coverage):
        antecedents_occurrence_count = 0
        consequents_occurrence_count = 0
        co_occurrence_count = 0
        only_antecedence_occurrence_count = 0
        only_consequence_occurrence_count = 0
        no_ant_no_cons_count = 0
        for transaction_index in range(len(enriched_transactions)):
            transaction = enriched_transactions[transaction_index]
            antecedent_match = True
            for antecedent in antecedents:
                if antecedent not in transaction:
                    antecedent_match = False
                    break
            if antecedent_match:
                dataset_coverage[transaction_index] = 1
                antecedents_occurrence_count += 1
            if consequent in transaction:
                consequents_occurrence_count += 1
                if antecedent_match:
                    co_occurrence_count += 1
                else:
                    only_consequence_occurrence_count += 1
            elif antecedent_match:
                only_antecedence_occurrence_count += 1
            else:
                no_ant_no_cons_count += 1

        stats = {}
        num_transactions = len(enriched_transactions)

        support_body = antecedents_occurrence_count / num_transactions

        stats['support'] = co_occurrence_count / num_transactions
        stats['confidence'] = stats['support'] / support_body
        stats['coverage'] = support_body
        stats["yulesq"] = calculate_yulesq(co_occurrence_count, no_ant_no_cons_count,
                                           only_consequence_occurrence_count,
                                           only_antecedence_occurrence_count)
        stats["interestingness"] = calculate_interestingness(stats['confidence'], stats['support'],
                                                             (consequents_occurrence_count / num_transactions),
                                                             num_transactions)
        stats["zhangs_metric"] = calculate_zhangs_metric(stats["support"],
                                                         (antecedents_occurrence_count / num_transactions),
                                                         (consequents_occurrence_count / num_transactions))
        return stats
