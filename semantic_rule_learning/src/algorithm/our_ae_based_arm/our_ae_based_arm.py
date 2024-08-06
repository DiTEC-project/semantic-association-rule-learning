import random
import time
import numpy as np
import torch

from itertools import chain, combinations
from torch import nn
from src.algorithm.our_ae_based_arm.autoencoder import AutoEncoder
from src.preprocessing.semantic_enrichment import *
from src.util.rule_quality import *


class OurAEBasedARM:
    """
    Implementation of our Autoencoder-based (AE-based) ARM method as part of the CHARM pipeline
    """

    def __init__(self, num_bins=10, num_neighbors=1, max_antecedents=2, similarity_threshold=0.8, noise_factor=0.5):
        """
        @param num_bins: number of bins to discretize numerical data into
        @param num_neighbors: number of neighbors to consider when enriching time series data with semantics
        @param noise_factor: amount of noise introduced for the one-hot encoded input of denoising Autoencoder
        @param similarity_threshold: feature similarity threshold
        @param max_antecedents: maximum number of antecedents that the learned rules will have
        """
        self.training_time = 0
        self.noise_factor = noise_factor
        self.num_bins = num_bins
        self.num_neighbors = num_neighbors
        self.similarity_threshold = similarity_threshold
        self.max_antecedents = max_antecedents

        self.model = None
        self.input_vectors = None
        self.softmax = nn.Softmax(dim=0)

    def create_input_vectors(self, knowledge_graph, transactions):
        """
        semantically enrich the given transactions using the knowledge graph, and apply one-hot encoding
        @param knowledge_graph: knowledge graph
        @param transactions: discrete sensor measurements in the form of transactions
        """
        # get input vectors in the form of one-hot encoded vectors
        self.input_vectors = semantic_enrichment_our_ae_based_arm(knowledge_graph, transactions, self.num_bins,
                                                                  self.num_neighbors)

    def generate_rules(self):
        """
        Extract association rules from the Autoencoder
        """
        association_rules = []
        input_vector_size = len(self.input_vectors['vector_tracker_list'][0])

        start = time.time()
        # feature combinations to be tested based on the self.max_antecedents parameter
        feature_combinations = list(chain.from_iterable(
            combinations(self.input_vectors['category_indices'][0], r) for r in range(self.max_antecedents + 1)))
        for category_list in feature_combinations[1:]:
            # create a vector with equal probabilities per feature class values
            unmarked_features = self.initialize_input_vectors(input_vector_size,
                                                              self.input_vectors["category_indices"][0], category_list)
            # mark feature class values in category_list in the unmarked_features
            test_vectors = self.mark_features(unmarked_features, list(category_list))
            for test_vector in test_vectors:
                # marked features are the candidate antecedents
                candidate_antecedents = [index for index, value in enumerate(test_vector) if value == 1]
                # perform a forward run on the trained Autoencoder
                implication_probabilities = self.model(torch.FloatTensor(test_vector),
                                                       self.input_vectors['category_indices'][0]) \
                    .detach().numpy().tolist()
                # make sure that the marked features have higher output probability than the similarity threshold
                high_support = True
                for ant in candidate_antecedents:
                    if implication_probabilities[ant] < self.similarity_threshold:
                        high_support = False
                        break
                if not high_support:
                    continue
                # go through the output probabilities (implication_probabilities) and check if they have higher
                # probability then the given similarity threshold, except the candidate antecedents to prevent
                # self implication
                consequent_list = []
                for prob_index in range(len(implication_probabilities)):
                    if prob_index not in candidate_antecedents:
                        # store the feature class values with high output probability
                        if implication_probabilities[prob_index] >= self.similarity_threshold:
                            consequent_list.append(prob_index)
                if len(consequent_list) > 0:
                    # format the rule based indices in consequent_list and candidate_antecedents list
                    new_rule = self.get_rule(candidate_antecedents, consequent_list)
                    # form rules one by one making sure each rule has one item in the consequent
                    # because p -> q ∧ r is equal to p -> q AND p -> r anyways
                    for consequent in new_rule['consequents']:
                        # Not used in the AE-based ARM evaluation, but for feature use cases accept only rules with
                        # dynamic values (sensor measurements) in the consequent part as they are more interesting
                        # if "_range_" in consequent:
                        association_rules.append({'antecedents': new_rule['antecedents'], 'consequent': consequent})
        execution_time = time.time() - start
        return association_rules, execution_time, self.training_time

    @staticmethod
    def initialize_input_vectors(input_vector_size, categories, marked_categories) -> list:
        """
        initialize an input vector with all equal probabilities
        @param input_vector_size: input vector size
        @param categories: list of features and their start and end indices in the input vector
        @param marked_categories: features to be marked so that we don't initialize them with equal probs
        """
        vector_with_unmarked_features = np.zeros(input_vector_size)
        for category in categories:
            if category not in marked_categories:
                # assign equal probabilities
                vector_with_unmarked_features[category['start']:category['end']] = 1 / (
                        category['end'] - category['start'])
        return list(vector_with_unmarked_features)

    def mark_features(self, unmarked_test_vector, features, test_vectors=[]):
        """
        Create a list of test vectors by marking the given features in the unmarked test vector
        Marking is done by assigning the probability of 1 (100%)
        @param unmarked_test_vector: vector with equal probabilities
        @param features: features to be marked
        @param test_vectors: existing test vectors
        """
        # Features are recursively marked. For instance, if features f1, f2, and f3 will be marked, then first
        # mark class values of f1 in separate vectors (test_vectors list version one), and then do a recursive call
        # and mark the features of f2 inside test_vectors and then repeat for f3. If f1, f2, and f3 have 3 possible
        # class values each, then the end result will contain 3x3x3=27 test vectors.
        if len(features) == 0:
            return test_vectors
        feature = features.pop()
        new_test_vectors = []
        for i in range(feature['end'] - feature['start']):
            if len(test_vectors) > 0:
                # for each of the existing test vectors, mark
                for vector in test_vectors:
                    new_vector = vector.copy()
                    # mark indices of class values in "feature" in the existing list of test vectors
                    new_vector[feature['start'] + i] = 1
                    new_test_vectors.append(new_vector)
            else:
                new_vector = unmarked_test_vector.copy()
                # mark indices of class values in "feature" in the new test vector
                new_vector[feature['start'] + i] = 1
                new_test_vectors.append(new_vector)
        #
        return self.mark_features(unmarked_test_vector, features, new_test_vectors)

    def reformat_rules(self, association_rules):
        """
        convert given association rules from vector format to text
        """
        for rule_index in range(len(association_rules)):
            rule = association_rules[rule_index]
            deconstructed_rule = self.get_deconstructed_rule(rule['antecedents'], rule['consequent'])
            association_rules[rule_index]['antecedents'] = deconstructed_rule['antecedents']
            association_rules[rule_index]['consequent'] = deconstructed_rule['consequent']
            association_rules[rule_index]['consequent_index'] = deconstructed_rule['consequent_index']
        return association_rules

    def calculate_stats(self, rules, transactions):
        """
        calculate rule quality stats for the given set of rules based on the input transactions
        """
        dataset_coverage = np.zeros(len(transactions))
        for rule_index in range(len(rules)):
            rule = rules[rule_index]
            antecedents_occurrence_count = 0
            consequents_occurrence_count = 0
            co_occurrence_count = 0
            only_antecedence_occurrence_count = 0
            only_consequence_occurrence_count = 0
            no_ant_no_cons_count = 0
            for index in range(len(self.input_vectors['vector_list'])):
                encoded_transaction = self.input_vectors['vector_list'][index]
                antecedent_match = True
                for antecedent in rule['antecedents']:
                    if encoded_transaction[self.input_vectors['vector_tracker_list'][index].index(antecedent)] == 0:
                        antecedent_match = False
                        break
                if antecedent_match:
                    dataset_coverage[index] = 1
                    antecedents_occurrence_count += 1
                if encoded_transaction[self.input_vectors['vector_tracker_list'][index].
                        index(rule['consequent'])] == 1:
                    consequents_occurrence_count += 1
                    if antecedent_match:
                        co_occurrence_count += 1
                    else:
                        only_consequence_occurrence_count += 1
                elif antecedent_match:
                    only_antecedence_occurrence_count += 1
                else:
                    no_ant_no_cons_count += 1

            num_transactions = len(transactions)
            support_body = antecedents_occurrence_count / num_transactions

            rule['support'] = co_occurrence_count / num_transactions
            rule['confidence'] = rule['support'] / support_body if support_body != 0 else 0
            rule['coverage'] = antecedents_occurrence_count / num_transactions
            rule["zhangs_metric"] = calculate_zhangs_metric(rule["support"],
                                                            (antecedents_occurrence_count / num_transactions),
                                                            (consequents_occurrence_count / num_transactions))

        return rules, dataset_coverage.sum() / len(transactions)

    @staticmethod
    def initialize_input_vector(input_vector_size, categories, exceptions) -> list:
        """
        Initialize an input vector with all equal probabilities per feature class values
        """
        input_vector = np.zeros(input_vector_size)
        for category in categories:
            if category not in exceptions:
                input_vector[category['start']:category['end']] = 1 / (category['end'] - category['start'])
        return list(input_vector)

    def get_rule(self, antecedents, consequents):
        """
        find the string form of a given rule in vector form
        @param antecedents: antecedents of th rule
        @param consequents: consequents of the rule
        """
        rule = {'antecedents': [], 'consequents': []}
        for antecedent in antecedents:
            rule['antecedents'].append(self.input_vectors['vector_tracker_list'][0][antecedent])

        for consequent in consequents:
            rule['consequents'].append(self.input_vectors['vector_tracker_list'][0][consequent])

        return rule

    @staticmethod
    def get_deconstructed_rule(antecedents, consequent):
        """
        convert rules in string form in the vector form back into string form
        """
        rule = {'antecedents': [], 'consequent': None}
        groups = {}
        unique_item_list = []
        item_index = -1
        for antecedent in antecedents:
            # item indices (items refers a sensor measurement together with associated semantics) are marked with
            # "_item_" string
            split_antecedent = antecedent.split('_item_')
            if split_antecedent[1] not in unique_item_list:
                unique_item_list.append(split_antecedent[1])
                item_index += 1
                groups[str(item_index)] = []
            groups[str(item_index)].append(split_antecedent[0])

        for key in groups:
            antecedent = {}
            for sub_item in groups[key]:
                if sub_item.startswith('sensor'):
                    # measurement aspect (e.g. water pressure, water flow rate) is stored in measurement_aspect
                    # property, while its range is stored in "measurement_range" property, which are marked in the
                    # vector_tracker_list with "_type_" and "_end_" keys.
                    antecedent['measurement_aspect'] = sub_item.split('_type_')[1].split('_end_')[0]
                    antecedent['measurement_range'] = sub_item.split('_range_')[1].split('_end_')[0]
                else:
                    split_item = sub_item.split('_')
                    antecedent[split_item[0]] = '_'.join(split_item[1:])

            rule['antecedents'].append(antecedent)

        split_consequent = consequent.split('_item_')
        if split_consequent[1] in groups:
            postfix = int(split_consequent[1])
        else:
            postfix = item_index + 1

        formatted_consequent = {}
        if consequent.startswith('sensor'):
            formatted_consequent['measurement_aspect'] = consequent.split('_type_')[1].split('_end_')[0]
            formatted_consequent['measurement_range'] = consequent.split('_range_')[1].split('_end_')[0]
        else:
            split_item = split_consequent[0].split("_")
            formatted_consequent[split_item[0]] = '_'.join(split_item[1:])

        # if the items in the consequent corresponds to one of the antecedents, then mark this with consequent_index
        # e.g. if a.feature1 & b.feature1 --> a.feature2, then antecedents = [a.feature1, b.feature1]
        # and consequent_index = 1
        rule['consequent_index'] = postfix
        rule['consequent'] = formatted_consequent

        return rule

    def train(self, model):
        """
        train the autoencoder
        """
        self.model = AutoEncoder(max(len(x) for x in self.input_vectors['vector_list']))

        if not self.model.load(model):
            self.train_ae_model()
            # self.model.save(model)

    def train_ae_model(self, loss_function=torch.nn.BCELoss(), lr=5e-3, epochs=2):
        """
        train the encoder on the semantically enriched transaction dataset
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=2e-8)
        vectors = self.input_vectors['vector_list']
        random.shuffle(vectors)

        training_start_time = time.time()
        for epoch in range(epochs):
            for index in range(len(vectors)):
                print("Training progress:", (index + 1 + (epoch * len(vectors))), "/", (len(vectors) * epochs),
                      end="\r")
                cat_vector = torch.FloatTensor(vectors[index])
                noisy_cat_vector = (cat_vector + torch.normal(0, self.noise_factor, cat_vector.shape)).clip(0, 1)

                reconstructed = self.model(noisy_cat_vector, self.input_vectors['category_indices'][index])
                loss = loss_function(reconstructed, cat_vector)
                # partial_losses = []
                # for category_range in self.input_vectors['category_indices'][index]:
                #     start = category_range['start']
                #     end = category_range['end']
                #     partial_losses.append(loss_function(reconstructed[start:end], cat_vector[start:end]))

                # loss = sum(partial_losses)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.training_time = time.time() - training_start_time
