import random

import numpy as np
import torch
import copy
import matplotlib.pyplot as plt

from torch import nn
from src.model.ae_general import AutoEncoderGeneral
from src.preprocessing.base_preprocessing import *


class AESemRLCat:
    def __init__(self, knowledge_graph, transactions, num_bins, noise_factor=0.5):
        self.knowledge_graph = knowledge_graph
        self.transactions = transactions

        self.noise_factor = noise_factor
        # num_bins refers to number of bins that numerical values (including sensor measurements) are categorized into
        self.num_bins = num_bins

        self.input_vectors = get_transactions_as_cat_vectors(knowledge_graph, transactions, num_bins)
        self.input_vector_category_indices = get_category_boundaries(self.input_vectors['vector_tracker_list'][0])
        self.model = None
        self.softmax = nn.Softmax(dim=0)

    def generate_rules(self, similarity_threshold=0.7):
        """
        generate rules using the ARM-AE algorithm by Berteloot et al. (2023)
        """
        association_rules = []
        input_vector_size = len(self.input_vectors['vector_tracker_list'][0])
        for category in self.input_vector_category_indices:
            rule_subset = []

            for category2 in self.input_vector_category_indices:
                if category == category2:
                    continue
                for cat_index in range(category['end'] - category['start']):
                    input_vector = self.initialize_input_vector(
                        input_vector_size, self.input_vector_category_indices, [category, category2])
                    input_vector[cat_index + category['start']] = 1

                    implication_probabilities = self.calculate_implication_probabilities(similarity_threshold,
                                                                                         input_vector, category2)

                    implication = self.does_imply(implication_probabilities, similarity_threshold)
                    if implication:
                        association_rules.append(
                            self.get_rule([cat_index + category['start']], [category2['start'] + implication]))
                        rule_subset.append({
                            'categories': [category, category2],
                            'indices': [cat_index + category['start'], category2['start'] + implication]
                        })

                while len(rule_subset) > 0:
                    rule = rule_subset.pop()
                    input_vector = np.zeros(input_vector_size)
                    input_vector[rule['indices']] = 1
                    for category3 in self.input_vector_category_indices:
                        if category3 not in rule['categories']:
                            implication_probabilities = \
                                self.calculate_implication_probabilities(similarity_threshold, input_vector, category3)
                            implication = self.does_imply(implication_probabilities, similarity_threshold)
                            if implication:
                                association_rules.append(
                                    self.get_rule(rule['indices'], [category3['start'] + implication]))
                                rule_subset.append({
                                    'categories': rule['categories'] + [category3],
                                    'indices': rule['indices'] + [category3['start'] + implication]
                                })

        return association_rules

    def initialize_input_vector(self, input_vector_size, categories, exceptions):
        input_vector = np.zeros(input_vector_size)
        for category in categories:
            if category not in exceptions:
                input_vector[category['start']:category['end']] = 1 / (category['end'] - category['start'])
        return input_vector

    def calculate_implication_probabilities(self, similarity_threshold, input_vector, category):
        base_prob = (1 - similarity_threshold) / (category['end'] - category['start'] - 1)
        for cat2_index in range(category['end'] - category['start']):
            input_vector[cat2_index + category['start']] = base_prob

        prev_index = 0
        implication_probabilities = []
        for cat2_index in range(category['end'] - category['start']):
            if prev_index != cat2_index:
                input_vector[prev_index + category['start']] = base_prob
            input_vector[cat2_index + category['start']] = similarity_threshold
            output = self.model(torch.FloatTensor(input_vector), self.input_vector_category_indices)
            implication_probabilities.append(output[category['start']:category['end']])
            prev_index = cat2_index

        return implication_probabilities

    def get_rule(self, antecedents, consequents):
        rule = {'antecedents': [], 'consequents': []}
        for antecedent in antecedents:
            rule['antecedents'].append(self.input_vectors['vector_tracker_list'][0][antecedent])

        for consequent in consequents:
            rule['consequents'].append(self.input_vectors['vector_tracker_list'][0][consequent])

        return rule

    def does_imply(self, implication_probabilities, similarity_threshold):
        for query_index in range(len(implication_probabilities)):
            strong_implication = False
            no_implication = 0
            for prob_index in range(len(implication_probabilities)):
                if query_index == prob_index:
                    continue
                prob = implication_probabilities[prob_index]
                if prob[query_index] > similarity_threshold:
                    strong_implication = True
                elif prob[query_index] < 0.5:
                    no_implication += 1
            if strong_implication and no_implication < (len(implication_probabilities) / 2):
                return query_index
        return False

    def train(self):
        """
        train the autoencoder
        """
        # pretrain categorical attributes from the knowledge graph, to create a numerical representation for them
        self.model = AutoEncoderGeneral(len(self.input_vectors['vector_list'][0]))

        if not self.model.load("test"):
            self.train_ae_cat_model()
            self.model.save("test")

    def train_ae_cat_model(self, loss_function=torch.nn.BCELoss(), lr=1e-3, epochs=50):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-8)

        vectors = self.input_vectors['vector_list']
        losses = []

        plt.style.use('fivethirtyeight')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        random.shuffle(vectors)
        for epoch in range(epochs):
            for index in range(len(vectors)):
                print("Progress:", (index + (epoch * len(vectors))), "/", (len(vectors) * epochs), end="\r")
                cat_vector = torch.FloatTensor(vectors[index])
                noisy_cat_vector = (cat_vector + torch.normal(0, self.noise_factor, cat_vector.shape)).clip(0, 1)

                reconstructed = self.model(noisy_cat_vector, self.input_vector_category_indices)

                partial_losses = []
                for category_range in self.input_vector_category_indices:
                    start = category_range['start']
                    end = category_range['end']
                    partial_losses.append(loss_function(reconstructed[start:end], cat_vector[start:end]))

                loss = sum(partial_losses)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().numpy().item())

        losses_to_print = []
        for index in range(len(losses)):
            if index % 1000 == 0:
                losses_to_print.append(losses[index])

        plt.plot(losses_to_print)
        plt.show()
        plt.plot(losses)
        plt.show()
