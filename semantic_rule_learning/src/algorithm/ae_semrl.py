import torch
import matplotlib.pyplot as plt

from src.model.ae_for_cat import AutoEncoderCat
from src.model.ae_for_kg import AutoEncoderKG
from src.model.ae_main import AutoEncoderMain
from src.preprocessing.base_preprocessing import *


class AESemRL:
    def __init__(self, knowledge_graph, transactions):
        self.knowledge_graph = knowledge_graph
        self.transactions = transactions

        self.input_vectors = get_transactions_as_vectors(knowledge_graph, transactions)
        self.model = None

    def generate_rules(self, num_rules, num_antecedents):
        """
        generate rules using the ARM-AE algorithm by Berteloot et al.
        :param num_rules: number of rules to generate
        :param num_antecedents: number of antecedents that each rule have
        """

        consequents = self.input_vectors['sensor']
        for consequent in consequents:
            consequent_rules = []
            for rule_index in range(num_rules):
                antecedents = []
                for antecedent_index in range(num_antecedents):
                    input_vector = antecedents + cons
                # output = self.model(torch.DoubleTensor(vector)).detach().numpy()

    def train(self):
        """
        train the autoencoder
        """
        # pretrain categorical attributes from the knowledge graph, to create a numerical representation for them
        ae_cat_model = AutoEncoderCat(len(self.input_vectors['categorical'][0]))
        pretrained_cat_model = self.train_ae_cat_model(ae_cat_model, self.input_vectors['categorical'])
        cat_model_weights = pretrained_cat_model.encoder[0].weight

        ae_kg_model = AutoEncoderKG(len(self.input_vectors['numerical'][0]), cat_model_weights)
        # pretrain categorical and numerical values from the knowledge graph
        pretrained_kg_model = self.train_ae_kg_model(ae_kg_model, self.input_vectors['categorical'],
                                                     self.input_vectors['numerical'])
        kg_model_weights = pretrained_kg_model.encoder[0].weight

        # train the main AE with the pretrained values above and the sensor data
        self.model = AutoEncoderMain(len(self.input_vectors['sensor'][0]), len(self.input_vectors['numerical'][0]),
                                     cat_model_weights, kg_model_weights)
        self.model = self.train_main_model(self.model, self.input_vectors)

    @staticmethod
    def train_main_model(model, vectors, loss_function=torch.nn.CrossEntropyLoss(), lr=1e-4, epochs=1):
        cat_vectors = vectors['categorical']
        num_vectors = vectors['numerical']
        transaction_vectors = vectors['sensor']

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)
        losses = []
        for epoch in range(epochs):
            for index in range(len(num_vectors)):
                cat_vector = torch.FloatTensor(cat_vectors[index])
                num_vector = torch.FloatTensor(num_vectors[index])
                transaction_vector = torch.FloatTensor(transaction_vectors[index])

                reconstructed = model(cat_vector, num_vector, transaction_vector)

                loss = loss_function(reconstructed, torch.DoubleTensor(cat_vectors[index] + num_vectors[index] +
                                                                       transaction_vectors[index]))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss)

        plt.style.use('fivethirtyeight')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.plot(torch.tensor(losses[-100:], requires_grad=True).detach().numpy())

        return model

    @staticmethod
    def train_ae_kg_model(model, cat_vectors, num_vectors, loss_function=torch.nn.CrossEntropyLoss(), lr=1e-4,
                          epochs=1):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)

        losses = []
        for epoch in range(epochs):
            for index in range(len(num_vectors)):
                cat_vector = torch.FloatTensor(cat_vectors[index])
                num_vector = torch.FloatTensor(num_vectors[index])

                reconstructed = model(cat_vector, num_vector)

                loss = loss_function(reconstructed, torch.DoubleTensor(num_vectors[index] + cat_vectors[index]))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss)

        plt.style.use('fivethirtyeight')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.plot(torch.tensor(losses[-100:], requires_grad=True).detach().numpy())

        return model

    @staticmethod
    def train_ae_cat_model(model, vectors, loss_function=torch.nn.CrossEntropyLoss(), lr=1e-4, epochs=1):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)

        losses = []
        for epoch in range(epochs):
            for vector in vectors:
                vector = torch.FloatTensor(vector)

                reconstructed = model(vector)

                loss = loss_function(reconstructed, vector)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss)

        plt.style.use('fivethirtyeight')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.plot(torch.tensor(losses[-100:], requires_grad=True).detach().numpy())

        return model

    @staticmethod
    def get_output_vectors(model, input_vectors):
        output_vectors = []
        for vector in input_vectors:
            output_vectors.append(model(torch.DoubleTensor(vector)))
        return output_vectors
