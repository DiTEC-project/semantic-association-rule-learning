import numpy as np
import torch
import matplotlib.pyplot as plt

from src.model.ae_for_cat import AutoEncoderCat
from src.model.ae_for_kg import AutoEncoderKG
from src.model.ae_main import AutoEncoderMain
from src.preprocessing.base_preprocessing import *


class AESemRLMixed:
    def __init__(self, knowledge_graph, transactions, noise_factor=0.5):
        self.knowledge_graph = knowledge_graph
        self.transactions = transactions

        self.noise_factor = noise_factor
        self.num_scaler = MinMaxScaler(feature_range=(-1, 1))

        self.input_vectors = get_transactions_as_vectors(knowledge_graph, transactions)
        self.model = None

    def generate_rules(self, num_rules, num_antecedents):
        """
        generate rules using the ARM-AE algorithm by Berteloot et al.
        :param num_rules: number of rules to generate
        :param num_antecedents: number of antecedents that each rule have
        """

        antecedents = self.input_vectors['sensor']
        rules = []
        for antecedent in antecedents:
            antecedent_rules = []
            for rule_index in range(num_rules):
                consequents = []
                for antecedent_index in range(num_antecedents):
                    input_vector = np.zeros(len(antecedent))
                    input_vector[0] = antecedent[0]
                    ae_output = self.model(torch.Tensor(np.zeros(len(self.input_vectors['categorical'][0]))),
                                           torch.Tensor(np.zeros(len(self.input_vectors['numerical'][0]))),
                                           torch.Tensor(input_vector))
                    ae_output = ae_output.detach().numpy().tolist()
                    print(self.input_vectors['id_tracker'][0])
                    for index in range(len(ae_output)):
                        if ae_output[index] > 0.5:
                            print(self.input_vectors['id_tracker'][index], ae_output[index])

    def train(self):
        """
        train the autoencoder
        """
        # pretrain categorical attributes from the knowledge graph, to create a numerical representation for them
        ae_cat_model = AutoEncoderCat(len(self.input_vectors['categorical'][0]))
        pretrained_cat_model = self.train_ae_cat_model(ae_cat_model, self.input_vectors['categorical'])

        ae_kg_model = AutoEncoderKG(len(self.input_vectors['numerical'][0]), pretrained_cat_model)
        # pretrain categorical and numerical values from the knowledge graph
        pretrained_kg_model = self.train_ae_kg_model(ae_kg_model, self.input_vectors['categorical'],
                                                     self.input_vectors['numerical'])

        # train the main AE with the pretrained values above and the sensor data
        self.model = AutoEncoderMain(len(self.input_vectors['sensor'][0]), len(self.input_vectors['numerical'][0]),
                                     pretrained_kg_model)
        self.model = self.train_main_model(self.model, self.input_vectors)

    def train_main_model(self, model, vectors, loss_function=torch.nn.MSELoss(), lr=1e-4, epochs=1):
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

                scaled_num_vector = self.num_scaler.fit_transform(num_vector.reshape(-1, 1))
                scaled_num_vector = scaled_num_vector.reshape(1, len(num_vector))[0].tolist()
                scaled_num_vector_tensor = torch.FloatTensor(scaled_num_vector)

                scaled_transaction_vector = self.num_scaler.fit_transform(transaction_vector.reshape(-1, 1))
                scaled_transaction_vector = scaled_transaction_vector.reshape(1, len(transaction_vector))[0].tolist()
                scaled_transaction_vector_tensor = torch.FloatTensor(scaled_transaction_vector)

                noisy_cat_vector = (cat_vector + torch.normal(0, self.noise_factor, cat_vector.shape)).clip(0, 1)
                noisy_num_vector = (scaled_num_vector_tensor + torch.normal(0, self.noise_factor,
                                                                            scaled_num_vector_tensor.shape)).clip(-1, 1)
                noisy_transaction_vector = (scaled_transaction_vector_tensor +
                                            torch.normal(0, self.noise_factor,
                                                         scaled_transaction_vector_tensor.shape)).clip(-1, 1)

                reconstructed = model(noisy_cat_vector, noisy_num_vector, noisy_transaction_vector)

                loss = loss_function(reconstructed, torch.FloatTensor(scaled_transaction_vector + scaled_num_vector +
                                                                      cat_vectors[index]))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss)

        plt.style.use('fivethirtyeight')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        # plt.plot(torch.tensor(losses[-100:], requires_grad=True).detach().numpy())

        return model

    def train_ae_kg_model(self, model, cat_vectors, num_vectors, loss_function=torch.nn.MSELoss(), lr=1e-8,
                          epochs=5):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        losses = []
        for epoch in range(epochs):
            for index in range(len(num_vectors)):
                cat_vector = torch.FloatTensor(cat_vectors[index])
                num_vector = torch.FloatTensor(num_vectors[index])

                scaled_num_vector = self.num_scaler.fit_transform(num_vector.reshape(-1, 1))
                scaled_num_vector = scaled_num_vector.reshape(1, len(num_vector))[0].tolist()
                scaled_num_vector_tensor = torch.FloatTensor(scaled_num_vector)

                noisy_cat_vector = (cat_vector + torch.normal(0, self.noise_factor, cat_vector.shape)).clip(0, 1)
                noisy_num_vector = (scaled_num_vector_tensor +
                                    torch.normal(0, self.noise_factor, scaled_num_vector_tensor.shape)).clip(-1, 1)

                reconstructed = model(noisy_cat_vector, noisy_num_vector)

                loss = loss_function(reconstructed, torch.FloatTensor(scaled_num_vector + cat_vectors[index]))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss)

        plt.style.use('fivethirtyeight')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        # plt.plot(torch.tensor(losses[-100:], requires_grad=True).detach().numpy())

        return model

    def train_ae_cat_model(self, model, vectors, loss_function=torch.nn.CrossEntropyLoss(), lr=1e-4, epochs=1):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)

        losses = []
        for epoch in range(epochs):
            for vector in vectors:
                cat_vector = torch.FloatTensor(vector)
                noisy_cat_vector = (cat_vector + torch.normal(0, self.noise_factor, cat_vector.shape)).clip(0, 1)

                reconstructed = model(noisy_cat_vector)

                loss = loss_function(reconstructed, cat_vector)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss)

        plt.style.use('fivethirtyeight')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        # plt.plot(torch.tensor(losses[-100:], requires_grad=True).detach().numpy())

        return model

    @staticmethod
    def get_output_vectors(model, input_vectors):
        output_vectors = []
        for vector in input_vectors:
            output_vectors.append(model(torch.DoubleTensor(vector)))
        return output_vectors
