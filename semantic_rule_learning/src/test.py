import torch
import math
import matplotlib.pyplot as plt
from torch import nn
import numpy as np


class AutoEncoder(nn.Module):
    def __init__(self, data_size):
        super().__init__()
        self.data_size = data_size
        self.encoder = nn.Sequential(
            nn.Linear(self.data_size, 3),
            nn.Tanh(),
            nn.Linear(3, 2),
            nn.Tanh(),
            nn.Linear(2, 1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, 2),
            nn.Tanh(),
            nn.Linear(2, 3),
            nn.Tanh(),
            nn.Linear(3, self.data_size),
        )

        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)

        self.softmax1 = nn.Softmax(dim=0)
        self.softmax2 = nn.Softmax(dim=0)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.zero_()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x[:2] = self.softmax1(x[:2])
        x[-2:] = self.softmax2(x[-2:])
        return x


def train(model, vectors, loss_function=torch.nn.BCELoss(), lr=5e-3, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=2e-8)
    noise_factor = 0.5

    losses = []
    for epoch in range(epochs):
        for vector in vectors:
            vector = torch.FloatTensor(vector)

            noisy_vector = (vector + torch.normal(0, noise_factor, vector.shape)).clip(0, 1)

            reconstructed = model(noisy_vector)

            loss1 = loss_function(reconstructed[0:2], vector[0:2])
            loss2 = loss_function(reconstructed[2:4], vector[2:4])
            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # losses.append(loss.detach().numpy().item())

    # Plotting the last 100 values
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    # plt.plot(losses[-50000:])
    # plt.show()

    return model


experiment_1_data = [
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1]
]

experiment_2_data = [
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
]

experiment_3_data = [
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
]

experiment_4_data = [
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 1, 0, 1],
]

experiment_5_data = [
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 1, 0, 1],
    [0, 1, 0, 1],
    [0, 1, 0, 1],
]


def is_preference(query1, query2, index1, index2, threshold):
    if (query1[index1] > threshold and query2[index1] > 0.5) or (query2[index1] > threshold and query1[index1] > 0.5):
        return index1
    if (query1[index2] > threshold and query2[index2] > 0.5) or (query2[index2] > threshold and query1[index2] > 0.5):
        return index2

    return False


if __name__ == "__main__":
    exp1 = False
    exp2 = False
    exp3 = False
    exp4 = False
    exp5 = True

    THRESHOLD = 0.7

    if exp1:
        num_of_fails = 0
        for i in range(40):
            autoencoder = AutoEncoder(len(experiment_1_data[0]))
            autoencoder = train(autoencoder, experiment_1_data)
            query1 = autoencoder(torch.FloatTensor([1, 0, (1 - THRESHOLD), THRESHOLD])).detach().numpy().tolist()
            query15 = autoencoder(torch.FloatTensor([1, 0, THRESHOLD, (1 - THRESHOLD)])).detach().numpy().tolist()
            query2 = autoencoder(torch.FloatTensor([0, 1, (1 - THRESHOLD), THRESHOLD])).detach().numpy().tolist()
            query25 = autoencoder(torch.FloatTensor([0, 1, THRESHOLD, (1 - THRESHOLD)])).detach().numpy().tolist()
            query3 = autoencoder(torch.FloatTensor([(1 - THRESHOLD), THRESHOLD, 1, 0])).detach().numpy().tolist()
            query35 = autoencoder(torch.FloatTensor([THRESHOLD, (1 - THRESHOLD), 1, 0])).detach().numpy().tolist()
            query4 = autoencoder(torch.FloatTensor([(1 - THRESHOLD), THRESHOLD, 0, 1])).detach().numpy().tolist()
            query45 = autoencoder(torch.FloatTensor([THRESHOLD, (1 - THRESHOLD), 0, 1])).detach().numpy().tolist()

            if is_preference(query1, query15, 2, 3, THRESHOLD):
                num_of_fails += 1
                print("Failure in query 1:\n", query1, "\n", query15)
            if is_preference(query2, query25, 2, 3, THRESHOLD):
                num_of_fails += 1
                print("Failure in query 2:\n", query2, "\n", query25)
            if is_preference(query3, query35, 0, 1, THRESHOLD) != 0:
                num_of_fails += 1
                print("Failure in query 3:\n", query3, "\n", query35)
            if is_preference(query4, query45, 0, 1, THRESHOLD) != 0:
                num_of_fails += 1
                print("Failure in query 4:\n", query4, "\n", query45)
        print("Experiment 1 - Number of failures:", num_of_fails)

    if exp2:
        num_of_fails = 0
        for i in range(40):
            autoencoder = AutoEncoder(len(experiment_2_data[0]))
            autoencoder = train(autoencoder, experiment_2_data)
            query1 = autoencoder(torch.FloatTensor([1, 0, (1 - THRESHOLD), THRESHOLD])).detach().numpy().tolist()
            query15 = autoencoder(torch.FloatTensor([1, 0, THRESHOLD, (1 - THRESHOLD)])).detach().numpy().tolist()
            query2 = autoencoder(torch.FloatTensor([0, 1, (1 - THRESHOLD), THRESHOLD])).detach().numpy().tolist()
            query25 = autoencoder(torch.FloatTensor([0, 1, THRESHOLD, (1 - THRESHOLD)])).detach().numpy().tolist()
            query3 = autoencoder(torch.FloatTensor([(1 - THRESHOLD), THRESHOLD, 1, 0])).detach().numpy().tolist()
            query35 = autoencoder(torch.FloatTensor([THRESHOLD, (1 - THRESHOLD), 1, 0])).detach().numpy().tolist()
            query4 = autoencoder(torch.FloatTensor([(1 - THRESHOLD), THRESHOLD, 0, 1])).detach().numpy().tolist()
            query45 = autoencoder(torch.FloatTensor([THRESHOLD, (1 - THRESHOLD), 0, 1])).detach().numpy().tolist()

            if is_preference(query1, query15, 2, 3, THRESHOLD):
                num_of_fails += 1
                print("Failure in query 1:\n", query1, "\n", query15)
            if is_preference(query2, query25, 2, 3, THRESHOLD):
                num_of_fails += 1
                print("Failure in query 2:\n", query2, "\n", query25)
            if is_preference(query3, query35, 0, 1, THRESHOLD) != 0:
                num_of_fails += 1
                print("Failure in query 3:\n", query3, "\n", query35)
            if is_preference(query4, query45, 0, 1, THRESHOLD) != 0:
                num_of_fails += 1
                print("Failure in query 4:\n", query4, "\n", query45)
        print("Experiment 2 - Number of failures:", num_of_fails)

    if exp3:
        num_of_fails = 0
        for i in range(40):
            autoencoder = AutoEncoder(len(experiment_3_data[0]))
            autoencoder = train(autoencoder, experiment_3_data)
            query1 = autoencoder(torch.FloatTensor([1, 0, (1 - THRESHOLD), THRESHOLD])).detach().numpy().tolist()
            query15 = autoencoder(torch.FloatTensor([1, 0, THRESHOLD, (1 - THRESHOLD)])).detach().numpy().tolist()
            query2 = autoencoder(torch.FloatTensor([0, 1, (1 - THRESHOLD), THRESHOLD])).detach().numpy().tolist()
            query25 = autoencoder(torch.FloatTensor([0, 1, THRESHOLD, (1 - THRESHOLD)])).detach().numpy().tolist()
            query3 = autoencoder(torch.FloatTensor([(1 - THRESHOLD), THRESHOLD, 1, 0])).detach().numpy().tolist()
            query35 = autoencoder(torch.FloatTensor([THRESHOLD, (1 - THRESHOLD), 1, 0])).detach().numpy().tolist()
            query4 = autoencoder(torch.FloatTensor([(1 - THRESHOLD), THRESHOLD, 0, 1])).detach().numpy().tolist()
            query45 = autoencoder(torch.FloatTensor([THRESHOLD, (1 - THRESHOLD), 0, 1])).detach().numpy().tolist()

            if is_preference(query1, query15, 2, 3, THRESHOLD):
                num_of_fails += 1
                print("Failure in query 1:\n", query1, "\n", query15)
            if is_preference(query2, query25, 2, 3, THRESHOLD) != 2:
                num_of_fails += 1
                print("Failure in query 2:\n", query2, "\n", query25)
            if is_preference(query3, query35, 0, 1, THRESHOLD):
                num_of_fails += 1
                print("Failure in query 3:\n", query3, "\n", query35)
            if is_preference(query4, query45, 0, 1, THRESHOLD) != 0:
                num_of_fails += 1
                print("Failure in query 4:\n", query4, "\n", query45)
        print("Experiment 3 - Number of failures:", num_of_fails)

    if exp4:
        num_of_fails = 0
        for i in range(40):
            autoencoder = AutoEncoder(len(experiment_4_data[0]))
            autoencoder = train(autoencoder, experiment_4_data)
            query1 = autoencoder(torch.FloatTensor([1, 0, (1 - THRESHOLD), THRESHOLD])).detach().numpy().tolist()
            query15 = autoencoder(torch.FloatTensor([1, 0, THRESHOLD, (1 - THRESHOLD)])).detach().numpy().tolist()
            query2 = autoencoder(torch.FloatTensor([0, 1, (1 - THRESHOLD), THRESHOLD])).detach().numpy().tolist()
            query25 = autoencoder(torch.FloatTensor([0, 1, THRESHOLD, (1 - THRESHOLD)])).detach().numpy().tolist()
            query3 = autoencoder(torch.FloatTensor([(1 - THRESHOLD), THRESHOLD, 1, 0])).detach().numpy().tolist()
            query35 = autoencoder(torch.FloatTensor([THRESHOLD, (1 - THRESHOLD), 1, 0])).detach().numpy().tolist()
            query4 = autoencoder(torch.FloatTensor([(1 - THRESHOLD), THRESHOLD, 0, 1])).detach().numpy().tolist()
            query45 = autoencoder(torch.FloatTensor([THRESHOLD, (1 - THRESHOLD), 0, 1])).detach().numpy().tolist()
            if is_preference(query1, query15, 2, 3, THRESHOLD):
                num_of_fails += 1
                print("Failure in query 1:\n", query1, "\n", query15)
            if is_preference(query2, query25, 2, 3, THRESHOLD) != 2:
                num_of_fails += 1
                print("Failure in query 2:\n", query2, "\n", query25)
            if is_preference(query3, query35, 0, 1, THRESHOLD):
                num_of_fails += 1
                print("Failure in query 3:\n", query3, "\n", query35)
            if is_preference(query4, query45, 0, 1, THRESHOLD) != 0:
                num_of_fails += 1
                print("Failure in query 4:\n", query4, "\n", query45)
        print("Experiment 4 - Number of failures:", num_of_fails)

    if exp5:
        num_of_fails = 0
        for i in range(40):
            autoencoder = AutoEncoder(len(experiment_5_data[0]))
            autoencoder = train(autoencoder, experiment_5_data)
            query1 = autoencoder(torch.FloatTensor([1, 0, (1 - THRESHOLD), THRESHOLD])).detach().numpy().tolist()
            query15 = autoencoder(torch.FloatTensor([1, 0, THRESHOLD, (1 - THRESHOLD)])).detach().numpy().tolist()
            query2 = autoencoder(torch.FloatTensor([0, 1, (1 - THRESHOLD), THRESHOLD])).detach().numpy().tolist()
            query25 = autoencoder(torch.FloatTensor([0, 1, THRESHOLD, (1 - THRESHOLD)])).detach().numpy().tolist()
            query3 = autoencoder(torch.FloatTensor([(1 - THRESHOLD), THRESHOLD, 1, 0])).detach().numpy().tolist()
            query35 = autoencoder(torch.FloatTensor([THRESHOLD, (1 - THRESHOLD), 1, 0])).detach().numpy().tolist()
            query4 = autoencoder(torch.FloatTensor([(1 - THRESHOLD), THRESHOLD, 0, 1])).detach().numpy().tolist()
            query45 = autoencoder(torch.FloatTensor([THRESHOLD, (1 - THRESHOLD), 0, 1])).detach().numpy().tolist()

            if is_preference(query1, query15, 2, 3, THRESHOLD) != 2:
                num_of_fails += 1
                print("Failure in query 1:\n", query1, "\n", query15)
            if is_preference(query2, query25, 2, 3, THRESHOLD) != 3:
                num_of_fails += 1
                print("Failure in query 2:\n", query2, "\n", query25)
            if is_preference(query3, query35, 0, 1, THRESHOLD) != 0:
                num_of_fails += 1
                print("Failure in query 3:\n", query3, "\n", query35)
            if is_preference(query4, query45, 0, 1, THRESHOLD) != 1:
                num_of_fails += 1
                print("Failure in query 4:\n", query4, "\n", query45)
        print("Experiment 5 - Number of failures:", num_of_fails)
