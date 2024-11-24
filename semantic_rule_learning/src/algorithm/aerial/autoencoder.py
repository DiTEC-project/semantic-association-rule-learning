import torch
import os
from torch import nn


class AutoEncoder(nn.Module):
    """
    An Autoencoder model for our AE-based ARM approach
    """

    def __init__(self, data_size):
        """
        :param data_size: number of features after one-hot encoding in the input data
        """
        super().__init__()
        self.data_size = data_size
        self.encoder = nn.Sequential(
            nn.Linear(self.data_size, int(1 * self.data_size / 8)),
            nn.Tanh(),
            nn.Linear(int(1 * self.data_size / 8), int(1 * self.data_size / 32)),
            nn.Tanh(),
            nn.Linear(int(1 * self.data_size / 32), int(1 * self.data_size / 128)),
        )
        self.decoder = nn.Sequential(
            nn.Linear(int(1 * self.data_size / 128), int(1 * self.data_size / 32)),
            nn.Tanh(),
            nn.Linear(int(1 * self.data_size / 32), int(1 * self.data_size / 8)),
            nn.Tanh(),
            nn.Linear(int(1 * self.data_size / 8), self.data_size)
        )

        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)
        self.softmax = nn.Softmax(dim=0)
        self.batch_norm = nn.BatchNorm1d(self.data_size)

    @staticmethod
    def init_weights(m):
        """
        all weights are initialized with values sampled from uniform distributions with the Xavier initialization
        and the biases are set to 0, as described in the paper by Delong et al. (2023)
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.zero_()

    def save(self, p):
        """
        Save the encoder and decoder models
        """
        torch.save(self.encoder.state_dict(), p + '_encoder.pt')
        torch.save(self.decoder.state_dict(), p + '_decoder.pt')

    def load(self, p):
        """
        load the encoder and decoder models
        """
        if os.path.isfile(p + '_encoder.pt') and os.path.isfile(p + '_decoder.pt'):
            self.encoder.load_state_dict(torch.load(p + '_encoder.pt'))
            self.decoder.load_state_dict(torch.load(p + '_decoder.pt'))
            self.encoder.eval()
            self.decoder.eval()
            return True
        else:
            return False

    def forward(self, x, input_vector_category_indices):
        y = self.encoder(x)
        y = self.decoder(y)

        # apply softmax to class values of each feature (category) individually
        for category_index in range(len(input_vector_category_indices)):
            category_range = input_vector_category_indices[category_index]
            y[category_range['start']:category_range['end']] = \
                self.softmax(y[category_range['start']:category_range['end']])

        return y
