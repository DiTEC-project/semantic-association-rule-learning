import torch
from torch import nn, optim
from torch.autograd import Variable


class AutoEncoderCat(nn.Module):
    """
    This autoencoder is used to create a numerical representation for the categorical values, as described
    in the paper by Delong et al. and it has no hidden layer.
    """

    def __init__(self, data_size):
        super().__init__()
        self.data_size = data_size
        output_layer = nn.Softmax()
        self.encoder = nn.Sequential(
            nn.Linear(self.data_size, int(self.data_size * 3 / 4)),
            output_layer,
        )
        self.decoder = nn.Sequential(
            nn.Linear(int(self.data_size * 3 / 4), self.data_size),
            output_layer,
        )

        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)

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
        torch.save(self.encoder.state_dict(), p + 'cat_encoder.pt')
        torch.save(self.decoder.state_dict(), p + 'cat_decoder.pt')

    def load(self, encoder_path, decoder_path):
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))
        self.encoder.eval()
        self.decoder.eval()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
