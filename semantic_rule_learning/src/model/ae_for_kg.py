import torch
from torch import nn, optim
from torch.autograd import Variable


class AutoEncoderKG(nn.Module):
    """
    This autoencoder is used to create a numerical representation for the categorical values AND the numerical values
    in the knowledge graph. It is inspired from the paper by Delong et al.
    """

    def __init__(self, num_feature_size, cat_model):
        """
        :param num_feature_size: size of the numerical features in the knowledge graph
        :param cat_model: pretrained model for the categorical features in the knowledge graph
        """
        super().__init__()
        self.cat_model_weights = cat_model.encoder[0].weight
        self.cat_model_biases = cat_model.encoder[0].bias

        self.num_feature_size = num_feature_size
        self.cat_model_weights = self.cat_model_weights

        reduced_cat_feature_size = int(self.cat_model_weights.shape[1] * 3 / 4)
        self.encoder_cat = nn.Sequential(
            nn.Linear(self.cat_model_weights.shape[1], reduced_cat_feature_size)
        )

        encoder_input_size = self.num_feature_size + reduced_cat_feature_size
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_size, int(encoder_input_size * 3 / 4)),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(int(encoder_input_size * 3 / 4), encoder_input_size)
        )

        self.decoder_cat = nn.Sequential(
            nn.Linear(reduced_cat_feature_size, self.cat_model_weights.shape[1]),
            nn.Softmax(dim=0),
        )

        self.encoder_cat[0].weight = self.cat_model_weights
        self.encoder_cat[0].bias = self.cat_model_biases
        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)
        self.decoder_cat.apply(self.init_weights)

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
        torch.save(self.encoder.state_dict(), p + 'kg_encoder.pt')
        torch.save(self.decoder.state_dict(), p + 'kg_decoder.pt')

    def load(self, encoder_path, decoder_path):
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))
        self.encoder.eval()
        self.decoder.eval()

    def forward(self, cat_vector, num_vector):
        x = self.encoder_cat(cat_vector)
        x = self.encoder(torch.FloatTensor(num_vector.detach().numpy().tolist() + x.detach().numpy().tolist()))
        x = self.decoder(x)

        de_cat_input = x.data[self.num_feature_size:]
        partial_output = x.data[:self.num_feature_size]
        x = self.decoder_cat(de_cat_input)
        output = torch.cat((partial_output, x))

        return output
