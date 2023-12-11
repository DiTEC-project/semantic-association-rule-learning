import torch
from torch import nn, optim
from torch.autograd import Variable


class AutoEncoderMain(nn.Module):
    """
    The main autoencoder that is used to create semantic association rules
    """

    def __init__(self, transactions_size, num_feature_size, cat_model_weights, kg_model_weights):
        """
        :param transactions_size: size of the transaction list (sensor measurements)
        :param cat_model_weights: pretrained weights for the categorical features in the knowledge graph
        :param kg_model_weights: pretrained weights for the numerical AND categorical features in the knowledge graph
        """
        super().__init__()
        self.transactions_size = transactions_size
        self.num_feature_size = num_feature_size
        self.cat_model_weights = cat_model_weights
        self.kg_model_weights = kg_model_weights

        output_layer = nn.Softmax()

        reduced_cat_feature_size = int(cat_model_weights.shape[1] * 3 / 4)
        self.encoder_cat = nn.Sequential(
            nn.Linear(cat_model_weights.shape[1], reduced_cat_feature_size),
            output_layer,
        )

        reduced_num_feature_size = int(kg_model_weights.shape[1] * 3 / 4)
        self.encoder_kg = nn.Sequential(
            nn.Linear(kg_model_weights.shape[1], reduced_num_feature_size),
            output_layer,
        )

        encoder_input_size = self.transactions_size + reduced_num_feature_size
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_size, int(encoder_input_size * 3 / 4)),
            output_layer,
            nn.Linear(int(encoder_input_size * 3 / 4), int(encoder_input_size * 2 / 4)),
            output_layer,
            nn.Linear(int(encoder_input_size * 2 / 4), int(encoder_input_size * 1 / 4)),
            output_layer,
        )

        self.decoder = nn.Sequential(
            nn.Linear(int(encoder_input_size * 1 / 4), int(encoder_input_size * 2 / 4)),
            output_layer,
            nn.Linear(int(encoder_input_size * 2 / 4), int(encoder_input_size * 3 / 4)),
            output_layer,
            nn.Linear(int(encoder_input_size * 3 / 4), encoder_input_size),
            output_layer,
        )

        self.decoder_kg = nn.Sequential(
            nn.Linear(reduced_num_feature_size, kg_model_weights.shape[1]),
            output_layer
        )
        self.decoder_cat = nn.Sequential(
            nn.Linear(reduced_cat_feature_size, cat_model_weights.shape[1]),
            output_layer,
        )

        self.encoder_cat[0].weight = cat_model_weights
        self.encoder_kg[0].weight = kg_model_weights
        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)
        self.decoder_kg.apply(self.init_weights)
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
        torch.save(self.encoder.state_dict(), p + 'encoder.pt')
        torch.save(self.decoder.state_dict(), p + 'decoder.pt')

    def load(self, encoder_path, decoder_path):
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))
        self.encoder.eval()
        self.decoder.eval()

    def forward(self, cat_vector, num_vector, transaction_vector):
        x = self.encoder_cat(cat_vector)
        x = self.encoder_kg(torch.FloatTensor(num_vector.detach().numpy().tolist() + x.detach().numpy().tolist()))
        x = self.encoder(torch.FloatTensor(transaction_vector.detach().numpy().tolist() + x.detach().numpy().tolist()))

        x = self.decoder(x)

        decoder_kg_input = x.data[self.transactions_size:]
        partial_output_transactions = x.data[:self.transactions_size]

        x = self.decoder_kg(decoder_kg_input)

        decoder_cat_input = x.data[self.num_feature_size:]
        partial_output_kg = x.data[:self.num_feature_size]

        x = self.decoder_cat(decoder_cat_input)

        output = torch.cat((partial_output_transactions, partial_output_kg, x))

        return output
