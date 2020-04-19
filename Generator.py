from numpy import prod
import torch
import torch.nn as nn
import torch.functional as F

class Generator(nn.Module):
    """
    The generator G maps a latent vector z to a data-space.
    """
    def __init__(self, params):
        super(Generator, self).__init__()
        self.color_channels = params['number_of_channels']
        feature_map_size = params['feature_map_size']
        latent_vector_size = params['latent_vector_size']
        number_of_attr = params['number_of_attr']

        # The size of the feature maps that are propagated through the generator
        self.number_of_random_layers = 4  # K
        self.random_input_resolution = 4  # N
        num_nodes = self.random_input_resolution * self.random_input_resolution  # A layer of NxN
        # Input for the latent vector
        self.InputLatentVector = nn.Sequential(
            # Inputs a vector of v elems, and outputs K layers of NxN pixels
            nn.Linear(latent_vector_size, self.number_of_random_layers * num_nodes),
            nn.LeakyReLU(),
        )

        self.feature_map_size = feature_map_size  # The resolution of the output image
        number_of_embedding_dim = 4  # The number of dimensions of the vector space of the source labels vector.
        # Input for the labels
        self.InputLabelsVector = nn.Sequential(
            # Inputs a vector of l labels, and outputs a DxA embedding layer (D parameters per label)
            nn.Embedding(number_of_attr, number_of_embedding_dim * number_of_attr),
            # Embedding layer as input, returns L NxN layers as a representation of each label
            nn.Linear(number_of_embedding_dim * number_of_attr, num_nodes)
        )

        self.main = nn.Sequential(
            # Input going into a convolution
            nn.ConvTranspose2d(in_channels=number_of_attr + self.number_of_random_layers,
                               out_channels=feature_map_size * 2,
                               kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(True),
            # state size. (gen_feature_map_size * 8) x 4 x 4
            nn.ConvTranspose2d(in_channels=feature_map_size*2, out_channels=feature_map_size * 2,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(True),
            # state size. (gen_feature_map_size * 4) x 8 x 8
            nn.ConvTranspose2d(in_channels=feature_map_size*2, out_channels=feature_map_size,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(True),
            # state size. (gen_feature_map_size * 2) x 16 x 16
            nn.ConvTranspose2d(in_channels=feature_map_size , out_channels=feature_map_size,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(True),
            # state size. (gen_feature_map_size) x 32 x 32
            nn.ConvTranspose2d(in_channels=feature_map_size, out_channels=self.color_channels,
                               kernel_size=4, stride=2, padding=1, bias=False),
        )

    def forward(self, latent_vector, labels):
        # Vector preparation
        latent_vector = latent_vector.squeeze()  # Take out dimensions that are equal to 1
        vector = self.InputLatentVector(latent_vector)
        vector = vector.reshape((vector.size(0), self.number_of_random_layers,
                                 self.random_input_resolution, self.random_input_resolution))

        # Labels preparation
        labels = labels.squeeze()  # Take out dimensions that are equal to 1
        labels = self.InputLabelsVector(labels)
        labels = labels.reshape((labels.size(0), 40, self.random_input_resolution, self.random_input_resolution))

        x = torch.cat((vector, labels), dim=1)

        y = self.main(x)
        z = nn.functional.tanh(nn.Linear(prod(y.size()), y.size(0)*3*64*64))
        return z.reshape(y.size(0), self.color_channels, self.feature_map_size, self.feature_map_size)

