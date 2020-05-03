import numpy as np
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Discriminator agent that tells if an input image, given the input labels comes from the real distribution
    or a fake one. The output is true or false.
    """
    def __init__(self, params):
        super(Discriminator, self).__init__()
        feature_map_size = self.feature_map_size = params['feature_map_size']
        self.number_of_attr = params['number_of_attr']
        number_of_color_channels = params['num_of_channels']

        self.embeddingLayer = nn.Embedding(self.number_of_attr, 50)
        num_nodes = self.feature_map_size * self.feature_map_size  # As it is a square picture of 64x64
        self.FCLayer = nn.Linear(50*self.number_of_attr, num_nodes)  # Embedding layer as input and returns a 64x64 array

        self.main = nn.Sequential(
            # input is (number_of_color_channels+1) x 64 x 64. The "+1" comes from embedding the labels layer.
            nn.Conv2d(in_channels=number_of_color_channels+1, out_channels=self.feature_map_size,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # State size: (dis_feature_map_size) x 32 x 32
            nn.Conv2d(in_channels=feature_map_size, out_channels=feature_map_size * 2,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=feature_map_size * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # State size: (dis_feature_map_size) x 16 x 16
            nn.Conv2d(in_channels=feature_map_size * 2, out_channels=feature_map_size * 4,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (dis_feature_map_size * 4) x 8 x 8
            nn.Conv2d(in_channels=feature_map_size * 4, out_channels=feature_map_size * 8,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=feature_map_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (dis_feature_map_size * 8) x 4 x 4
            nn.Conv2d(in_channels=feature_map_size * 8, out_channels=1,
                      kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, images, labels):

        a = self.embeddingLayer(labels)  # Embed the input labels in a vector space
        a = a.reshape((a.size(0), -1))  # Reshape the las two components to have n labels with k=NxM components
        a = self.FCLayer(a)
        a = a.reshape((a.size(0), self.feature_map_size, self.feature_map_size))
        a = a.unsqueeze(1)  # Add one dimension for the 64x64 word embedding of each image
        x = torch.cat((images, a), dim=1)

        y = self.main(x)
        return y
