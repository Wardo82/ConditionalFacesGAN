from torch.utils import data
import io
import numpy as np
import pandas as pd
import skimage.io as io
import os


class CelebADataset(data.Dataset):
    """ CelebA dataset. Imports the images and the attribute labels."""

    def __init__(self, dataroot, labelsfile, transform=None):
        """
        :param dataroot: Path to the CelebA folder where the images and labels are.
        :param labelsfile: Path to where the labels of the CelebA images are
        :param transform: Optional transform to be applied on a sample.
        """
        self.attrLabels = pd.read_csv(labelsfile, header=0, delim_whitespace=True)
        self.dataroot = dataroot
        self.transform = transform

    def __len__(self):
        return len(self.attrLabels)

    def __getitem__(self, item):
        image_path = self.attrLabels.index[item]
        imgName = os.path.join(self.dataroot+'/img_align_celeba/', image_path)

        image = io.imread(imgName)
        if self.transform:
            image = self.transform(image)

        attributes = self.attrLabels.iloc[item, :].values
        attributes[np.where(attributes < 0)] = 0
        sample = {'image': image, 'attributes': attributes}

        return sample
