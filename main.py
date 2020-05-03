from __future__ import print_function
# %matplotlib inline
import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import CelebADataset as Datasets
from Generator import Generator
from Discriminator import Discriminator
from Solver import GANSolver
from IPython.display import HTML

# Preliminary variables (in alphabetical order)
batch_size = 64  # Batch size during training
dataroot = "../Datasets/celeba-min"  # Dataset's root directory
image_size = 64  # Spatial size of training images. All will end up image_size x image_size
latent_vector_size = 100  # Size of z latent vector used as random input of the generator
manualSeed = 999  # Set random seed for reproducibility
# manualSeed = random.randint(1, 10000) # use if you want new results
number_of_color_channels = 3  # Number of channels in the training images
number_of_gpu = 0  # Number of GPUs available. Use 0 for CPU mode.
workers = 0  # Number of workers for dataloader

# Initialize random seed for random and torch package
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# Find the available device and use it
device = torch.device("cuda:0" if (torch.cuda.is_available() and number_of_gpu > 0) else "cpu")


# Create the data set using the data root directory with the images and attributes
dataset = Datasets.CelebADataset(dataroot, dataroot+"/list_attr_celeba.txt",
                                 transform=transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor()]))
# Create the DataLoader object using the previously created data set
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

# Plot some training images
sample_batched = next(iter(dataloader))
print(sample_batched['image'].size(),
      sample_batched['attributes'].size())
#plt.figure(figsize=(8, 8))
#plt.axis("off")
#plt.title("Training Images")
#plt.imshow(np.transpose(vutils.make_grid(sample_batched['image'].to(device)[:64],
#                                        padding=2,
#                                      normalize=True).cpu(),
  #                      (1, 2, 0)))
#plt.show()

# === Implementation ===
# Weight Initialization: The paper states that all model weights shall be randomly initialized
# from a Normal distribution with mean=0, stdev=0.02.
# TODO: Understand what nn.init does
def weights_init(model):
    """
    Custom weights initialization called on netG and netD right after model initialization
    :param model: The model whose weights are to be modified
    """
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


generator_parameters = {  # Used in the Generator's constructor
    'feature_map_size': 64,
    'number_of_channels': number_of_color_channels,
    'latent_vector_size': latent_vector_size,
    'number_of_attr': 40
}

# Create the generator
netG = Generator(generator_parameters).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (number_of_gpu > 1):
    netG = nn.DataParallel(netG, list(range(number_of_gpu)))

# Apply the weight init function to randomly initialize all the weights to mean=0, stdev=0.2
netG.apply(weights_init)

# Print the model
print(netG)
# Let's use the Generator to test functionality
# Generate batch of latent vectors
noise = torch.randn(batch_size, latent_vector_size, 1, 1, device=device)  # Random latent vector
fake_attributes = torch.LongTensor(np.random.randint(0, 1, (batch_size, 40)))  # Random vector of attributes
# Generate fake image batch with the Generator
fake = netG(noise, fake_attributes)
#plt.figure(figsize=(8, 8))
#plt.axis("off")
#plt.title("First Generator outcome.")
#plt.imshow(np.transpose(vutils.make_grid(fake[:112].cpu().detach(), padding=2, normalize=True), (1, 2, 0)))
#plt.show()


discriminator_parameters = {  # Used in the Discriminator's constructor
    'feature_map_size': 64,
    'num_of_channels': number_of_color_channels,
    'number_of_attr': 40
}

# Create the Discriminator
netD = Discriminator(discriminator_parameters).to(device)

# Handle multi-gpu if desired
if(device.type == 'cuda') and (number_of_gpu > 1):
    netD = nn.DataParallel(netD, list(range(number_of_gpu)))

# Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2
netD.apply(weights_init)

# Print the model
print(netD)

# Let's use the Discriminator to test functionality:
# Use the fake images made by the Generator
fake_labels = netD(fake, fake_attributes)
#plt.figure(figsize=(8, 8))
#plt.axis("off")
#plt.title("First Discriminator outcome.")
#plt.hist(fake_labels.flatten().cpu().detach(), 100)
#plt.show()


if __name__ == '__main__':

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    # the progression of the generator
    fixed_noise = torch.randn(2, latent_vector_size, 1, 1, device=device)
    fixed_attributes = torch.LongTensor(np.random.randint(0, 1, (2, 40)))

    # Now the trainning loop

    # Lists to keep track of progress
    img_list = []

    # Number of training epochs
    num_epochs = 1

    # Learning rate for optimizers
    lr = 0.0002

    # Beta hyperparam for Adam optimizers. According to paper should be 0.5
    beta1 = 0.5


    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    solver = GANSolver(netG, netD, optimizerG, optimizerD, criterion)
    solver.train(dataloader, fixed_noise, fixed_attributes, num_epochs, img_list)

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(solver.G_loss_history,label="G")
    plt.plot(solver.D_loss_history,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # %%capture
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())