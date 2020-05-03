import numpy as np
import torch.optim
import torch.nn as nn
import torchvision.utils as vutils

class GANSolver(object):

    def __init__(self, gen, dis, optimG=torch.optim.Adam, optimD=torch.optim.Adam,
                       loss_func=nn.CrossEntropyLoss()):
        self.generator = gen
        self.discriminator = dis
        self.optimizerD = optimD
        self.optimizerG = optimG
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.G_loss_history = []
        self.D_loss_history = []

    def train(self, dataloader, fixed_noise, fixed_attributes, num_epochs=10, img_list=[]):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Establish convention for real and fake labels during training
        real_label = 1
        fake_label = 0

        latent_vector_size = 100

        iters = 0

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(dataloader, 0):

                # 1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                # this means minimize D(G(z))
                # Train with all-real batch
                self.discriminator.zero_grad()
                # Format batch
                images_device = data['image'].to(device)
                attributes_device = data['attributes'].to(device)
                b_size = images_device.size(0)
                label = torch.full((b_size,), real_label, device=device)
                # Forward pass real batch through D
                output = self.discriminator(images_device, attributes_device).flatten()
                # Calculate loss on all-real batch
                errD_real = self.loss_func(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                # Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, latent_vector_size, 1, 1, device=device)
                fake_attributes = torch.LongTensor(np.random.randint(0, 1, (b_size, 40)))
                # Generate fake image batch with the Generator
                fake = self.generator(noise, fake_attributes)
                label.fill_(fake_label)
                # Classify all fake batch with the Discriminator
                output = self.discriminator(fake.detach(), fake_attributes).flatten()
                # Calculate D's loss on the all-fake batch
                errD_fake = self.loss_func(output, label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                self.optimizerD.step()

                # 2) Update G network: maximize log(D(G(z)))
                self.generator.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.discriminator(fake.detach(), fake_attributes).flatten()
                # Calculate G's loss based on this output
                errG = self.loss_func(output, label)
                # Calcualte gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\t Loss_D: %.4f\t Loss_G: %4.f\t D(x):%.4f\t D(G(z)): %.4f/%.4f' % (
                    epoch, num_epochs,
                    i, len(dataloader),
                    errD.item(), errG.item(),
                    D_x, D_G_z1, D_G_z2))

                # Save losses for plotting later
                self.G_loss_history.append(errG.item())
                self.D_loss_history.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                    with torch.no_grad():
                        fake = self.generator(fixed_noise, fixed_attributes).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1

