from __future__ import print_function
# %matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchsummary import summary
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# cloned a git repo called over9000 from https://github.com/mgrankin/over9000
# make the over9000 as source root to prevent import problems
from over9000.over9000 import RangerLars

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
MODEL_NAME = 'dcgan_model'
MODEL_EXT = '.pth'


def save_ckpt(model, optimizer, save_dir, epoch=None, loss=None):
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }

    # composed the model name
    current_model_name = MODEL_NAME
    if epoch is not None:
        current_model_name += '_epoch_%s' % epoch

    # check if the same name already existed in the save directory
    matched_model_files = [os.path.splitext(n)[0] for n in os.listdir(save_dir) if n.startswith(MODEL_NAME)]
    if current_model_name in matched_model_files:
        # change name so we won't override an older checkpoint
        current_model_name += '_1'

    # save model
    ckpt_path = os.path.join(save_dir, current_model_name + MODEL_EXT)
    torch.save(save_dict, ckpt_path)


def _load_ckpt(model, optimizer, ckpt_path):
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


def load_ckpt_for_eval(model, optimizer, ckpt_path):
    model, optimizer, epoch, loss = _load_ckpt(model, optimizer, ckpt_path)
    # set to eval mode
    model.eval()
    return model, optimizer, epoch, loss


def load_ckpt_for_train(model, optimizer, ckpt_path):
    model, optimizer, epoch, loss = _load_ckpt(model, optimizer, ckpt_path)
    # set to eval mode
    model.train()
    return model, optimizer, epoch, loss


# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# Root directory for dataset
dataroot = "Images"

ckptroot = 'models'
g_ckpt_root = os.path.join(ckptroot, 'G')
d_ckpt_root = os.path.join(ckptroot, 'D')

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer. Should be of power of 2!
image_size = 128

# Number of iterations to save the generator output
gen_out_sample = 100

# Number of iterations to show the generator output. After showing the output the queue will be empty
gen_out_flush = 500

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 128

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 32

# Number of training epochs
num_epochs = 1000

# Number of epochs to save the model
save_epochs = 40

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = torch.cuda.device_count()

dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.show()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        layers_list = []
        # compute num of layers from the image size (which should be the output and a power of 2). assuming the input is ZX1X1
        size_exp = int(np.log2(image_size))
        num_hidden_layers = size_exp - 2  # exclude the input and output layers
        # current feature map size multiplier
        fm = 2**(num_hidden_layers-1)

        for i in range(num_hidden_layers):
            if i == 0:
                # insert first hidden layer
                layers_list.extend([
                    nn.ConvTranspose2d(nz, ngf * fm, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(ngf * fm),
                ])
            else:
                layers_list.extend([
                    nn.ConvTranspose2d(ngf * fm * 2, ngf * fm, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf * fm)
                ])
            # add the activation
            layers_list.append(nn.ReLU(True))
            # reduce the multplier by factor of 2
            fm //= 2
        # insert last layer
        layers_list.extend([
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        ])

        # build the model
        self.main = nn.Sequential(*layers_list)

    def forward(self, input):
        return self.main(input)


# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
print(netG)
summary(netG, (nz, 1, 1))


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        layers_list = []
        size_exp = int(np.log2(image_size))
        num_hidden_layers = size_exp - 2
        # current feature map size multiplier
        fm = 1
        for i in range(num_hidden_layers):
            if i==0:
                layers_list.append(nn.Conv2d(nc, ndf * fm, 4, 2, 1, bias=False))
            else:
                layers_list.extend([
                    nn.Conv2d(ndf * fm // 2, ndf * fm, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf * fm)
                ])
            # add the activation
            layers_list.append(nn.LeakyReLU(0.2, inplace=True))
            # increase the multplier by factor of 2
            fm *= 2
        # insert last layer
        layers_list.extend([
            nn.Conv2d(ndf * fm // 2, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        ])

        # build the model
        self.main = nn.Sequential(*layers_list)

    def forward(self, input):
        return self.main(input)


# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)
summary(netD, (nc, image_size, image_size))

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
# optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
# optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Setup optimizers for both G and D
optimizerD = RangerLars(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = RangerLars(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(1, num_epochs + 1):

    if epoch % save_epochs == 0:
        # Save models
        save_ckpt(netD, optimizerD, d_ckpt_root, epoch)
        save_ckpt(netG, optimizerG, g_ckpt_root, epoch)
        print('Save the models for epoch: %d' % epoch)

    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % gen_out_sample == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        # Show the progress by ploting the generator output on the fixed niose
        if iters % gen_out_flush == 0:
            # %%capture
            fig = plt.figure(figsize=(8, 8))
            plt.axis("off")
            ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
            plt.show()
            img_list = []

        iters += 1

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %%capture
fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
plt.show()

# Save models
save_ckpt(netD, optimizerD, d_ckpt_root, epoch)
save_ckpt(netG, optimizerG, g_ckpt_root, epoch)
print('Save the models for epoch: %d' % epoch)
