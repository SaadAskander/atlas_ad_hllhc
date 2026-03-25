import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import numpy as np
import ad_tools.tools as tools


class BetaVAEMark5Encoder(nn.Module):
    """
    Beta VAE Mark 3 Encoder 
    """
    def __init__(self, latent_dim = 7):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 6, out_channels = 8, kernel_size = 3, padding = (1,0))
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = (1,0))
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = (1,0))
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d((5,2))
        self.pool3 = nn.MaxPool2d((5,2))
        self.flatten = nn.Flatten(start_dim = 1)
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)
        self.circular_padding = nn.CircularPad2d((1,1,0,0))


    
    def forward(self, input):
        # Convolutional Block 1
        output1 = self.circular_padding(input)
        output2 = self.conv1(output1)
        output3 = F.leaky_relu(output2)
        output4 = self.pool1(output3)



        # Convolutional Block 2
        output5 = self.circular_padding(output4)
        output6 = self.conv2(output5)
        output7 = F.leaky_relu(output6)
        output8  = self.pool2(output7)



        # Convolutional Block 3
        output9 = self.circular_padding(output8)
        output10 = self.conv3(output9)
        output11 = F.leaky_relu(output10)
        output12 = self.pool3(output11)



        # Latent space mapping
        output13 = self.flatten(output12)
        mu = self.mu(output13)
        logvar = self.logvar(output13)
        
        return mu, logvar

class BetaVAEMark5Decoder(nn.Module):
    """
    Beta VAE Mark 3  Decoder.
    """
    def __init__(self, latent_dim = 7):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 256)
    
        self.unpool1 = nn.ConvTranspose2d(in_channels = 32, out_channels = 32, kernel_size = (5,2), stride = (5,2))
        self.transconv1 = nn.ConvTranspose2d(in_channels = 32 , out_channels = 16 ,kernel_size = 3, padding = (1,2))
        


        self.unpool2 = nn.ConvTranspose2d(in_channels = 16, out_channels = 16, kernel_size = (5,2), stride = (5,2))
        self.transconv2 = nn.ConvTranspose2d(in_channels = 16 , out_channels = 8 ,kernel_size = 3, padding = (1,2))
        


        self.unpool3 = nn.ConvTranspose2d(in_channels = 8, out_channels = 8, kernel_size = 2, stride = 2)
        self.transconv3 = nn.ConvTranspose2d(in_channels = 8 , out_channels = 6 ,kernel_size = 3, padding = (1,2))        
        self.circular = nn.CircularPad2d((1,1,0,0))
        
    
    def forward(self, latent_vector):
        output1 = self.linear(latent_vector)
        output2 = F.leaky_relu(output1)
        output3 = torch.reshape(output2, shape = (-1,32,1,8))


        
        # Deconvolution Block 1
        output4 = self.unpool1(output3)
        output5 = self.circular(output4)
        output6 = self.transconv1(output5)
        output7 = F.leaky_relu(output6)



        # Deconvolution Block2
        output8 = self.unpool2(output7)
        output9 = self.circular(output8)
        output10 = self.transconv2(output9)
        output11 = F.leaky_relu(output10)



        # Deconvolution Block8
        output12 = self.unpool3(output11)
        output13 = self.circular(output12)
        output14 = self.transconv3(output13)
        output15 = F.relu(output14)
        return output15

class BetaVAEMark5(nn.Module):
    """
    Beta VAE Mark 3.
    """

    def __init__(self, latent_dim = 7):
        super().__init__()
        self.encoder = BetaVAEMark5Encoder(latent_dim= latent_dim)
        self.decoder = BetaVAEMark5Decoder(latent_dim=latent_dim)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) 
        return mu + eps * std
    
    def forward(self, input):
        mu, logvar = self.encoder(input)
        z = self.reparameterise(mu, logvar)
        output = self.decoder(z)
        return output, mu, logvar,z
    
model = BetaVAEMark5() 
beta = np.zeros(100)
tools.train(project_name = "BetaVAE5beta0", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark5/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark5/training_losses", beta = beta, lr= 1e-4)

tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark5/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark5/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark5/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark5/testing_losses")
