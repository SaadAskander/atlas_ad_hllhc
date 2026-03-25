import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import numpy as np
import ad_tools.tools as tools

class BetaVAEMark3_1Encoder(nn.Module):
    """
    Beta VAE Mark 3 Encoder 
    """
    def __init__(self, latent_dim = 7):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 6, out_channels = 8, kernel_size = 3, padding = (1,0))
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3,padding = (1,0))
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3,padding = (1,0))
        self.pool1 = nn.MaxPool2d(2, return_indices = True)
        self.pool2 = nn.MaxPool2d((5,2), return_indices = True)
        self.pool3 = nn.MaxPool2d((5,2), return_indices = True)
        self.conv4 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3,padding = (1,0))
        self.pool4 = nn.MaxPool2d((1,2), return_indices = True)
        self.flatten = nn.Flatten(start_dim = 1)
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)
        self.circular_padding = nn.CircularPad2d((1,1,0,0))


    
    def forward(self, input):
        # Convolutional Block 1
        output1 = self.circular_padding(input)
        output2 = self.conv1(output1)
        output3 = F.leaky_relu(output2)
        output4, indices1 = self.pool1(output3)



        # Convolutional Block 2
        output5 = self.circular_padding(output4)
        output6 = self.conv2(output5)
        output7 = F.leaky_relu(output6)
        output8 , indices2 = self.pool2(output7)



        # Convolutional Block 3
        output9 = self.circular_padding(output8)
        output10 = self.conv3(output9)
        output11 = F.leaky_relu(output10)
        output12, indices3 = self.pool3(output11)

        # Convolutional Block 4
        output13 = self.circular_padding(output12)
        output14 = self.conv4(output13)
        output15 = F.leaky_relu(output14)
        output16, indices4 = self.pool4(output15)


        # Latent space mapping
        output17 = self.flatten(output16)
        mu = self.mu(output17)
        logvar = self.logvar(output17)
        
        return mu, logvar,indices1,indices2,indices3, indices4

    
class BetaVAEMark3_1Decoder(nn.Module):
    """
    Beta VAE Mark 3  Decoder.
    """
    def __init__(self, latent_dim = 7):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 256)

        self.unpool0 = nn.MaxUnpool2d(kernel_size = (1,2))
        self.transconv0 = nn.ConvTranspose2d(in_channels = 64 , out_channels = 32 ,kernel_size = 3, padding = (1,2))

        self.unpool1 = nn.MaxUnpool2d(kernel_size = (5,2))
        self.transconv1 = nn.ConvTranspose2d(in_channels = 32 , out_channels = 16 ,kernel_size = 3, padding = (1,2))
        


        self.unpool2 = nn.MaxUnpool2d(kernel_size = (5,2))
        self.transconv2 = nn.ConvTranspose2d(in_channels = 16 , out_channels = 8 ,kernel_size = 3, padding = (1,2))
        


        self.unpool3 = nn.MaxUnpool2d(kernel_size = 2)
        self.transconv3 = nn.ConvTranspose2d(in_channels = 8 , out_channels = 6 ,kernel_size = 3, padding = (1,2))        
        self.circular = nn.CircularPad2d((1,1,0,0))
        
    
    def forward(self, latent_vector,indices1, indices2, indices3, indices4):
        output1 = self.linear(latent_vector)
        output2 = F.leaky_relu(output1)
        output3 = torch.reshape(output2, shape = (-1,64,1,4))


        # Deconvolution Block 0
        output4 = self.unpool0(output3, indices4)
        output5 = self.circular(output4)
        output6 = self.transconv0(output5)
        output7 = F.leaky_relu(output6)

        # Deconvolution Block 1
        output4 = self.unpool1(output7, indices3)
        output5 = self.circular(output4)
        output6 = self.transconv1(output5)
        output7 = F.leaky_relu(output6)



        # Deconvolution Block2
        output8 = self.unpool2(output7, indices2)
        output9 = self.circular(output8)
        output10 = self.transconv2(output9)
        output11 = F.leaky_relu(output10)



        # Deconvolution Block8
        output12 = self.unpool3(output11, indices1)
        output13 = self.circular(output12)
        output14 = self.transconv3(output13)
        output15 = F.relu(output14)
        return output15
    
class BetaVAEMark3_1(nn.Module):
    """
    Beta VAE Mark 3.
    """

    def __init__(self, latent_dim = 256):
        super().__init__()
        self.encoder = BetaVAEMark3_1Encoder(latent_dim= latent_dim)
        self.decoder = BetaVAEMark3_1Decoder(latent_dim=latent_dim)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) 
        return mu + eps * std
    
    def forward(self, input):
        mu, logvar, indices1, indices2, indices3, indices4 = self.encoder(input)
        z = self.reparameterise(mu, logvar)
        output = self.decoder(z, indices1, indices2, indices3, indices4)
        return output, mu, logvar,z
    
class BetaVAEMark3_2Encoder(nn.Module):
    """
    Beta VAE Mark 3 Encoder 
    """
    def __init__(self, latent_dim = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 6, out_channels = 8, kernel_size = 3, padding = (1,0))
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3,padding = (1,0))
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3,padding = (1,0))
        self.pool1 = nn.MaxPool2d(2, return_indices = True)
        self.pool2 = nn.MaxPool2d((5,2), return_indices = True)
        self.pool3 = nn.MaxPool2d((5,2), return_indices = True)
        self.conv4 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3,padding = (1,0))
        self.pool4 = nn.MaxPool2d((1,2), return_indices = True)
        self.flatten = nn.Flatten(start_dim = 1)
        self.linear1 = nn.Linear(256,128)
        self.linear2  = nn.Linear(128,64)
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)
        self.circular_padding = nn.CircularPad2d((1,1,0,0))


    
    def forward(self, input):
        # Convolutional Block 1
        output1 = self.circular_padding(input)
        output2 = self.conv1(output1)
        output3 = F.leaky_relu(output2)
        output4, indices1 = self.pool1(output3)



        # Convolutional Block 2
        output5 = self.circular_padding(output4)
        output6 = self.conv2(output5)
        output7 = F.leaky_relu(output6)
        output8 , indices2 = self.pool2(output7)



        # Convolutional Block 3
        output9 = self.circular_padding(output8)
        output10 = self.conv3(output9)
        output11 = F.leaky_relu(output10)
        output12, indices3 = self.pool3(output11)

        # Convolutional Block 4
        output13 = self.circular_padding(output12)
        output14 = self.conv4(output13)
        output15 = F.leaky_relu(output14)
        output16, indices4 = self.pool4(output15)


        # Latent space mapping
        output17 = self.flatten(output16)
        output18 = self.linear1(output17)
        output19 = self.linear2(output18)
        mu = self.mu(output19)
        logvar = self.logvar(output19)
        
        return mu, logvar,indices1,indices2,indices3, indices4

    
class BetaVAEMark3_2Decoder(nn.Module):
    """
    Beta VAE Mark 3  Decoder.
    """
    def __init__(self, latent_dim = 16):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 256)

        self.unpool0 = nn.MaxUnpool2d(kernel_size = (1,2))
        self.transconv0 = nn.ConvTranspose2d(in_channels = 64 , out_channels = 32 ,kernel_size = 3, padding = (1,2))

        self.unpool1 = nn.MaxUnpool2d(kernel_size = (5,2))
        self.transconv1 = nn.ConvTranspose2d(in_channels = 32 , out_channels = 16 ,kernel_size = 3, padding = (1,2))
        


        self.unpool2 = nn.MaxUnpool2d(kernel_size = (5,2))
        self.transconv2 = nn.ConvTranspose2d(in_channels = 16 , out_channels = 8 ,kernel_size = 3, padding = (1,2))
        


        self.unpool3 = nn.MaxUnpool2d(kernel_size = 2)
        self.transconv3 = nn.ConvTranspose2d(in_channels = 8 , out_channels = 6 ,kernel_size = 3, padding = (1,2))        
        self.circular = nn.CircularPad2d((1,1,0,0))
        
    
    def forward(self, latent_vector,indices1, indices2, indices3, indices4):
        output1 = self.linear(latent_vector)
        output2 = F.leaky_relu(output1)
        output3 = torch.reshape(output2, shape = (-1,64,1,4))


        # Deconvolution Block 0
        output4 = self.unpool0(output3, indices4)
        output5 = self.circular(output4)
        output6 = self.transconv0(output5)
        output7 = F.leaky_relu(output6)

        # Deconvolution Block 1
        output4 = self.unpool1(output7, indices3)
        output5 = self.circular(output4)
        output6 = self.transconv1(output5)
        output7 = F.leaky_relu(output6)



        # Deconvolution Block2
        output8 = self.unpool2(output7, indices2)
        output9 = self.circular(output8)
        output10 = self.transconv2(output9)
        output11 = F.leaky_relu(output10)



        # Deconvolution Block8
        output12 = self.unpool3(output11, indices1)
        output13 = self.circular(output12)
        output14 = self.transconv3(output13)
        output15 = F.relu(output14)
        return output15
    
class BetaVAEMark3_2(nn.Module):
    """
    Beta VAE Mark 3.
    """

    def __init__(self, latent_dim = 64):
        super().__init__()
        self.encoder = BetaVAEMark3_2Encoder(latent_dim= latent_dim)
        self.decoder = BetaVAEMark3_2Decoder(latent_dim=latent_dim)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) 
        return mu + eps * std
    
    def forward(self, input):
        mu, logvar, indices1, indices2, indices3, indices4 = self.encoder(input)
        z = self.reparameterise(mu, logvar)
        output = self.decoder(z, indices1, indices2, indices3, indices4)
        return output, mu, logvar,z
    
class BetaVAEMark3_3Encoder(nn.Module):
    """
    Beta VAE Mark 3 Encoder 
    """
    def __init__(self, latent_dim = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 6, out_channels = 8, kernel_size = 3, padding = (1,0))
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3,padding = (1,0))
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3,padding = (1,0))
        self.pool1 = nn.MaxPool2d(2, return_indices = True)
        self.pool2 = nn.MaxPool2d((5,2), return_indices = True)
        self.pool3 = nn.MaxPool2d((5,2), return_indices = True)
        self.conv4 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3,padding = (1,0))
        self.pool4 = nn.MaxPool2d((1,2), return_indices = True)
        self.flatten = nn.Flatten(start_dim = 1)
        self.linear1 = nn.Linear(256,128)
        self.linear2  = nn.Linear(128,64)
        self.linear3 = nn.Linear(64,32)
        self.mu = nn.Linear(32, latent_dim)
        self.logvar = nn.Linear(32, latent_dim)
        self.circular_padding = nn.CircularPad2d((1,1,0,0))


    
    def forward(self, input):
        # Convolutional Block 1
        output1 = self.circular_padding(input)
        output2 = self.conv1(output1)
        output3 = F.leaky_relu(output2)
        output4, indices1 = self.pool1(output3)



        # Convolutional Block 2
        output5 = self.circular_padding(output4)
        output6 = self.conv2(output5)
        output7 = F.leaky_relu(output6)
        output8 , indices2 = self.pool2(output7)



        # Convolutional Block 3
        output9 = self.circular_padding(output8)
        output10 = self.conv3(output9)
        output11 = F.leaky_relu(output10)
        output12, indices3 = self.pool3(output11)

        # Convolutional Block 4
        output13 = self.circular_padding(output12)
        output14 = self.conv4(output13)
        output15 = F.leaky_relu(output14)
        output16, indices4 = self.pool4(output15)


        # Latent space mapping
        output17 = self.flatten(output16)
        output18 = self.linear1(output17)
        output18 = F.leaky_relu(output18)
        output19 = self.linear2(output18)
        output19 = F.leaky_relu(output19)
        output20 = self.linear3(output19)
        output21 = F.leaky_relu(output20)
        mu = self.mu(output21)
        logvar = self.logvar(output21)
        
        return mu, logvar,indices1,indices2,indices3, indices4

    
class BetaVAEMark3_3Decoder(nn.Module):
    """
    Beta VAE Mark 3  Decoder.
    """
    def __init__(self, latent_dim = 32):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 256)

        self.unpool0 = nn.MaxUnpool2d(kernel_size = (1,2))
        self.transconv0 = nn.ConvTranspose2d(in_channels = 64 , out_channels = 32 ,kernel_size = 3, padding = (1,2))

        self.unpool1 = nn.MaxUnpool2d(kernel_size = (5,2))
        self.transconv1 = nn.ConvTranspose2d(in_channels = 32 , out_channels = 16 ,kernel_size = 3, padding = (1,2))
        


        self.unpool2 = nn.MaxUnpool2d(kernel_size = (5,2))
        self.transconv2 = nn.ConvTranspose2d(in_channels = 16 , out_channels = 8 ,kernel_size = 3, padding = (1,2))
        


        self.unpool3 = nn.MaxUnpool2d(kernel_size = 2)
        self.transconv3 = nn.ConvTranspose2d(in_channels = 8 , out_channels = 6 ,kernel_size = 3, padding = (1,2))        
        self.circular = nn.CircularPad2d((1,1,0,0))
        
    
    def forward(self, latent_vector,indices1, indices2, indices3, indices4):
        output1 = self.linear(latent_vector)
        output2 = F.leaky_relu(output1)
        output3 = torch.reshape(output2, shape = (-1,64,1,4))


        # Deconvolution Block 0
        output4 = self.unpool0(output3, indices4)
        output5 = self.circular(output4)
        output6 = self.transconv0(output5)
        output7 = F.leaky_relu(output6)

        # Deconvolution Block 1
        output4 = self.unpool1(output7, indices3)
        output5 = self.circular(output4)
        output6 = self.transconv1(output5)
        output7 = F.leaky_relu(output6)



        # Deconvolution Block2
        output8 = self.unpool2(output7, indices2)
        output9 = self.circular(output8)
        output10 = self.transconv2(output9)
        output11 = F.leaky_relu(output10)



        # Deconvolution Block8
        output12 = self.unpool3(output11, indices1)
        output13 = self.circular(output12)
        output14 = self.transconv3(output13)
        output15 = F.relu(output14)
        return output15
    
class BetaVAEMark3_3(nn.Module):
    """
    Beta VAE Mark 3.
    """

    def __init__(self, latent_dim = 32):
        super().__init__()
        self.encoder = BetaVAEMark3_3Encoder(latent_dim= latent_dim)
        self.decoder = BetaVAEMark3_3Decoder(latent_dim=latent_dim)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) 
        return mu + eps * std
    
    def forward(self, input):
        mu, logvar, indices1, indices2, indices3, indices4 = self.encoder(input)
        z = self.reparameterise(mu, logvar)
        output = self.decoder(z, indices1, indices2, indices3, indices4)
        return output, mu, logvar,z
    

print("Beta VAE Mark3_1")
print("Beta = 1")
model = BetaVAEMark3_1() 
beta_upper = 1
beta = np.ones(100) * beta_upper
beta[:50] = np.linspace(0,49,50) * beta_upper/50
tools.train(project_name = "Architecture_Tests", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/architectures_tests/BetaVAEMark3_1/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/architectures_tests/BetaVAEMark3_1/training_losses", beta = beta)
tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/architectures_tests/BetaVAEMark3_1/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/architectures_tests/BetaVAEMark3_1/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/architectures_tests/BetaVAEMark3_1/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/architectures_tests/BetaVAEMark3_1/testing_losses")  



print("Beta VAE Mark3_2")
print("Beta = 1")
model = BetaVAEMark3_2() 
beta_upper = 1
beta = np.ones(100) * beta_upper
beta[:50] = np.linspace(0,49,50) * beta_upper/50
tools.train(project_name = "Architecture_Tests", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/architectures_tests/BetaVAEMark3_2/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/architectures_tests/BetaVAEMark3_2/training_losses", beta = beta)
tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/architectures_tests/BetaVAEMark3_2/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/architectures_tests/BetaVAEMark3_2/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/architectures_tests/BetaVAEMark3_2/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/architectures_tests/BetaVAEMark3_2/testing_losses")  



print("Beta VAE Mark3_3")
print("Beta = 1")
model = BetaVAEMark3_3() 
beta_upper = 1
beta = np.ones(100) * beta_upper
beta[:50] = np.linspace(0,49,50) * beta_upper/50
tools.train(project_name = "Architecture_Tests", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/architectures_tests/BetaVAEMark3_3/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/architectures_tests/BetaVAEMark3_3/training_losses", beta = beta)
tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/architectures_tests/BetaVAEMark3_3/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/architectures_tests/BetaVAEMark3_3/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/architectures_tests/BetaVAEMark3_3/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/architectures_tests/BetaVAEMark3_3/testing_losses")  
