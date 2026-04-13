import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import numpy as np
import ad_tools.tools as tools
import spconv.pytorch as sp


class BetaVAEMark10Encoder(nn.Module):
    """
    Beta VAE Mark 4 Encoder
    """

    def __init__(self, latent_dim = 4):
        super().__init__()
        self.submanspconv1 = sp.SubMConv2d(in_channels = 6, out_channels = 8, kernel_size = 3)
        self.sppool1 = sp.SparseMaxPool2d(kernel_size= 2, stride = 2)
        self.submanspconv2 = sp.SubMConv2d(in_channels = 8, out_channels = 16, kernel_size = 3)
        self.sppool2 = sp.SparseMaxPool2d(kernel_size= (5,2), stride = (5,2))
        self.submanspconv3 = sp.SubMConv2d(in_channels = 16, out_channels = 32, kernel_size = 3)
        self.sppool3 = sp.SparseMaxPool2d(kernel_size= (5,2), stride = (5,2))
        self.flatten = nn.Flatten(start_dim = 1)
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)


    def forward(self, input):
        # Convolutional Block 1
        input = torch.permute(input, dims = (0,2,3,1))
        input = sp.SparseConvTensor.from_dense(input)
        output1 = self.submanspconv1(input)
        output2 = output1.replace_feature(F.leaky_relu(output1.features))
        output3 = self.sppool1(output2)



        # Convolutional Block 2
        output4 = self.submanspconv2(output3)
        output5 = output4.replace_feature(F.leaky_relu(output4.features))
        output6 = self.sppool2(output5)



        # Convolutional Block 3
        output7 = self.submanspconv3(output6)
        output8 = output7.replace_feature(F.leaky_relu(output7.features))
        output9 = self.sppool3(output8)



        # Latent space mapping
        output10 = output9.dense()
        output11 = self.flatten(output10)
        mu = self.mu(output11)
        logvar = self.logvar(output11)
        
        return mu, logvar
    

class BetaVAEMark10Decoder(nn.Module):
    """
    Beta VAE Mark 4 Decoder
    """



    def __init__(self, latent_dim = 4):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 256)
    
        self.unpool1 = sp.SparseConvTranspose2d(in_channels = 32, out_channels = 32,  kernel_size = (5,2), stride = (5,2))
        self.transconv1 = sp.SubMConv2d(in_channels = 32 , out_channels = 16 ,kernel_size = 3)
        
        self.unpool2 = sp.SparseConvTranspose2d(in_channels = 16, out_channels = 16, kernel_size = (5,2), stride = (5,2))
        self.transconv2 = sp.SubMConv2d(in_channels = 16 , out_channels = 8 ,kernel_size = 3)
        
        self.unpool3 = sp.SparseConvTranspose2d(in_channels = 8, out_channels = 8,kernel_size = 2, stride = 2)
        self.transconv3 = sp.SubMConv2d(in_channels = 8 , out_channels = 6 ,kernel_size = 3)



    def forward(self, latent_vector):
        output1 = self.linear(latent_vector)
        output2 = F.leaky_relu(output1)
        output3 = torch.permute(torch.reshape(output2, shape = (-1,32,1,8)), dims = (0,2,3,1))
        output4 = sp.SparseConvTensor.from_dense(output3)


        
        # Deconvolution Block 1
        output5 = self.unpool1(output4)
        output6 = self.transconv1(output5)
        output7 = output6.replace_feature(F.leaky_relu(output6.features))



        # Deconvolution Block2
        output8 = self.unpool2(output7)
        output9 = self.transconv2(output8)
        output10 = output9.replace_feature(F.leaky_relu(output9.features))



        # Deconvolution Block8
        output11 = self.unpool3(output10)
        output12 = self.transconv3(output11)
        output13 = output12.replace_feature(F.relu(output12.features))
        output14 = output13.dense()
        return output14
    

class BetaVAEMark10(nn.Module):
    """
    Beta VAE Mark 4.
    """

    def __init__(self, latent_dim = 4):
        super().__init__()
        self.encoder = BetaVAEMark10Encoder(latent_dim= latent_dim)
        self.decoder = BetaVAEMark10Decoder(latent_dim=latent_dim)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) 
        return mu + eps * std
    
    def forward(self, input):
        mu, logvar = self.encoder(input)
        z = self.reparameterise(mu, logvar)
        output = self.decoder(z)
        return output, mu, logvar,z


epochs = 100
model = BetaVAEMark10() 
beta_max = 1e-1
beta = np.ones(epochs) * beta_max
tools.train(project_name = "BetaVAEMark10", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/1e_1/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/1e_1/training_losses", beta = beta, epochs= epochs, lr = 1e-4)

tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/1e_1/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/1e_1/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/1e_1/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/1e_1/testing_losses")

model = BetaVAEMark10() 
beta_max = 2e-1
beta = np.ones(epochs) * beta_max
tools.train(project_name = "BetaVAEMark10", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/2e_1/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/2e_1/training_losses", beta = beta, epochs= epochs, lr = 1e-4)

tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/2e_1/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/2e_1/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/2e_1/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/2e_1/testing_losses")

model = BetaVAEMark10() 
beta_max = 3e-1
beta = np.ones(epochs) * beta_max
tools.train(project_name = "BetaVAEMark10", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/3e_1/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/3e_1/training_losses", beta = beta, epochs= epochs, lr = 1e-4)

tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/3e_1/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/3e_1/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/3e_1/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/3e_1/testing_losses")


model = BetaVAEMark10() 
beta_max = 4e-1
beta = np.ones(epochs) * beta_max
tools.train(project_name = "BetaVAEMark10", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/4e_1/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/4e_1/training_losses", beta = beta, epochs= epochs, lr = 1e-4)

tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/4e_1/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/4e_1/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/4e_1/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/4e_1/testing_losses")


model = BetaVAEMark10() 
beta_max = 5e-1
beta = np.ones(epochs) * beta_max
tools.train(project_name = "BetaVAEMark10", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/5e_1/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/5e_1/training_losses", beta = beta, epochs= epochs, lr = 1e-4)

tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/5e_1/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/5e_1/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/5e_1/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/5e_1/testing_losses")


model = BetaVAEMark10() 
beta_max = 6e-1
beta = np.ones(epochs) * beta_max
tools.train(project_name = "BetaVAEMark10", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/6e_1/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/6e_1/training_losses", beta = beta, epochs= epochs, lr = 1e-4)

tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/6e_1/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/6e_1/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/6e_1/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/6e_1/testing_losses")




model = BetaVAEMark10() 
beta_max = 7e-1
beta = np.ones(epochs) * beta_max
tools.train(project_name = "BetaVAEMark10", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/7e_1/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/7e_1/training_losses", beta = beta, epochs= epochs, lr = 1e-4)

tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/7e_1/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/7e_1/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/7e_1/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/7e_1/testing_losses")




model = BetaVAEMark10() 
beta_max = 8e-1
beta = np.ones(epochs) * beta_max
tools.train(project_name = "BetaVAEMark10", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/8e_1/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/8e_1/training_losses", beta = beta, epochs= epochs, lr = 1e-4)

tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/8e_1/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/8e_1/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/8e_1/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/8e_1/testing_losses")




model = BetaVAEMark10() 
beta_max = 9e-1
beta = np.ones(epochs) * beta_max
tools.train(project_name = "BetaVAEMark10", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/9e_1/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/9e_1/training_losses", beta = beta, epochs= epochs, lr = 1e-4)

tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/9e_1/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/9e_1/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/9e_1/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/9e_1/testing_losses")


model = BetaVAEMark10() 
beta_max = 10e-1
beta = np.ones(epochs) * beta_max
tools.train(project_name = "BetaVAEMark10", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/10e_1/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/10e_1/training_losses", beta = beta, epochs= epochs, lr = 1e-4)

tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/10e_1/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/10e_1/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/10e_1/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark10/10e_1/testing_losses")
