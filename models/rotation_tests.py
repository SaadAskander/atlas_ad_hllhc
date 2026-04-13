import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import numpy as np
import ad_tools.tools as tools
import matplotlib.pyplot as plt
import awkward as ak 
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import vector as vec
import zuko as zk
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import spconv.pytorch as sp


#print("Testing NFVAE")
class NFVAEEncoder(nn.Module):
    """
    NF-VAE Encoder 
    """
    def __init__(self, latent_dim = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 6, out_channels = 8, kernel_size = 3, padding = (1,0))
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3,padding = (1,0))
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3,padding = (1,0))
        self.pool1 = nn.MaxPool2d(2, return_indices = True)
        self.pool2 = nn.MaxPool2d((5,2), return_indices = True)
        self.pool3 = nn.MaxPool2d((5,2), return_indices = True)
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



        # Latent space mapping
        output13 = self.flatten(output12)
        mu = self.mu(output13)
        logvar = self.logvar(output13)



        return mu, logvar,indices1,indices2,indices3
    
class NFVAEDecoder(nn.Module):
    """
    NF-VAE Decoder.
    """
    def __init__(self, latent_dim = 256):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 256)
    
        self.unpool1 = nn.MaxUnpool2d(kernel_size = (5,2))
        self.transconv1 = nn.ConvTranspose2d(in_channels = 32 , out_channels = 16 ,kernel_size = 3, padding = (1,2))
        


        self.unpool2 = nn.MaxUnpool2d(kernel_size = (5,2))
        self.transconv2 = nn.ConvTranspose2d(in_channels = 16 , out_channels = 8 ,kernel_size = 3, padding = (1,2))
        


        self.unpool3 = nn.MaxUnpool2d(kernel_size = 2)
        self.transconv3 = nn.ConvTranspose2d(in_channels = 8 , out_channels = 6 ,kernel_size = 3, padding = (1,2))        
        self.circular = nn.CircularPad2d((1,1,0,0))
        
    
    def forward(self, latent_vector,indices1, indices2, indices3):
        output1 = self.linear(latent_vector)
        output2 = F.leaky_relu(output1)
        output3 = torch.reshape(output2, shape = (-1,32,1,8))


        
        # Deconvolution Block 1
        output4 = self.unpool1(output3, indices3)
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

class NFVAE(nn.Module):
    """
    NF-VAE using RealNVP
    """

    def __init__(self,latent_dim = 256):
        super().__init__()
        self.encoder = NFVAEEncoder(latent_dim = latent_dim)
        self.decoder = NFVAEDecoder(latent_dim = latent_dim)
        self.flow = zk.flows.NSF(features = latent_dim, transforms = 10, bins = 4, hidden_features = [latent_dim * 4, latent_dim * 4])

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) 
        return mu + eps * std
    
    def forward(self, input):
        mu, logvar, indices1, indices2, indices3 = self.encoder(input)
        std = torch.exp(0.5 * logvar)
        z0 = self.reparameterise(mu, logvar)



        q0 = torch.distributions.Normal(mu, std)

        # Flow transformation
        transform = self.flow().transform
        z = transform(z0)
        log_det = transform.log_abs_det_jacobian(z0, z)

        # KL loss calculation
        p = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z)) # Prior
        log_q0 = torch.sum(q0.log_prob(z0), dim=-1)
        log_p = torch.sum(p.log_prob(z), dim=-1)
        kld = log_q0 - log_det - log_p

        # Decoder Output
        output = self.decoder(z, indices1, indices2, indices3)
        return output,mu,logvar,kld,z


"""
print("4e-1")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NFVAE(latent_dim = 4).to(device)
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/NFVAE1/NFBest/l4/4e_1/weights/NFVAEL4_weights_epoch100.pth"))
tools.test(model = model,
        signal_acceptance_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFVAE1/NFBest/l4/4e_1/signal_acceptance_rates",
        phi_invariance_study_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFVAE1/NFBest/l4/4e_1/phi_invariance_study",
        latent_code_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFVAE1/NFBest/l4/4e_1/latent_vectors",
        testing_losses_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFVAE1/NFBest/l4/4e_1/testing_losses", nf = True)
"""

#print("Testing Vanilla VAE")
class BetaVAEMark3Encoder(nn.Module):
    """
    Beta VAE Mark 3 Encoder 
    """
    def __init__(self, latent_dim = 4, log_clamping = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 6, out_channels = 8, kernel_size = 3, padding = (1,0))
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3,padding = (1,0))
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3,padding = (1,0))
        self.pool1 = nn.MaxPool2d(2, return_indices = True)
        self.pool2 = nn.MaxPool2d((5,2), return_indices = True)
        self.pool3 = nn.MaxPool2d((5,2), return_indices = True)
        self.flatten = nn.Flatten(start_dim = 1)
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)
        self.circular_padding = nn.CircularPad2d((1,1,0,0))
        self.log_clamping = log_clamping


    
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



        # Latent space mapping
        output13 = self.flatten(output12)
        mu = self.mu(output13)
        logvar = self.logvar(output13)
        if self.log_clamping:
            pass
        
        return mu, logvar,indices1,indices2,indices3

    
class BetaVAEMark3Decoder(nn.Module):
    """
    Beta VAE Mark 3  Decoder.
    """
    def __init__(self, latent_dim = 4):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 256)
    
        self.unpool1 = nn.MaxUnpool2d(kernel_size = (5,2))
        self.transconv1 = nn.ConvTranspose2d(in_channels = 32 , out_channels = 16 ,kernel_size = 3, padding = (1,2))
        


        self.unpool2 = nn.MaxUnpool2d(kernel_size = (5,2))
        self.transconv2 = nn.ConvTranspose2d(in_channels = 16 , out_channels = 8 ,kernel_size = 3, padding = (1,2))
        


        self.unpool3 = nn.MaxUnpool2d(kernel_size = 2)
        self.transconv3 = nn.ConvTranspose2d(in_channels = 8 , out_channels = 6 ,kernel_size = 3, padding = (1,2))        
        self.circular = nn.CircularPad2d((1,1,0,0))
        
    
    def forward(self, latent_vector,indices1, indices2, indices3):
        output1 = self.linear(latent_vector)
        output2 = F.leaky_relu(output1)
        output3 = torch.reshape(output2, shape = (-1,32,1,8))


        
        # Deconvolution Block 1
        output4 = self.unpool1(output3, indices3)
        output5 = self.circular(output4)
        output6 = self.transconv1(output5)
        output7 = F.leaky_relu(output6)



        # Deconvolution Block2
        output8 = self.unpool2(output7, indices2)
        output9 = self.circular(output8)
        output10 = self.transconv2(output9)
        output11 = F.leaky_relu(output10)



        # Deconvolution Block8
        output12 = self.unpool3(output11,indices1)
        output13 = self.circular(output12)
        output14 = self.transconv3(output13)
        output15 = F.relu(output14)
        return output15
    
class BetaVAEMark3(nn.Module):
    """
    Beta VAE Mark 3.
    """

    def __init__(self, latent_dim = 4, log_clamping = True):
        super().__init__()
        self.encoder = BetaVAEMark3Encoder(latent_dim = latent_dim, log_clamping = log_clamping)
        self.decoder = BetaVAEMark3Decoder(latent_dim = latent_dim)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) 
        return mu + eps * std
    
    def forward(self, input):
        mu, logvar, indices1, indices2, indices3 = self.encoder(input)
        z = self.reparameterise(mu, logvar)
        output = self.decoder(z, indices1, indices2, indices3)
        return output, mu, logvar,z


"""
print("1e-1")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BetaVAEMark3(latent_dim = 4,log_clamping= False).to(device)
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark3/best_tuning/l4/1e_1/weights/BetaVAEMark3L4_weights_epoch100.pth"))
tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark3/best_tuning/l4/1e_1/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark3/best_tuning/l4/1e_1/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark3/best_tuning/l4/1e_1/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark3/best_tuning/l4/1e_1/testing_losses")
"""

print("Testing sparsevae")
class SparseVAE2(nn.Module):
    """
    Sparse VAE.
    """

    def __init__(self, latent_dim = 4):
        super().__init__()

        # Encoder
        self.subspconv1 = sp.SubMConv2d(in_channels = 6, out_channels = 8, kernel_size = 3)
        self.sppool1 = sp.SparseMaxPool2d(kernel_size= 2, stride = 2, indice_key="1")
        self.subspconv2 = sp.SubMConv2d(in_channels = 8, out_channels = 16, kernel_size = 3)
        self.sppool2 = sp.SparseMaxPool2d(kernel_size= (5,2), stride = (5,2), indice_key="2")
        self.subspconv3 = sp.SubMConv2d(in_channels = 16, out_channels = 32, kernel_size = 3)
        self.sppool3 = sp.SparseMaxPool2d(kernel_size= (5,2), stride = (5,2), indice_key="3")
        self.mu = sp.SparseConv2d(in_channels = 32, out_channels = 1, kernel_size = (1,5), indice_key = "4")
        self.logvar = sp.SparseConv2d(in_channels = 32, out_channels = 1, kernel_size = (1,5))

        # Decoder
        self.unsppool = sp.SparseInverseConv2d(in_channels = 1, out_channels= 32, kernel_size = (1,5), indice_key="4")
        self.unsppool1 = sp.SparseInverseConv2d(in_channels = 32, out_channels = 32, kernel_size = (5,2), indice_key="3")
        self.subspconv4 = sp.SubMConv2d(in_channels = 32 , out_channels = 16 ,kernel_size = 3)
        
        self.unsppool2 = sp.SparseInverseConv2d(in_channels = 16, out_channels = 16, kernel_size = (5,2), indice_key="2")
        self.subspconv5 = sp.SubMConv2d(in_channels = 16 , out_channels = 8 ,kernel_size = 3)
        
        self.unsppool3 = sp.SparseInverseConv2d(in_channels = 8, out_channels = 8,kernel_size = 2, indice_key="1")
        self.subspconv6 = sp.SubMConv2d(in_channels = 8 , out_channels = 6 ,kernel_size = 3)


    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) 
        return mu + eps * std
    
  

    def forward(self, input):
        # Converting to sparse tensor
        input = torch.permute(input, dims = (0,2,3,1))
        input = sp.SparseConvTensor.from_dense(input)

        # Convolution Block 1
        output1 = self.subspconv1(input)
        output2 = output1.replace_feature(F.leaky_relu(output1.features))
        output3 = self.sppool1(output2)



        # Convolutional Block 2
        output4 = self.subspconv2(output3)
        output5 = output4.replace_feature(F.leaky_relu(output4.features))
        output6 = self.sppool2(output5)



        # Convolutional Block 3
        output7 = self.subspconv3(output6)
        output8 = output7.replace_feature(F.leaky_relu(output7.features))
        output9 = self.sppool3(output8)

        # Latent space mapping and reparametrisation
        mu = self.mu(output9)
        logvar = self.logvar(output9)
        z = self.reparameterise(mu.features, logvar.features)
        z = mu.replace_feature(z)


        # Reconstructing the sparse tensor
        output10 = self.unsppool(z)
        output10 = output10.replace_feature(F.leaky_relu(output10.features))
        output12 = self.unsppool1(output10)
        output12 = output12.replace_feature(F.leaky_relu(output12.features))
        output13 = self.subspconv4(output12)
        output13 = output13.replace_feature(F.leaky_relu(output13.features))
        output14 = self.unsppool2(output13)
        output14 = output14.replace_feature(F.leaky_relu(output14.features))
        output15 = self.subspconv5(output14)
        output15 = output15.replace_feature(F.leaky_relu(output15.features))
        output16 = self.unsppool3(output15)
        output16 = output16.replace_feature(F.leaky_relu(output16.features))
        output17 = self.subspconv6(output16)
        output17 = output17.replace_feature(F.leaky_relu(output17.features))


        return output17.dense(), torch.flatten(mu.dense(), start_dim = 1), torch.flatten(logvar.dense(), start_dim = 1),torch.flatten(z.dense(), start_dim = 1)


"""
print("Baseline VAE")
model = BetaVAEMark3(latent_dim = 4).to(device)
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark3/best_tuning/l4/1e_1/weights/BetaVAEMark3L4_weights_epoch100.pth"))
tools.test(model = model,
        signal_acceptance_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark3/best_tuning/l4/1e_1/signal_acceptance_rates",
        phi_invariance_study_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark3/best_tuning/l4/1e_1/phi_invariance_study",
        latent_code_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark3/best_tuning/l4/1e_1/latent_vectors",
        testing_losses_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark3/best_tuning/l4/1e_1/testing_losses")

print("Sparse Rotation")
model = SparseVAE2(latent_dim = 4).to(device)
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/SparseVAE2/7e_2/weights/SparseVAE2_weights_epoch100.pth"))
tools.test(model = model,
        signal_acceptance_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/SparseVAE2/7e_2/signal_acceptance_rates",
        phi_invariance_study_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/SparseVAE2/7e_2/phi_invariance_study",
        latent_code_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/SparseVAE2/7e_2/latent_vectors",
        testing_losses_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/SparseVAE2/7e_2/testing_losses")
"""
print("NFVAE")
model = NFVAE(latent_dim = 4).to(device)
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/NFVAE1/NFBest/l4/4e_1/weights/NFVAEL4_weights_epoch100.pth"))
tools.test(model = model,
        signal_acceptance_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFVAE1/NFBest/l4/4e_1/signal_acceptance_rates",
        phi_invariance_study_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFVAE1/NFBest/l4/4e_1/phi_invariance_study",
        latent_code_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFVAE1/NFBest/l4/4e_1/latent_vectors",
        testing_losses_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFVAE1/NFBest/l4/4e_1/testing_losses", nf=True)