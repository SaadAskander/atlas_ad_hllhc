import spconv.pytorch as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import ad_tools.tools as tools
import numpy as np
import awkward as ak
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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



print("7e-2")
model = SparseVAE2(latent_dim = 4).to(device)
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/SparseVAE2/7e_1/weights/SparseVAE2_weights_epoch100.pth"))
tools.test(model = model,
        signal_acceptance_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/SparseVAE2/7e_1/signal_acceptance_rates",
        phi_invariance_study_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/SparseVAE2/7e_1/phi_invariance_study",
        latent_code_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/SparseVAE2/7e_1/latent_vectors",
        testing_losses_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/SparseVAE2/7e_1/testing_losses")

