import spconv.pytorch as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import ad_tools.tools as tools
import numpy as np
import awkward as ak
import zuko as zk
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NFSparseVAE(nn.Module):
    """
    Sparse VAE.
    """

    def __init__(self, latent_dim = 4):
        super().__init__()

        # Encoder
        self.subspconv1 = sp.SubMConv2d(in_channels = 6, out_channels = 8, kernel_size = 3)
        self.sppool1 = sp.SparseMaxPool2d(kernel_size= 2, stride = 2, indice_key= "key_1")
        self.subspconv2 = sp.SubMConv2d(in_channels = 8, out_channels = 16, kernel_size = 3)
        self.sppool2 = sp.SparseMaxPool2d(kernel_size= (5,2), stride = (5,2), indice_key= "key_2")
        self.subspconv3 = sp.SubMConv2d(in_channels = 16, out_channels = 32, kernel_size = 3)
        self.sppool3 = sp.SparseMaxPool2d(kernel_size= (5,2), stride = (5,2), indice_key= "key_3")
        self.mu = nn.Linear(32, latent_dim)
        self.logvar = nn.Linear(32, latent_dim)
        self.linear = nn.Linear(latent_dim, 32)

        # Decoder
        self.unsppool1 = sp.SparseInverseConv2d(in_channels = 32, out_channels = 32, kernel_size = (5,2), indice_key = "key_3")
        self.subspconv4 = sp.SubMConv2d(in_channels = 32 , out_channels = 16 ,kernel_size = 3)
        
        self.unsppool2 = sp.SparseInverseConv2d(in_channels = 16, out_channels = 16, kernel_size = (5,2), indice_key = "key_2")
        self.subspconv5 = sp.SubMConv2d(in_channels = 16 , out_channels = 8 ,kernel_size = 3)
        
        self.unsppool3 = sp.SparseInverseConv2d(in_channels = 8, out_channels = 8,kernel_size = 2 , indice_key = "key_1")
        self.subspconv6 = sp.SubMConv2d(in_channels = 8 , out_channels = 6 ,kernel_size = 3)
        self.flow = zk.flows.NSF(features = latent_dim, transforms = 10, bins = 4, hidden_features = [latent_dim * 4, latent_dim * 4])


    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) 
        return mu + eps * std
    
    def custom_kl_loss(self, kld_per_celltower, indices,batch_size):
        """
        Based on https://github.com/mpp-hep/DarkFlow/blob/master/darkflow/utils/network_utils.py
        """


        # Event for each cell tower
        batch_indices = indices[:, 0]

        kld_per_event = torch.zeros(batch_size, device = kld_per_celltower.device)

        # Summing all kld per event
        kld_per_event.index_add_(0,batch_indices,kld_per_celltower)

        return kld_per_event

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

        # Storing structural meta data from last layer of encoder for reconstruction in decoder
        indices = output9.indices
        shape = output9.spatial_shape
        batch_size = output9.batch_size

        # Converting to a dense tensor 
        output10 = output9.features 

        mu = self.mu(output10)
        logvar = self.logvar(output10)
        std = torch.exp(0.5 * logvar)
        z0 = self.reparameterise(mu, logvar)

        
        

        # Flow transformation
        q0 = torch.distributions.Normal(mu, std)
        transform = self.flow().transform
        z = transform(z0)
        log_det = transform.log_abs_det_jacobian(z0, z)

        # KL loss calculation
        p = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z)) # Prior
        log_q0 = torch.sum(q0.log_prob(z0), dim=-1)
        log_p = torch.sum(p.log_prob(z), dim=-1)
        kld_per_celltower = log_q0 - log_det - log_p

        kld = self.custom_kl_loss(kld_per_celltower=kld_per_celltower, indices=indices, batch_size=batch_size)
        output10 = self.linear(z)

        # Reconstructing the sparse tensor
        output11 = output9.replace_feature(output10)
        output12 = self.unsppool1(output11)
        output13 = self.subspconv4(output12)
        output14 = self.unsppool2(output13)
        output15 = self.subspconv5(output14)
        output16 = self.unsppool3(output15)
        output17 = self.subspconv6(output16).dense()

        return output17, mu, logvar,kld,z


print("1e-1")
epochs = 100
beta_max = 1e-1
beta = np.ones(epochs) * beta_max
beta[:50] = np.linspace(0,49,50) * beta_max/50
model = NFSparseVAE(latent_dim = 4).to(device)
tools.train(project_name = "NFSparseVAE", model = model,
        weights_directory_path=f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/1e_1/weights",
        training_losses_directory_path=f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/1e_1/training_losses",
        beta = beta, epochs = epochs, nf = True)
model = NFSparseVAE(latent_dim = 4).to(device)
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/1e_1/weights/NFSparseVAE_weights_epoch100.pth"))
tools.test(model = model,
        signal_acceptance_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/1e_1/signal_acceptance_rates",
        phi_invariance_study_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/1e_1/phi_invariance_study",
        latent_code_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/1e_1/latent_vectors",
        nf = True,testing_losses_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/1e_1/testing_losses")



print("2e-1")
epochs = 100
beta_max = 2e-1
beta = np.ones(epochs) * beta_max
beta[:50] = np.linspace(0,49,50) * beta_max/50
model = NFSparseVAE(latent_dim = 4).to(device)
tools.train(project_name = "NFSparseVAE", model = model,
        weights_directory_path=f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/2e_1/weights",
        training_losses_directory_path=f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/2e_1/training_losses",
        beta = beta, epochs = epochs, nf = True)
model = NFSparseVAE(latent_dim = 4).to(device)
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/2e_1/weights/NFSparseVAE_weights_epoch100.pth"))
tools.test(model = model,
        signal_acceptance_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/2e_1/signal_acceptance_rates",
        phi_invariance_study_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/2e_1/phi_invariance_study",
        latent_code_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/2e_1/latent_vectors",
        nf = True,testing_losses_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/2e_1/testing_losses")



print("3e-1")
epochs = 100
beta_max = 3e-1
beta = np.ones(epochs) * beta_max
beta[:50] = np.linspace(0,49,50) * beta_max/50
model = NFSparseVAE(latent_dim = 4).to(device)
tools.train(project_name = "NFSparseVAE", model = model,
        weights_directory_path=f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/3e_1/weights",
        training_losses_directory_path=f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/3e_1/training_losses",
        beta = beta, epochs = epochs, nf = True)
model = NFSparseVAE(latent_dim = 4).to(device)
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/3e_1/weights/NFSparseVAE_weights_epoch100.pth"))
tools.test(model = model,
        signal_acceptance_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/3e_1/signal_acceptance_rates",
        phi_invariance_study_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/3e_1/phi_invariance_study",
        latent_code_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/3e_1/latent_vectors",
        nf = True,testing_losses_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/3e_1/testing_losses")



print("4e-1")
epochs = 100
beta_max = 4e-1
beta = np.ones(epochs) * beta_max
beta[:50] = np.linspace(0,49,50) * beta_max/50
model = NFSparseVAE(latent_dim = 4).to(device)
tools.train(project_name = "NFSparseVAE", model = model,
        weights_directory_path=f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/4e_1/weights",
        training_losses_directory_path=f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/4e_1/training_losses",
        beta = beta, epochs = epochs, nf = True)
model = NFSparseVAE(latent_dim = 4).to(device)
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/4e_1/weights/NFSparseVAE_weights_epoch100.pth"))
tools.test(model = model,
        signal_acceptance_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/4e_1/signal_acceptance_rates",
        phi_invariance_study_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/4e_1/phi_invariance_study",
        latent_code_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/4e_1/latent_vectors",
        nf = True,testing_losses_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/4e_1/testing_losses")



print("5e-1")
epochs = 100
beta_max = 5e-1
beta = np.ones(epochs) * beta_max
beta[:50] = np.linspace(0,49,50) * beta_max/50
model = NFSparseVAE(latent_dim = 4).to(device)
tools.train(project_name = "NFSparseVAE", model = model,
        weights_directory_path=f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/5e_1/weights",
        training_losses_directory_path=f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/5e_1/training_losses",
        beta = beta, epochs = epochs, nf = True)
model = NFSparseVAE(latent_dim = 4).to(device)
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/5e_1/weights/NFSparseVAE_weights_epoch100.pth"))
tools.test(model = model,
        signal_acceptance_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/5e_1/signal_acceptance_rates",
        phi_invariance_study_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/5e_1/phi_invariance_study",
        latent_code_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/5e_1/latent_vectors",
        nf = True,testing_losses_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/5e_1/testing_losses")

print("6e-1")
epochs = 100
beta_max = 6e-1
beta = np.ones(epochs) * beta_max
beta[:50] = np.linspace(0,49,50) * beta_max/50
model = NFSparseVAE(latent_dim = 4).to(device)
tools.train(project_name = "NFSparseVAE", model = model,
        weights_directory_path=f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/6e_1/weights",
        training_losses_directory_path=f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/6e_1/training_losses",
        beta = beta, epochs = epochs, nf = True)
model = NFSparseVAE(latent_dim = 4).to(device)
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/6e_1/weights/NFSparseVAE_weights_epoch100.pth"))
tools.test(model = model,
        signal_acceptance_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/6e_1/signal_acceptance_rates",
        phi_invariance_study_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/6e_1/phi_invariance_study",
        latent_code_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/6e_1/latent_vectors",
        nf = True,testing_losses_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/6e_1/testing_losses")


print("7e-1")
epochs = 100
beta_max = 7e-1
beta = np.ones(epochs) * beta_max
beta[:50] = np.linspace(0,49,50) * beta_max/50
model = NFSparseVAE(latent_dim = 4).to(device)
tools.train(project_name = "NFSparseVAE", model = model,
        weights_directory_path=f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/7e_1/weights",
        training_losses_directory_path=f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/7e_1/training_losses",
        beta = beta, epochs = epochs, nf = True)
model = NFSparseVAE(latent_dim = 4).to(device)
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/7e_1/weights/NFSparseVAE_weights_epoch100.pth"))
tools.test(model = model,
        signal_acceptance_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/7e_1/signal_acceptance_rates",
        phi_invariance_study_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/7e_1/phi_invariance_study",
        latent_code_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/7e_1/latent_vectors",
        nf = True,testing_losses_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/7e_1/testing_losses")


print("8e-1")
epochs = 100
beta_max = 8e-1
beta = np.ones(epochs) * beta_max
beta[:50] = np.linspace(0,49,50) * beta_max/50
model = NFSparseVAE(latent_dim = 4).to(device)
tools.train(project_name = "NFSparseVAE", model = model,
        weights_directory_path=f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/8e_1/weights",
        training_losses_directory_path=f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/8e_1/training_losses",
        beta = beta, epochs = epochs, nf = True)
model = NFSparseVAE(latent_dim = 4).to(device)
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/8e_1/weights/NFSparseVAE_weights_epoch100.pth"))
tools.test(model = model,
        signal_acceptance_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/8e_1/signal_acceptance_rates",
        phi_invariance_study_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/8e_1/phi_invariance_study",
        latent_code_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/8e_1/latent_vectors",
        nf = True,testing_losses_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/8e_1/testing_losses")


print("9e-1")
epochs = 100
beta_max = 9e-1
beta = np.ones(epochs) * beta_max
beta[:50] = np.linspace(0,49,50) * beta_max/50
model = NFSparseVAE(latent_dim = 4).to(device)
tools.train(project_name = "NFSparseVAE", model = model,
        weights_directory_path=f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/9e_1/weights",
        training_losses_directory_path=f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/9e_1/training_losses",
        beta = beta, epochs = epochs, nf = True)
model = NFSparseVAE(latent_dim = 4).to(device)
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/9e_1/weights/NFSparseVAE_weights_epoch100.pth"))
tools.test(model = model,
        signal_acceptance_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/9e_1/signal_acceptance_rates",
        phi_invariance_study_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/9e_1/phi_invariance_study",
        latent_code_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/9e_1/latent_vectors",
        nf = True,testing_losses_directory = f"/home/xzcapask/atlas_ad_hllhc/data/model_data/NFSparseVAE/9e_1/testing_losses")


