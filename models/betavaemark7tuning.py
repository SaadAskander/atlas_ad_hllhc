import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import numpy as np
import ad_tools.tools as tools
import spconv.pytorch as sp

class BetaVAEMark7Encoder(nn.Module):
    """
    Beta VAE Mark 4 Encoder
    """

    def __init__(self, latent_dim = 7, log_clamping = True):
        super().__init__()
        self.submanspconv1 = sp.SubMConv2d(in_channels = 6, out_channels = 8, kernel_size = 3)
        self.sppool1 = sp.SparseConv2d(in_channels = 8, out_channels = 8,kernel_size= 2, stride = 2)
        self.submanspconv2 = sp.SubMConv2d(in_channels = 8, out_channels = 16, kernel_size = 3)
        self.sppool2 = sp.SparseConv2d(in_channels = 16, out_channels = 16, kernel_size= (5,2), stride = (5,2))
        self.submanspconv3 = sp.SubMConv2d(in_channels = 16, out_channels = 32, kernel_size = 3)
        self.sppool3 = sp.SparseConv2d(in_channels = 32, out_channels = 32, kernel_size= (5,2), stride = (5,2))
        self.flatten = nn.Flatten(start_dim = 1)
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)
        self.logvar_clamp = log_clamping


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
        if self.logvar_clamp:
            logvar = torch.clamp(logvar,-5,0)
        
        return mu, logvar
    

class BetaVAEMark7Decoder(nn.Module):
    """
    Beta VAE Mark 4 Decoder
    """



    def __init__(self, latent_dim = 7):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 256)
    
        self.unpool1 = sp.SparseConvTranspose2d(in_channels = 32, out_channels = 32, kernel_size = (5,2), stride = 2)
        self.transconv1 = sp.SparseConvTranspose2d(in_channels = 32 , out_channels = 16 ,kernel_size = 3, padding = (1,1))
        
        self.unpool2 = sp.SparseConvTranspose2d(in_channels = 16, out_channels = 16, kernel_size = (5,2), stride = (5,2))
        self.transconv2 = sp.SparseConvTranspose2d(in_channels = 16 , out_channels = 8 ,kernel_size = 3, padding = (1,1))
        
        self.unpool3 = sp.SparseConvTranspose2d(in_channels = 8, out_channels = 8,kernel_size = 2, stride = 2)
        self.transconv3 = sp.SparseConvTranspose2d(in_channels = 8 , out_channels = 6 ,kernel_size = 3, padding = (1,1))



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
        output13 = output12.replace_feature(F.leaky_relu(output12.features))
        output14 = output13.dense()
        return output14
    

class BetaVAEMark7(nn.Module):
    """
    Beta VAE Mark 4.
    """

    def __init__(self, latent_dim = 7, log_clamping = True):
        super().__init__()
        self.encoder = BetaVAEMark7Encoder(latent_dim= latent_dim, log_clamping = log_clamping)
        self.decoder = BetaVAEMark7Decoder(latent_dim=latent_dim)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) 
        return mu + eps * std
    
    def forward(self, input):
        mu, logvar = self.encoder(input)
        z = self.reparameterise(mu, logvar)
        output = self.decoder(z)
        return output, mu, logvar,z

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')






"""
# L7 tests
# Beta 1e-1
print("L7 1e-1")
model = BetaVAEMark7(latent_dim = 7)
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/beta0_l7/weights/BetaVAE7_weights_epoch300.pth"))
beta = np.ones(100) * 1e-1
beta[:50] = np.linspace(0,49,50) * 1e-1/50
tools.train(project_name = "BetaVAEMark7", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l7_b1/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l7_b1/training_losses", beta = beta)



model = BetaVAEMark7(latent_dim = 7,log_clamping= False).to(device)
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l7_b1/weights/BetaVAEMark7_weights_epoch100.pth"))
tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l7_b1/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l7_b1/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l7_b1/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l7_b1/testing_losses")



# Beta 1e-2
print("L7 1e-2")
model = BetaVAEMark7(latent_dim = 7)
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/beta0_l7/weights/BetaVAE7_weights_epoch300.pth"))
beta = np.ones(100) * 1e-2
beta[:50] = np.linspace(0,49,50) * 1e-2/50
tools.train(project_name = "BetaVAEMark7", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l7_b2/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l7_b2/training_losses", beta = beta)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BetaVAEMark7(latent_dim = 7,log_clamping= False).to(device)
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l7_b2/weights/BetaVAEMark7_weights_epoch100.pth"))
tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l7_b2/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l7_b2/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l7_b2/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l7_b2/testing_losses")


# Beta 1e-3
print("L7 1e-3")
model = BetaVAEMark7(latent_dim = 7) 
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/beta0_l7/weights/BetaVAE7_weights_epoch300.pth"))
beta = np.ones(100) * 1e-3
beta[:50] = np.linspace(0,49,50) * 1e-3/50
tools.train(project_name = "BetaVAEMark7", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l7_b3/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l7_b3/training_losses", beta = beta)

model = BetaVAEMark7(latent_dim = 7,log_clamping= False).to(device)
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l7_b3/weights/BetaVAEMark7_weights_epoch100.pth"))
tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l7_b3/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l7_b3/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l7_b3/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l7_b3/testing_losses")


# Beta 1e-4
print("L7 1e-4")
model = BetaVAEMark7(latent_dim = 7) 
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/beta0_l7/weights/BetaVAE7_weights_epoch300.pth"))
beta = np.ones(100) * 1e-4
beta[:50] = np.linspace(0,49,50) * 1e-4/50
tools.train(project_name = "BetaVAEMark7", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l7_b4/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l7_b4/training_losses", beta = beta)

model = BetaVAEMark7(latent_dim = 7,log_clamping= False).to(device) 
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l7_b4/weights/BetaVAEMark7_weights_epoch100.pth"))
tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l7_b4/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l7_b4/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l7_b4/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l7_b4/testing_losses")


"""

# L32 tests
# Beta 1e-1
print("L32 1e-1")
model = BetaVAEMark7(latent_dim = 32)
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/beta0_l32/weights/BetaVAE7_weights_epoch300.pth"))
beta = np.ones(100) * 1e-1
beta[:50] = np.linspace(0,49,50) * 1e-1/50
tools.train(project_name = "BetaVAEMark7", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l32_b1/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l32_b1/training_losses", beta = beta)

model = BetaVAEMark7(latent_dim = 32,log_clamping= False).to(device) 
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l32_b1/weights/BetaVAEMark7_weights_epoch100.pth"))
tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l32_b1/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l32_b1/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l32_b1/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l32_b1/testing_losses")



# Beta 1e-2
print("L32 1e-2")
model = BetaVAEMark7(latent_dim = 32) 
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/beta0_l32/weights/BetaVAE7_weights_epoch300.pth"))
beta = np.ones(100) * 1e-2
beta[:50] = np.linspace(0,49,50) * 1e-2/50
tools.train(project_name = "BetaVAEMark7", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l32_b2/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l32_b2/training_losses", beta = beta)

model = BetaVAEMark7(latent_dim = 32,log_clamping= False).to(device) 
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l32_b2/weights/BetaVAEMark7_weights_epoch100.pth"))
tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l32_b2/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l32_b2/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l32_b2/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l32_b2/testing_losses")


# Beta 1e-3
print("L32 1e-3")
model = BetaVAEMark7(latent_dim = 32) 
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/beta0_l32/weights/BetaVAE7_weights_epoch300.pth"))

beta = np.ones(100) * 1e-3
beta[:50] = np.linspace(0,49,50) * 1e-3/50
tools.train(project_name = "BetaVAEMark7", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l32_b3/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l32_b3/training_losses", beta = beta)

model = BetaVAEMark7(latent_dim = 32,log_clamping= False).to(device)
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l32_b3/weights/BetaVAEMark7_weights_epoch100.pth"))
tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l32_b3/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l32_b3/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l32_b3/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l32_b3/testing_losses")


# Beta 1e-4
print("L32 1e-4")
model = BetaVAEMark7(latent_dim = 32) 
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/beta0_l32/weights/BetaVAE7_weights_epoch300.pth"))

beta = np.ones(100) * 1e-4
beta[:50] = np.linspace(0,49,50) * 1e-4/50
tools.train(project_name = "BetaVAEMark7", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l32_b4/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l32_b4/training_losses", beta = beta)

model = BetaVAEMark7(latent_dim = 32,log_clamping= False).to(device) 
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l32_b4/weights/BetaVAEMark7_weights_epoch100.pth"))
tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l32_b4/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l32_b4/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l32_b4/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l32_b4/testing_losses")



# L256 tests
# Beta 1e-1
print("L256 1e-1")
model = BetaVAEMark7(latent_dim = 256) 
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/beta0_l256/weights/BetaVAE7_weights_epoch300.pth"))
beta = np.ones(100) * 1e-1
beta[:50] = np.linspace(0,49,50) * 1e-1/50
tools.train(project_name = "BetaVAEMark7", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l256_b1/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l256_b1/training_losses", beta = beta)

model = BetaVAEMark7(latent_dim = 256,log_clamping= False).to(device) 
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l256_b1/weights/BetaVAEMark7_weights_epoch100.pth"))
tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l256_b1/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l256_b1/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l256_b1/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l256_b1/testing_losses")



# Beta 1e-2
print("L256 1e-2")
model = BetaVAEMark7(latent_dim = 256) 
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/beta0_l256/weights/BetaVAE7_weights_epoch300.pth"))
beta = np.ones(100) * 1e-2
beta[:50] = np.linspace(0,49,50) * 1e-2/50
tools.train(project_name = "BetaVAEMark7", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l256_b2/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l256_b2/training_losses", beta = beta)

model = BetaVAEMark7(latent_dim = 256,log_clamping= False).to(device) 
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l256_b2/weights/BetaVAEMark7_weights_epoch100.pth"))
tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l256_b2/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l256_b2/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l256_b2/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l256_b2/testing_losses")


# Beta 1e-3
print("L256 1e-3")
model = BetaVAEMark7(latent_dim = 256) 
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/beta0_l256/weights/BetaVAE7_weights_epoch300.pth"))
beta = np.ones(100) * 1e-3
beta[:50] = np.linspace(0,49,50) * 1e-3/50
tools.train(project_name = "BetaVAEMark7", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l256_b3/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l256_b3/training_losses", beta = beta)

model = BetaVAEMark7(latent_dim = 256,log_clamping= False).to(device) 
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l256_b3/weights/BetaVAEMark7_weights_epoch100.pth"))
tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l256_b3/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l256_b3/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l256_b3/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l256_b3/testing_losses")


# Beta 1e-4
print("L256 1e-4")
model = BetaVAEMark7(latent_dim = 256)
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/beta0_l256/weights/BetaVAE7_weights_epoch300.pth"))
beta = np.ones(100) * 1e-4
beta[:50] = np.linspace(0,49,50) * 1e-4/50
tools.train(project_name = "BetaVAEMark7", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l256_b4/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l256_b4/training_losses", beta = beta)

model = BetaVAEMark7(latent_dim = 256,log_clamping= False).to(device) 
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l256_b4/weights/BetaVAEMark7_weights_epoch100.pth"))
tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l256_b4/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l256_b4/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l256_b4/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark7/l256_b4/testing_losses")

print("Finished Tuning")