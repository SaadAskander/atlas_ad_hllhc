import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ad_tools.tools as tools

class BetaVAEMark8Encoder(nn.Module):
    """
    Beta VAE Mark 8 Encoder 
    """
    def __init__(self, latent_dim = 7, log_clamping = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 6, out_channels = 8, kernel_size = 3, padding = (1,0))
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3,padding = (1,0))
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3,padding = (1,0))
        self.pool1 = nn.MaxPool2d(2, return_indices = True)
        self.pool2 = nn.MaxPool2d((5,2), return_indices = True)
        self.pool3 = nn.MaxPool2d((5,2), return_indices = True)
        self.flatten = nn.Flatten(start_dim = 1)
        self.dense1 = nn.Linear(256, 64)
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)
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
        output14 = self.dense1(output13)
        output15 = F.leaky_relu(output14)
        mu = self.mu(output15)
        logvar = self.logvar(output15)
        if self.log_clamping:
            logvar = torch.clamp(logvar,-10,5)
        
        return mu, logvar,indices1,indices2,indices3

    
class BetaVAEMark8Decoder(nn.Module):
    """
    Beta VAE Mark 8  Decoder.
    """
    def __init__(self, latent_dim = 7):
        super().__init__()
        self.dense1 = nn.Linear(latent_dim, 64)
        self.dense2 = nn.Linear(64, 256)
    
        self.unpool1 = nn.MaxUnpool2d(kernel_size = (5,2))
        self.transconv1 = nn.ConvTranspose2d(in_channels = 32 , out_channels = 16 ,kernel_size = 3, padding = (1,2))
        


        self.unpool2 = nn.MaxUnpool2d(kernel_size = (5,2))
        self.transconv2 = nn.ConvTranspose2d(in_channels = 16 , out_channels = 8 ,kernel_size = 3, padding = (1,2))
        


        self.unpool3 = nn.MaxUnpool2d(kernel_size = 2)
        self.transconv3 = nn.ConvTranspose2d(in_channels = 8 , out_channels = 6 ,kernel_size = 3, padding = (1,2))        
        self.circular = nn.CircularPad2d((1,1,0,0))
        
    
    def forward(self, latent_vector,indices1, indices2, indices3):
        output1 = self.dense1(latent_vector)
        output1 = F.leaky_relu(output1)
        output1 = self.dense2(output1)
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
    
class BetaVAEMark8(nn.Module):
    """
    Beta VAE Mark 8.
    """

    def __init__(self, latent_dim = 7, log_clamping = True):
        super().__init__()
        self.encoder = BetaVAEMark8Encoder(latent_dim = latent_dim, log_clamping = log_clamping)
        self.decoder = BetaVAEMark8Decoder(latent_dim = latent_dim)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) 
        return mu + eps * std
    
    def forward(self, input):
        mu, logvar, indices1, indices2, indices3 = self.encoder(input)
        z = self.reparameterise(mu, logvar)
        output = self.decoder(z, indices1, indices2, indices3)
        return output, mu, logvar,z
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### 7D Latent Space ###
print("l7")
# Beta = 1e-1
print("Beta = 1e-1")
model = BetaVAEMark8(latent_dim = 7)
beta = np.ones(100) * 1e-1
beta[:50] = np.linspace(0,49,50) * 1e-1/50
tools.train(project_name = "BetaVAEMark8", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l7/b1e_1/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l7/b1e_1/training_losses", beta = beta)

model = BetaVAEMark8(latent_dim = 7,log_clamping = False).to(device) 
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l7/b1e_1/weights/BetaVAEMark8_weights_epoch100.pth"))
tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l7/b1e_1/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l7/b1e_1/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l7/b1e_1/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l7/b1e_1/testing_losses")

# Beta = 1e-2
print("Beta = 1e-2")
model = BetaVAEMark8(latent_dim = 7)
beta = np.ones(100) * 1e-2
beta[:50] = np.linspace(0,49,50) * 1e-2/50
tools.train(project_name = "BetaVAEMark8", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l7/b1e_2/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l7/b1e_2/training_losses", beta = beta)

model = BetaVAEMark8(latent_dim = 7,log_clamping = False).to(device) 
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l7/b1e_2/weights/BetaVAEMark8_weights_epoch100.pth"))
tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l7/b1e_2/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l7/b1e_2/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l7/b1e_2/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l7/b1e_2/testing_losses")

# Beta = 1e-3
print("Beta = 1e-3")
model = BetaVAEMark8(latent_dim = 7)
beta = np.ones(100) * 1e-3
beta[:50] = np.linspace(0,49,50) * 1e-3/50
tools.train(project_name = "BetaVAEMark8", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l7/b1e_3/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l7/b1e_3/training_losses", beta = beta)

model = BetaVAEMark8(latent_dim = 7,log_clamping = False).to(device) 
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l7/b1e_3/weights/BetaVAEMark8_weights_epoch100.pth"))
tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l7/b1e_3/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l7/b1e_3/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l7/b1e_3/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l7/b1e_3/testing_losses")


### 4D Latent Space ###
print("l4")
# Beta = 1e-1
print("Beta = 1e-1")
model = BetaVAEMark8(latent_dim = 4)
beta = np.ones(100) * 1e-1
beta[:50] = np.linspace(0,49,50) * 1e-1/50
tools.train(project_name = "BetaVAEMark8", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l4/b1e_1/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l4/b1e_1/training_losses", beta = beta)

model = BetaVAEMark8(latent_dim = 4,log_clamping = False).to(device) 
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l4/b1e_1/weights/BetaVAEMark8_weights_epoch100.pth"))
tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l4/b1e_1/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l4/b1e_1/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l4/b1e_1/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l4/b1e_1/testing_losses")

# Beta = 1e-2
print("Beta = 1e-2")
model = BetaVAEMark8(latent_dim = 4)
beta = np.ones(100) * 1e-2
beta[:50] = np.linspace(0,49,50) * 1e-2/50
tools.train(project_name = "BetaVAEMark8", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l4/b1e_2/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l4/b1e_2/training_losses", beta = beta)

model = BetaVAEMark8(latent_dim = 4,log_clamping = False).to(device) 
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l4/b1e_2/weights/BetaVAEMark8_weights_epoch100.pth"))
tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l4/b1e_2/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l4/b1e_2/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l4/b1e_2/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l4/b1e_2/testing_losses")

# Beta = 1e-3
print("Beta = 1e-3")
model = BetaVAEMark8(latent_dim = 4)
beta = np.ones(100) * 1e-3
beta[:50] = np.linspace(0,49,50) * 1e-3/50
tools.train(project_name = "BetaVAEMark8", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l4/b1e_3/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l4/b1e_3/training_losses", beta = beta)

model = BetaVAEMark8(latent_dim = 4,log_clamping = False).to(device) 
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l4/b1e_3/weights/BetaVAEMark8_weights_epoch100.pth"))
tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l4/b1e_3/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l4/b1e_3/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l4/b1e_3/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l4/b1e_3/testing_losses")


### 2D Latent Space ###
print("l2")
# Beta = 1e-1
print("Beta = 1e-1")
model = BetaVAEMark8(latent_dim = 2)
beta = np.ones(100) * 1e-1
beta[:50] = np.linspace(0,49,50) * 1e-1/50
tools.train(project_name = "BetaVAEMark8", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l2/b1e_1/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l2/b1e_1/training_losses", beta = beta)

model = BetaVAEMark8(latent_dim = 2,log_clamping = False).to(device) 
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l2/b1e_1/weights/BetaVAEMark8_weights_epoch100.pth"))
tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l2/b1e_1/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l2/b1e_1/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l2/b1e_1/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l2/b1e_1/testing_losses")

# Beta = 1e-2
print("Beta = 1e-2")
model = BetaVAEMark8(latent_dim = 2)
beta = np.ones(100) * 1e-2
beta[:50] = np.linspace(0,49,50) * 1e-2/50
tools.train(project_name = "BetaVAEMark8", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l2/b1e_2/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l2/b1e_2/training_losses", beta = beta)

model = BetaVAEMark8(latent_dim = 2,log_clamping = False).to(device) 
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l2/b1e_2/weights/BetaVAEMark8_weights_epoch100.pth"))
tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l2/b1e_2/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l2/b1e_2/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l2/b1e_2/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l2/b1e_2/testing_losses")

# Beta = 1e-3
print("Beta = 1e-3")
model = BetaVAEMark8(latent_dim = 2)
beta = np.ones(100) * 1e-3
beta[:50] = np.linspace(0,49,50) * 1e-3/50
tools.train(project_name = "BetaVAEMark8", model = model,
            weights_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l2/b1e_3/weights",
            training_losses_directory_path="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l2/b1e_3/training_losses", beta = beta)

model = BetaVAEMark8(latent_dim = 2,log_clamping = False).to(device) 
model.load_state_dict(torch.load("/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l2/b1e_3/weights/BetaVAEMark8_weights_epoch100.pth"))
tools.test(model = model,
           signal_acceptance_directory = "/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l2/b1e_3/signal_acceptance_rates",
           phi_invariance_study_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l2/b1e_3/phi_invariance_study",
           latent_code_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l2/b1e_3/latent_vectors",
           testing_losses_directory="/home/xzcapask/atlas_ad_hllhc/data/model_data/BetaVAEMark8/l2/b1e_3/testing_losses")