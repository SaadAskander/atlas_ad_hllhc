import torch 
import torch.nn as nn
from torch.utils.data import IterableDataset,TensorDataset,DataLoader
import awkward as ak
import pyarrow.parquet as pa
import torch 
import torch.nn.functional as F 
import numpy as np
import wandb # W & B Tracking
import matplotlib.pyplot as plt
import vector as vec
vec.register_awkward()



def ATLASMapStyleDatasetMaker(filepath,lower_row_index = 0, upper_row_index = 49, transform = None,
                               pileup_cutoff = 2, batch_size = 128, transform1 = False, transform2 = False,
                                transform2_5 = False, transform3 = False, transform4 = False, shuffle = True):
    """
    Creates a map style dataset of preprocessed ATLAS calorimeter images. 

    NB RMS scaling is relative to the RMS of JZ0 cell tower energy of the JZ0 dataset.
    
    NB! Row group 249 of JZ0 dataset has only 500 images. This dataset maker assumes the rows are completely filled.
    Therefore DON'T USE row 249 of JZ0 dataset.

    Pileup supression applied.



    Inputs:
    filepath: String containing path to parquet file with a cell_tower column.
    lower_row_index: inclusive lower bound on row indices to load images from. 0 by default.
    upper_row_index: inclusive upper bound on row indices to load images form. 49 by default. 
    pileup_cuttoff: pileup supression cutoff limit
    transform: Custom transform to add
    transform1: Combining just the first layer
    transform2: Combines the first 2 layers of the calorimeter images and the 5th and the 6th layers.
    transform2_5: Combining the first 2 layers and the last 3 layers.
    transform3: Combines the first 3 layers of the calorimeter images and the last 3 layers.
    transfrom4: Combines all layers of the calorimeter images.



    Outputs:
    images: Dataloader of a map-style dataset of the calorimeter images.
    """

    assert upper_row_index < 249, "The upper row group index must be an integer less than 249."

    # Loading dataset in batches due to unresolved loading issues in DIAS
    num_rows = upper_row_index + 1 - lower_row_index 
    num_images = 1000 * num_rows 
    images = torch.empty((num_images,50,64,6)) 

    for i, column_index in enumerate(range(lower_row_index, upper_row_index + 1)):
        image_batch = ak.from_parquet(filepath, row_groups = [column_index], columns = "cell_towers")
        image_batch = ak.to_torch(image_batch.cell_towers).to(torch.float32)
        images[i * 1000 : (i + 1) * 1000] = image_batch

    # Permute to NCHW format
    images = torch.permute(images, dims = (0,3,1,2))
        
    # Pileup supression
    summed_images = torch.sum(images, dim = 1, keepdim = True)
    images = torch.where(summed_images >= pileup_cutoff, images, 0)

    # RMS scaling 
    jz0_rms_energy = 0.0955 # GeV by calculation
    images = images / jz0_rms_energy   

    if transform:
        images = transform(images)
    
    # Layer Combining for sparsity study
    # ECal layers are only combined with ECal layers. Tile Cal layers are only combined with Tile Cal layers. 
    # Combining just the 3rd and 4th layers. Leaves 5 layers in total
    if transform1:
        layer3 = images[:,[2,3]].sum(axis = 1, keepdims = True)
        images = torch.concatenate([images[:,:2], layer3, images[:,4:] ], axis = 1)
    
    # Combining 3rd layer with the 4th. Combining 5th layer with the 6th. Leaves 4 layers in total.
    if transform2:
        layer3 = images[:,[2,3]].sum(axis = 1, keepdims = True)
        layer4 = images[:,[4,5]].sum(axis = 1, keepdims = True)
        images = torch.concatenate([images[:,:2], layer3, layer4], axis = 1)

    # Combining layer 1 and 2. Combining layer 3 and 4. Combining layer 5 and 6. Leaves 3 layers in total.
    if transform2_5:
        layer1 = images[:,[0,1]].sum(axis = 1, keepdims = True)
        layer2 = images[:,[2,3]].sum(axis = 1, keepdims = True)
        layer3 = images[:,[4,5]].sum(axis = 1, keepdims = True)
        images = torch.concatenate([layer1, layer2, layer3], axis = 1)    
    
    # Combining first 4 layers. Combining last 2 layers. Leaves 2 layers in total.
    if transform3: 
        layer1 = images[:,0:4].sum(axis = 1, keepdims = True)
        layer2 = images[:,4:].sum(axis = 1, keepdims = True)
        images = torch.concatenate([layer1, layer2], axis = 1)
    
    # Combining all layers.
    if transform4: 
        images = images.sum(axis = 1, keepdims = True)

    images = TensorDataset(images)

    images = DataLoader(images, batch_size = batch_size, shuffle = shuffle)
    
    return images



def training_loss(recon_image, input_image, mu, logvar, beta = 1, nf = False, kld = None):
    """
    Computes the total weighted training loss function for
    the VAE:
    
    (MSE + beta * KL Divergence).

    NB! NF IS FOR NORMALISING FLOWS
    """
    if nf:
        recon_loss = torch.mean(F.mse_loss(recon_image, input_image, reduction = "none"), dim = (1,2,3))
        mean_batch_recon_loss = torch.mean(recon_loss)

        kld = torch.clamp(kld, 0.5)
        mean_batch_kl_loss = torch.mean(kld)
        mean_batch_total_loss = torch.mean(recon_loss + beta * kld)
        return mean_batch_recon_loss, mean_batch_kl_loss, mean_batch_total_loss

    else:
        recon_loss = torch.mean(F.mse_loss(recon_image, input_image, reduction = "none"), dim = (1,2,3)) 
        kl_loss = - 0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim = -1)
        kl_loss = torch.clamp(kl_loss, 0.5)

        mean_batch_recon_loss = torch.mean(recon_loss)
        mean_batch_kl_loss = torch.mean(kl_loss)
        mean_batch_total_loss = torch.mean(recon_loss + beta * kl_loss)
        return mean_batch_recon_loss, mean_batch_kl_loss, mean_batch_total_loss



def testing_loss(recon_image, input_image, mu, logvar, nf = False, kld = None):
    """
    Computes the total testing loss function for
    the VAE:
    
    (MSE + KL Divergence).
    NB! NF IS FOR NORMALISING FLOWS
    """
    
    if nf:
        recon_loss = torch.mean(F.mse_loss(recon_image, input_image, reduction = "none"), dim = (1,2,3))
        mean_batch_recon_loss = torch.mean(recon_loss)
        
        kld = torch.clamp(kld, 0.5)
        mean_batch_kl_loss = torch.mean(kld)
        mean_batch_total_loss = torch.mean(recon_loss + kld)
        return mean_batch_recon_loss, mean_batch_kl_loss, mean_batch_total_loss
    else:
        recon_loss = torch.mean(F.mse_loss(recon_image, input_image, reduction = "none"), dim = (1,2,3)) 
        kl_loss = - 0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim = -1)

        mean_batch_recon_loss = torch.mean(recon_loss)
        mean_batch_kl_loss = torch.mean(kl_loss)
        mean_batch_total_loss = torch.mean(recon_loss + kl_loss)
        return mean_batch_recon_loss, mean_batch_kl_loss, mean_batch_total_loss
    







def epoch_train(model, dataloader, device, optimiser, num_events = 150000, batch_size = 128,beta = 1, nf = False):
    """
    Trains VAE for an epoch.

    Assuming 150K events are used by default. Change if this isn't true.

    Inputs:
    nf: Set nf = True if using the normalising flow VAE.

    """

    
    if nf:
        recon_loss = 0
        kl_loss = 0
        total_loss = 0

        model.train() 
        for images in dataloader:
            optimiser.zero_grad() 
            images = images[0].to(device)
            
            # Forward pass
            recon_image,mu,logvar,kld,z = model(images) 
            batch_mean_recon_loss,batch_mean_kl_loss,batch_mean_total_loss = training_loss(recon_image, images, 
                                                                                        mu, logvar,beta = beta,
                                                                                        nf = True,
                                                                                        kld  = kld)

            # Backward pass and weight update
            batch_mean_total_loss.backward() 

            
            
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)
            
            optimiser.step() 

            # Cumulative error until current batch
            recon_loss += batch_mean_recon_loss.item()
            kl_loss += batch_mean_kl_loss.item()
            total_loss += batch_mean_total_loss.item() 

            


        # Average loss of epoch
        num_batches = np.ceil(num_events/batch_size)
        avg_recon_loss = recon_loss / num_batches
        avg_kl_loss = kl_loss / num_batches 
        avg_total_loss = total_loss / num_batches

        return avg_recon_loss, avg_kl_loss, avg_total_loss, total_norm.item()

    else:
        recon_loss = 0
        kl_loss = 0
        total_loss = 0

        model.train() 
        for images in dataloader:
            optimiser.zero_grad() 
            images = images[0].to(device)
            
            # Forward pass
            recon_image, mu, logvar,z = model(images) 
            batch_mean_recon_loss,batch_mean_kl_loss,batch_mean_total_loss = training_loss(recon_image, images, 
                                                                                        mu, logvar,beta = beta)

            # Backward pass and weight update
            batch_mean_total_loss.backward() 

            
            
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)
            
            optimiser.step() 

            # Cumulative error until current batch
            recon_loss += batch_mean_recon_loss.item()
            kl_loss += batch_mean_kl_loss.item()
            total_loss += batch_mean_total_loss.item() 

            


        # Average loss of epoch
        num_batches = np.ceil(num_events/batch_size)
        avg_recon_loss = recon_loss / num_batches
        avg_kl_loss = kl_loss / num_batches 
        avg_total_loss = total_loss / num_batches

        return avg_recon_loss, avg_kl_loss, avg_total_loss, total_norm.item()





def epoch_valid(model, dataloader, device, num_events = 50000, batch_size = 128, nf = False):
    """
    Validation loop for an epoch.

    Assuming 50K events used for validation by default. 
    """
    if nf:
        model.eval()
        recon_loss = 0
        kl_loss = 0
        total_loss = 0
        with torch.no_grad(): 
            for images in dataloader:
                images = images[0].to(device)
                recon_images, mu, logvar,kld,z = model(images)
                batch_mean_recon_loss, batch_mean_kl_loss, batch_mean_total_loss = testing_loss(recon_image=recon_images, 
                                                                                                input_image=images, mu = mu, 
                                                                                                logvar = logvar,nf = nf, kld = kld)        
                recon_loss += batch_mean_recon_loss.item()
                kl_loss += batch_mean_kl_loss.item()
                total_loss += batch_mean_total_loss.item()
        
        num_batches = np.ceil(num_events/batch_size)
        avg_recon_loss = recon_loss / num_batches
        avg_kl_loss = kl_loss / num_batches
        avg_total_loss = total_loss / num_batches 
        return avg_recon_loss, avg_kl_loss, avg_total_loss 
    else: 
        model.eval()
        recon_loss = 0
        kl_loss = 0
        total_loss = 0
        with torch.no_grad(): 
            for images in dataloader:
                images = images[0].to(device)
                recon_images, mu, logvar,z = model(images)
                batch_mean_recon_loss, batch_mean_kl_loss, batch_mean_total_loss = testing_loss(recon_image=recon_images, 
                                                                                                input_image=images, mu = mu, 
                                                                                                logvar = logvar)        
                recon_loss += batch_mean_recon_loss.item()
                kl_loss += batch_mean_kl_loss.item()
                total_loss += batch_mean_total_loss.item()
        
        num_batches = np.ceil(num_events/batch_size)
        avg_recon_loss = recon_loss / num_batches
        avg_kl_loss = kl_loss / num_batches
        avg_total_loss = total_loss / num_batches 
        return avg_recon_loss, avg_kl_loss, avg_total_loss




def train(project_name, model,beta, weights_directory_path,
          training_losses_directory_path, epochs = 100, lr = 1e-4, pileup_cutoff = 2,
          transform1 = False, transform2_5 = False, transform2 = False, transform3 = False, transform4 = False,nf = False):
    
    """
    Trains a VAE for specified number of epochs
    beta should be an array of beta values of size epochs
    NB! NF IS FOR NORMALISING FLOWS
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_training_events = 150000
    num_validation_events = 50000
    batch_size = 128

    training_data_loader = ATLASMapStyleDatasetMaker(filepath 
                = "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/JZ0_no_filter.parquet",
                upper_row_index = 149,
                batch_size = batch_size,
                pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3,
                transform4 = transform4, transform1 = transform1, transform2_5 = transform2_5)
    validation_data_loader = ATLASMapStyleDatasetMaker(filepath = 
                            "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/JZ0_no_filter.parquet",
                            lower_row_index = 150,
                              upper_row_index = 199,
                                batch_size = batch_size,
                                pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3,
                                 transform4 = transform4,
                                 transform1 = transform1, transform2_5 = transform2_5)
    
    
    wandb.login()
    config = {"epochs": epochs, "lr":lr}
    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr = lr) 
    
    # Storing losses from each epoch
    valid_recon_losses = []
    valid_kl_losses = []
    valid_total_losses = []
    train_recon_losses = []
    train_kl_losses = []
    train_total_losses = []

    with wandb.init(project = project_name, config = config) as run:
     
        for i in range(epochs):
            print(f"Starting Epoch {i + 1}")

            # Training loop =
            (train_avg_recon_loss,train_avg_kl_loss, train_avg_total_loss,
             total_norm
            ) = epoch_train(model = model,
                        dataloader = training_data_loader, 
                        device = device, 
                        optimiser = optimiser, 
                        beta = beta[i],
                        num_events = num_training_events, nf=nf)
            train_recon_losses.append(train_avg_recon_loss)
            train_kl_losses.append(train_avg_kl_loss)
            train_total_losses.append(train_avg_total_loss)

            # Validation loop
            valid_avg_recon_loss,valid_avg_kl_loss, valid_avg_total_loss = epoch_valid(model = model,
                                                                                       dataloader = validation_data_loader,
                                                                                       device = device,
                                                                                       num_events = num_validation_events,nf=nf)
            valid_recon_losses.append(valid_avg_recon_loss)
            valid_kl_losses.append(valid_avg_kl_loss)
            valid_total_losses.append(valid_avg_total_loss)



            run.log({"Training Total Loss": train_avg_total_loss,"Training Recon Loss": train_avg_recon_loss,
                      "Training KL Loss": train_avg_kl_loss, "Validation Total Loss": valid_avg_total_loss,
                      "Validation Recon Loss": valid_avg_recon_loss,
                      "Validation KL Loss": valid_avg_kl_loss ,"total_norm": total_norm})
            
            torch.save(model.state_dict(), f'{weights_directory_path}/{project_name}_weights_epoch{i + 1}.pth')
            print(f"Epoch {i+1} Complete")
            
        
        print("Training Complete")
        
        # Saving training and validation losses
        np.save(file = f"{training_losses_directory_path}/validation_total_losses.npy", arr = valid_total_losses)
        np.save(file = f"{training_losses_directory_path}/training_total_losses.npy", arr = train_total_losses)
        np.save(file = f"{training_losses_directory_path}/validation_recon_losses.npy", arr = valid_recon_losses)
        np.save(file = f"{training_losses_directory_path}/training_recon_losses.npy", arr = train_recon_losses)
        np.save(file = f"{training_losses_directory_path}/validation_kl_losses.npy", arr = valid_kl_losses)
        np.save(file = f"{training_losses_directory_path}/training_kl_losses.npy", arr = train_kl_losses)





def evaluate(model,dataloader, nf = False):
    """
    Passes dataset through the VAE and evaluates error and latent codes for each image.
    """
    if nf:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        recon_losses = []
        kl_losses = []
        total_losses = []
        latent_vectors = []

        with torch.no_grad():
            for images in dataloader:
                images = images[0].to(device)
                recon_image, mu, logvar,kld,z = model(images)
                recon_loss,kl_loss,total_loss = testing_loss(recon_image, images, mu, logvar, nf = nf, kld = kld)
                recon_losses.append(recon_loss)
                kl_losses.append(kl_loss)
                total_losses.append(total_loss)
                latent_vectors.append(z.cpu().numpy())

            
        return (torch.tensor(recon_losses).cpu(), torch.tensor(kl_losses).cpu(),
                torch.tensor(total_losses).cpu(), latent_vectors)
    else: 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        recon_losses = []
        kl_losses = []
        total_losses = []
        latent_vectors = []

        with torch.no_grad():
            for images in dataloader:
                images = images[0].to(device)
                recon_image, mu, logvar,z = model(images)
                recon_loss,kl_loss,total_loss = testing_loss(recon_image, images, mu, logvar)
                recon_losses.append(recon_loss)
                kl_losses.append(kl_loss)
                total_losses.append(total_loss)
                latent_vectors.append(z.cpu().numpy())

            
        return (torch.tensor(recon_losses).cpu(), torch.tensor(kl_losses).cpu(),
                torch.tensor(total_losses).cpu(), latent_vectors)




def anomaly_cutoff_score(jz0_test_loss):
    """
    Takes the reconstruction errors of the jz0 and signal events and computes the cut off anomaly score and 
    signal_acceptance_rates.
    """

    cutoff_score = np.percentile(jz0_test_loss, 97.5)
    return cutoff_score







def signal_acceptance_rate(signal_test_loss, cutoff_score):
    signal_rate = (sum(signal_test_loss > cutoff_score) / len(signal_test_loss)) * 100
    return signal_rate







def test(model,signal_acceptance_directory,phi_invariance_study_directory, latent_code_directory, 
         testing_losses_directory, pileup_cutoff = 2,  transform1 = False, transform2_5 = False, transform2 = False, 
         transform3 = False, transform4 = False, shuffle = False, return_kl_scores = False,nf = False):
    """
    Tests the performance of the VAE.

    Inputs: 
        return_kl_scores: return kl signal acceptance scores
    """
    if nf:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Loading Datasets
        jz0_test_loader = ATLASMapStyleDatasetMaker(filepath =
                    "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/JZ0_no_filter.parquet",
                    lower_row_index = 200, upper_row_index = 248,
                    # See docs for explaination of why upper_row_index = 248 
                    batch_size = 1, shuffle = shuffle,
                    pileup_cutoff = pileup_cutoff,transform1 = transform1, transform2_5=transform2_5,
                    transform2 = transform2, transform3 = transform3, transform4 = transform4) 
        
        ggF_test_loader = ATLASMapStyleDatasetMaker(filepath =
                    "//home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/ggF_SM_HH4b.parquet",
                    upper_row_index= 49,
                    batch_size = 1,shuffle = shuffle,transform1 = transform1, transform2_5=transform2_5,
                    pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3, transform4 = transform4)
        
        vbf_test_loader = ATLASMapStyleDatasetMaker(filepath = 
                    "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/VBF_SM_HH4b.parquet",
                    upper_row_index= 49,
                    batch_size = 1,shuffle = shuffle,transform1 = transform1, transform2_5=transform2_5,
                    pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3, transform4 = transform4)
        
        hs_test_loader = ATLASMapStyleDatasetMaker(filepath =
                    "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/HZ_bbvv.parquet",
                    upper_row_index= 49,
                    batch_size = 1,shuffle = shuffle,transform1 = transform1, transform2_5=transform2_5,
                    pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3, transform4 = transform4)
        
        transform = lambda x: torch.roll(x, shifts = 16, dims = 3)
        vbf_test_pi4_dataset = ATLASMapStyleDatasetMaker(filepath =
                    "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/VBF_SM_HH4b.parquet",
                    lower_row_index = 0,
                    upper_row_index= 40,
                    transform = transform, batch_size = 1,shuffle = shuffle,transform1 = transform1, transform2_5=transform2_5,
                    pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3, transform4 = transform4)
        
        transform = lambda x: torch.roll(x, shifts = 32, dims = 3)
        vbf_test_pi2_dataset = ATLASMapStyleDatasetMaker(filepath =
                    "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/VBF_SM_HH4b.parquet",
                    lower_row_index = 0,
                    upper_row_index= 40,
                    transform = transform, batch_size=1,shuffle = shuffle,transform1 = transform1, transform2_5=transform2_5,
                    pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3, transform4 = transform4)
        
        transform = lambda x: torch.roll(x, shifts = 48, dims = 3)
        vbf_test_3pi4_dataset = ATLASMapStyleDatasetMaker(filepath =
                    "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/VBF_SM_HH4b.parquet",
                    lower_row_index = 0,
                    upper_row_index= 40,
                    transform = transform, batch_size=1,shuffle = shuffle,transform1 = transform1, transform2_5=transform2_5,
                    pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3, transform4 = transform4)
        
        
        transform = lambda x: torch.roll(x, shifts = 16, dims = 3)
        ggf_test_pi4_dataset = ATLASMapStyleDatasetMaker(filepath =
                    "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/ggF_SM_HH4b.parquet",
                    lower_row_index = 0,
                    upper_row_index= 40,
                    transform = transform, batch_size = 1,shuffle = shuffle,transform1 = transform1, transform2_5=transform2_5,
                    pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3, transform4 = transform4)
        
        transform = lambda x: torch.roll(x, shifts = 32, dims = 3)
        ggf_test_pi2_dataset = ATLASMapStyleDatasetMaker(filepath =
                    "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/ggF_SM_HH4b.parquet",
                    lower_row_index = 0,
                    upper_row_index= 40,
                    transform = transform, batch_size=1,shuffle = shuffle,transform1 = transform1, transform2_5=transform2_5,
                    pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3, transform4 = transform4)
        
        transform = lambda x: torch.roll(x, shifts = 48, dims = 3)
        ggf_test_3pi4_dataset = ATLASMapStyleDatasetMaker(filepath =
                    "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/ggF_SM_HH4b.parquet",
                    lower_row_index = 0,
                    upper_row_index= 40,
                    transform = transform, batch_size=1,shuffle = shuffle,transform1 = transform1, transform2_5=transform2_5,
                    pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3, transform4 = transform4)

        transform = lambda x: torch.roll(x, shifts = 16, dims = 3)
        hs_test_pi4_dataset = ATLASMapStyleDatasetMaker(filepath =
                    "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/HZ_bbvv.parquet",
                    lower_row_index = 0,
                    upper_row_index= 40,
                    transform = transform, batch_size = 1,shuffle = shuffle,transform1 = transform1, transform2_5=transform2_5,
                    pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3, transform4 = transform4)
        
        transform = lambda x: torch.roll(x, shifts = 32, dims = 3)
        hs_test_pi2_dataset = ATLASMapStyleDatasetMaker(filepath =
                    "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/HZ_bbvv.parquet",
                    lower_row_index = 0,
                    upper_row_index= 40,
                    transform = transform, batch_size=1,shuffle = shuffle,transform1 = transform1, transform2_5=transform2_5,
                    pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3, transform4 = transform4)
        
        transform = lambda x: torch.roll(x, shifts = 48, dims = 3)
        hs_test_3pi4_dataset = ATLASMapStyleDatasetMaker(filepath =
                    "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/HZ_bbvv.parquet",
                    lower_row_index = 0,
                    upper_row_index= 40,
                    transform = transform, batch_size=1,shuffle = shuffle,transform1 = transform1, transform2_5=transform2_5,
                    pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3, transform4 = transform4)

        transform = lambda x: torch.roll(x, shifts = 16, dims = 3)
        jz0_test_pi4_dataset = ATLASMapStyleDatasetMaker(filepath =
                    "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/JZ0_no_filter.parquet",
                    lower_row_index = 200,
                    upper_row_index= 248,
                    transform = transform, batch_size = 1,shuffle = shuffle,transform1 = transform1, transform2_5=transform2_5,
                    pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3, transform4 = transform4)
        
        transform = lambda x: torch.roll(x, shifts = 32, dims = 3)
        jz0_test_pi2_dataset = ATLASMapStyleDatasetMaker(filepath =
                    "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/JZ0_no_filter.parquet",
                    lower_row_index = 200,
                    upper_row_index= 248,
                    transform = transform, batch_size=1,shuffle = shuffle,transform1 = transform1, transform2_5=transform2_5,
                    pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3, transform4 = transform4)
        
        transform = lambda x: torch.roll(x, shifts = 48, dims = 3)
        jz0_test_3pi4_dataset = ATLASMapStyleDatasetMaker(filepath =
                    "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/JZ0_no_filter.parquet",
                    lower_row_index = 200,
                    upper_row_index= 248,
                    transform = transform, batch_size=1,shuffle = shuffle,transform1 = transform1, transform2_5=transform2_5,
                    pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3, transform4 = transform4)
        
        ggf_info = ak.from_parquet("//home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/ggF_SM_HH4b.parquet", columns = "AntiKt4TruthDressedWZJets", row_groups = range(50))
        vbf_info = ak.from_parquet("/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/VBF_SM_HH4b.parquet", columns = "AntiKt4TruthDressedWZJets", row_groups = range(50))
        hs_info = ak.from_parquet("/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/HZ_bbvv.parquet", columns = "AntiKt4TruthDressedWZJets", row_groups = range(50))

        # Determining anomaly cutoff scores
        jz0_test_mse, jz0_test_kl, jz0_test_total_losses,jz0_latent_codes = evaluate(nf = nf,model = model,
                                                                                    dataloader = jz0_test_loader)
        mse_cutoff= anomaly_cutoff_score(jz0_test_mse)
        kl_cutoff= anomaly_cutoff_score(jz0_test_kl)
        total_cutoff= anomaly_cutoff_score(jz0_test_total_losses)

        # Signal acceptance rates
        ggF_test_mse, ggF_test_kl, ggF_test_total,ggF_latent_codes = evaluate(nf = nf,model = model,
                                                                                    dataloader = ggF_test_loader)
        mse_ggf_signal = signal_acceptance_rate(ggF_test_mse, mse_cutoff)
        kl_ggf_signal = signal_acceptance_rate(ggF_test_kl, kl_cutoff)
        total_ggf_signal = signal_acceptance_rate(ggF_test_total, total_cutoff)

        vbf_test_mse, vbf_test_kl, vbf_test_total, vbf_latent_codes = evaluate(nf = nf,model = model,
                                                                                    dataloader = vbf_test_loader)
        mse_vbf_signal = signal_acceptance_rate(vbf_test_mse, mse_cutoff)
        kl_vbf_signal = signal_acceptance_rate(vbf_test_kl, kl_cutoff)
        total_vbf_signal = signal_acceptance_rate(vbf_test_total, total_cutoff)

        hs_test_mse, hs_test_kl, hs_test_total, hs_latent_codes = evaluate(nf = nf,model = model,
                                                                                dataloader = hs_test_loader)
        mse_hs_signal = signal_acceptance_rate(hs_test_mse, mse_cutoff)
        kl_hs_signal = signal_acceptance_rate(hs_test_kl, kl_cutoff)
        total_hs_signal = signal_acceptance_rate(hs_test_total, total_cutoff)

        print(f"The mse ggf singal acceptance rate is {mse_ggf_signal}")
        print(f"The kl ggf singal acceptance rate is {kl_ggf_signal}")
        print(f"The recon ggf singal acceptance rate is {total_ggf_signal}")

        print(f"The mse vbf singal acceptance rate is {mse_vbf_signal}")
        print(f"The kl vbf singal acceptance rate is {kl_vbf_signal}")
        print(f"The recon vbf singal acceptance rate is {total_vbf_signal}")

        print(f"The mse hs singal acceptance rate is {mse_hs_signal}")
        print(f"The kl hs singal acceptance rate is {kl_hs_signal}")
        print(f"The recon hs singal acceptance rate is {total_hs_signal}")

        # Missing Transverse Energy (MTE) and Scalar Summed Transverse Energy (HT) Analysis
        
        # MTE
        mte_ggf = np.sum(ggf_info.AntiKt4TruthDressedWZJets[:], axis = 1).rho
        mte_vbf = np.sum(vbf_info.AntiKt4TruthDressedWZJets[:], axis = 1).rho
        mte_hs = np.sum(hs_info.AntiKt4TruthDressedWZJets[:], axis = 1).rho

        # HT
        ht_ggf = np.sum(ggf_info.AntiKt4TruthDressedWZJets.rho, axis = 1)
        ht_vbf = np.sum(vbf_info.AntiKt4TruthDressedWZJets.rho, axis = 1)
        ht_hs = np.sum(hs_info.AntiKt4TruthDressedWZJets.rho, axis = 1)

        # MTE/HT
        mte_ht_ggf = mte_ggf / ht_ggf
        mte_ht_vbf = mte_vbf / ht_vbf
        mte_ht_hs = mte_hs / ht_hs

        # MTE of registered signals
        mte_of_registered_ggf_signals = mte_ggf[ggF_test_kl.detach().cpu().numpy() >= kl_cutoff]
        mte_of_registered_vbf_signals = mte_vbf[vbf_test_kl.detach().cpu().numpy() >= kl_cutoff]
        mte_of_registered_hs_signals = mte_hs[hs_test_kl.detach().cpu().numpy() >= kl_cutoff]

        # MTE of unregistered signals
        mte_of_unregistered_ggf_signals = mte_ggf[ggF_test_kl.detach().cpu().numpy() < kl_cutoff]
        mte_of_unregistered_vbf_signals = mte_vbf[vbf_test_kl.detach().cpu().numpy() < kl_cutoff]
        mte_of_unregistered_hs_signals = mte_hs[hs_test_kl.detach().cpu().numpy() < kl_cutoff]

        # ht of registered signals
        ht_of_registered_ggf_signals = ht_ggf[ggF_test_kl.detach().cpu().numpy() >= kl_cutoff]
        ht_of_registered_vbf_signals = ht_vbf[vbf_test_kl.detach().cpu().numpy() >= kl_cutoff]
        ht_of_registered_hs_signals = ht_hs[hs_test_kl.detach().cpu().numpy() >= kl_cutoff]

        # ht of unregistered signals
        ht_of_unregistered_ggf_signals = ht_ggf[ggF_test_kl.detach().cpu().numpy() < kl_cutoff]
        ht_of_unregistered_vbf_signals = ht_vbf[vbf_test_kl.detach().cpu().numpy() < kl_cutoff]
        ht_of_unregistered_hs_signals = ht_hs[hs_test_kl.detach().cpu().numpy() < kl_cutoff]

        # mte_ht of registered signals
        mte_ht_of_registered_ggf_signals = mte_ht_ggf[ggF_test_kl.detach().cpu().numpy() >= kl_cutoff]
        mte_ht_of_registered_vbf_signals = mte_ht_vbf[vbf_test_kl.detach().cpu().numpy() >= kl_cutoff]
        mte_ht_of_registered_hs_signals = mte_ht_hs[hs_test_kl.detach().cpu().numpy() >= kl_cutoff]

        # mte_ht of unregistered signals
        mte_ht_of_unregistered_ggf_signals = mte_ht_ggf[ggF_test_kl.detach().cpu().numpy() < kl_cutoff]
        mte_ht_of_unregistered_vbf_signals = mte_ht_vbf[vbf_test_kl.detach().cpu().numpy() < kl_cutoff]
        mte_ht_of_unregistered_hs_signals = mte_ht_hs[hs_test_kl.detach().cpu().numpy() < kl_cutoff]
        
        # Test pileup


        # Storing MTE, HT and MTE/HT 
        np.save(file = f"{signal_acceptance_directory}/mte_of_registered_ggf_signals", arr = mte_of_registered_ggf_signals)
        np.save(file = f"{signal_acceptance_directory}/mte_of_registered_vbf_signals", arr = mte_of_registered_vbf_signals)
        np.save(file = f"{signal_acceptance_directory}/mte_of_registered_hs_signals", arr = mte_of_registered_hs_signals)

        np.save(file = f"{signal_acceptance_directory}/mte_of_unregistered_ggf_signals", arr = mte_of_unregistered_ggf_signals)
        np.save(file = f"{signal_acceptance_directory}/mte_of_unregistered_vbf_signals", arr = mte_of_unregistered_vbf_signals)
        np.save(file = f"{signal_acceptance_directory}/mte_of_unregistered_hs_signals", arr = mte_of_unregistered_hs_signals)

        np.save(file = f"{signal_acceptance_directory}/ht_of_registered_ggf_signals", arr = ht_of_registered_ggf_signals)
        np.save(file = f"{signal_acceptance_directory}/ht_of_registered_vbf_signals", arr = ht_of_registered_vbf_signals)
        np.save(file = f"{signal_acceptance_directory}/ht_of_registered_hs_signals", arr = ht_of_registered_hs_signals)

        np.save(file = f"{signal_acceptance_directory}/ht_of_unregistered_ggf_signals", arr = ht_of_unregistered_ggf_signals)
        np.save(file = f"{signal_acceptance_directory}/ht_of_unregistered_vbf_signals", arr = ht_of_unregistered_vbf_signals)
        np.save(file = f"{signal_acceptance_directory}/ht_of_unregistered_hs_signals", arr = ht_of_unregistered_hs_signals)

        np.save(file = f"{signal_acceptance_directory}/mte_ht_of_registered_ggf_signals", arr = mte_ht_of_registered_ggf_signals)
        np.save(file = f"{signal_acceptance_directory}/mte_ht_of_registered_vbf_signals", arr = mte_ht_of_registered_vbf_signals)
        np.save(file = f"{signal_acceptance_directory}/mte_ht_of_registered_hs_signals", arr = mte_ht_of_registered_hs_signals)

        np.save(file = f"{signal_acceptance_directory}/mte_ht_of_unregistered_ggf_signals", arr = mte_ht_of_unregistered_ggf_signals)
        np.save(file = f"{signal_acceptance_directory}/mte_ht_of_unregistered_vbf_signals", arr = mte_ht_of_unregistered_vbf_signals)
        np.save(file = f"{signal_acceptance_directory}/mte_ht_of_unregistered_hs_signals", arr = mte_ht_of_unregistered_hs_signals)

        
        # Storing cutoff scores
        np.save(file = f"{signal_acceptance_directory}/MSECutoff.npy", arr = mse_cutoff)
        np.save(file = f"{signal_acceptance_directory}/KLCutoff.npy", arr = kl_cutoff)
        np.save(file = f"{signal_acceptance_directory}/TotalCutoff.npy", arr = total_cutoff)
        #np.save(file = f"{latent_code_directory}/jz0_latent_codes.npy", arr = jz0_latent_codes)
        np.save(file = f"{testing_losses_directory}/jz0_mse_losses.npy", arr = jz0_test_mse)
        np.save(file = f"{testing_losses_directory}/jz0_kl_losses.npy", arr = jz0_test_kl)
        np.save(file = f"{testing_losses_directory}/jz0_total_losses.npy", arr=jz0_test_total_losses)

        

        # Storing losses
        np.save(file = f"{signal_acceptance_directory}/MSEggFsignalrate.npy", arr = mse_ggf_signal)
        np.save(file = f"{signal_acceptance_directory}/KLggFsignalrate.npy", arr = kl_ggf_signal)
        np.save(file = f"{signal_acceptance_directory}/TotalggFsignalrate.npy", arr = total_ggf_signal)
        #np.save(file = f"{latent_code_directory}/ggF_latent_codes.npy", arr = ggF_latent_codes)
        np.save(file = f"{testing_losses_directory}/ggF_mse_losses.npy", arr = ggF_test_mse)
        np.save(file = f"{testing_losses_directory}/ggF_kl_losses.npy", arr = ggF_test_kl)
        np.save(file = f"{testing_losses_directory}/ggF_total_losses.npy", arr=ggF_test_total)

        np.save(file = f"{signal_acceptance_directory}/MSEvbfsignalrate.npy", arr = mse_vbf_signal)
        np.save(file = f"{signal_acceptance_directory}/KLvbfsignalrate.npy", arr = kl_vbf_signal)
        np.save(file = f"{signal_acceptance_directory}/Totalvbfsignalrate.npy", arr = total_vbf_signal)
        #np.save(file = f"{latent_code_directory}/vbf_latent_codes.npy", arr = vbf_latent_codes)
        np.save(file = f"{testing_losses_directory}/vbf_mse_losses.npy", arr = vbf_test_mse)
        np.save(file = f"{testing_losses_directory}/vbf_kl_losses.npy", arr = vbf_test_kl)
        np.save(file = f"{testing_losses_directory}/vbf_total_losses.npy", arr=vbf_test_total)


        np.save(file = f"{signal_acceptance_directory}/MSEhssignalrate.npy", arr = mse_hs_signal)
        np.save(file = f"{signal_acceptance_directory}/KLhssignalrate.npy", arr = kl_hs_signal)
        np.save(file = f"{signal_acceptance_directory}/Totalhssignalrate.npy", arr = total_hs_signal)
        #np.save(file = f"{latent_code_directory}/hs_latent_codes.npy", arr = hs_latent_codes)
        np.save(file = f"{testing_losses_directory}/hs_mse_losses.npy", arr = hs_test_mse)
        np.save(file = f"{testing_losses_directory}/hs_kl_losses.npy", arr = hs_test_kl)
        np.save(file = f"{testing_losses_directory}/hs_total_losses.npy", arr=hs_test_total)

        # Reconstructions
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))

        for batch_idx, features in enumerate(jz0_test_loader):
            if batch_idx >= 3: break  # Stop after 3 images
        
            input = features[0].to(device)
            recon_image, mu, logvar,kld,z = model(input)
            input = input.cpu().detach().numpy()
            recon_image = recon_image.cpu().detach().numpy()

            im1 = axes[0,batch_idx].imshow(input[0,0])
            fig.colorbar(mappable = im1,ax = axes[0,batch_idx])
            axes[0,batch_idx].set_title(f"JZ0 Image {batch_idx + 1}")
        
    

        
            im2 = axes[1,batch_idx].imshow(recon_image[0,0])
            fig.colorbar(mappable = im2,ax = axes[1,batch_idx])
            axes[1,batch_idx].set_title(f"Reconstructed JZ0 Image {batch_idx + 1}")


        fig.savefig(f"{testing_losses_directory}/JZ0_Reconstructions.png")

        # phi invariance study
        jz0_pi4_mse,jz0_pi4_kl,jz0_test_pi4_total_losses,jz0_test_pi4_latent_codes= evaluate(nf = nf,model = model,
                                            dataloader = jz0_test_pi4_dataset)
        jz0_pi2_mse,jz0_pi2_kl,jz0_test_pi2_total_losses,jz0_test_pi2_latent_codes = evaluate(nf = nf,model = model,
                                            dataloader = jz0_test_pi2_dataset)
        jz0_3pi4_mse,jz0_3pi4_kl,jz0_test_3pi4_total_losses,jz0_test_3pi4_latent_codes = evaluate(nf = nf,model = model,
                                            dataloader = jz0_test_3pi4_dataset)

        np.save(file = f"{phi_invariance_study_directory}/jz0pi4totallosses.npy", arr = jz0_test_pi4_total_losses)
        np.save(file = f"{phi_invariance_study_directory}/jz0pi2totallosses.npy", arr = jz0_test_pi2_total_losses)
        np.save(file = f"{phi_invariance_study_directory}/jz03pi4totallosses.npy", arr = jz0_test_3pi4_total_losses)

        np.save(file = f"{phi_invariance_study_directory}/jz0pi4mse.npy", arr = jz0_pi4_mse)
        np.save(file = f"{phi_invariance_study_directory}/jz0pi2mse.npy", arr = jz0_pi2_mse)
        np.save(file = f"{phi_invariance_study_directory}/jz03pi4mse.npy", arr = jz0_3pi4_mse)

        np.save(file = f"{phi_invariance_study_directory}/jz0pi4kl.npy", arr = jz0_pi4_kl)
        np.save(file = f"{phi_invariance_study_directory}/jz0pi2kl.npy", arr = jz0_pi2_kl)
        np.save(file = f"{phi_invariance_study_directory}/jz03pi4kl.npy", arr = jz0_3pi4_kl)

        # Signal Phi Invariance

        vbf_pi4_mse,vbf_pi4_kl,vbf_test_pi4_total_losses,vbf_test_pi4_latent_codes= evaluate(nf = nf,model = model,
                                            dataloader = vbf_test_pi4_dataset)
        vbf_pi2_mse,vbf_pi2_kl,vbf_test_pi2_total_losses,vbf_test_pi2_latent_codes = evaluate(nf = nf,model = model,
                                            dataloader = vbf_test_pi2_dataset)
        vbf_3pi4_mse,vbf_3pi4_kl,vbf_test_3pi4_total_losses,vbf_test_3pi4_latent_codes = evaluate(nf = nf,model = model,
                                            dataloader = vbf_test_3pi4_dataset)

        np.save(file = f"{phi_invariance_study_directory}/vbfpi4totallosses.npy", arr = vbf_test_pi4_total_losses)
        np.save(file = f"{phi_invariance_study_directory}/vbfpi2totallosses.npy", arr = vbf_test_pi2_total_losses)
        np.save(file = f"{phi_invariance_study_directory}/vbf3pi4totallosses.npy", arr = vbf_test_3pi4_total_losses)

        np.save(file = f"{phi_invariance_study_directory}/vbfpi4mse.npy", arr = vbf_pi4_mse)
        np.save(file = f"{phi_invariance_study_directory}/vbfpi2mse.npy", arr = vbf_pi2_mse)
        np.save(file = f"{phi_invariance_study_directory}/vbf3pi4mse.npy", arr = vbf_3pi4_mse)

        np.save(file = f"{phi_invariance_study_directory}/vbfpi4kl.npy", arr = vbf_pi4_kl)
        np.save(file = f"{phi_invariance_study_directory}/vbfpi2kl.npy", arr = vbf_pi2_kl)
        np.save(file = f"{phi_invariance_study_directory}/vbf3pi4kl.npy", arr = vbf_3pi4_kl)


        #np.save(file = f"{latent_code_directory}/vbfpi4kl.npy", arr = vbf_test_pi4_latent_codes)
        #np.save(file = f"{latent_code_directory}/ggf0pi2kl.npy", arr = vbf_test_pi2_latent_codes)
        #np.save(file = f"{latent_code_directory}/vbf3pi4kl.npy", arr = vbf_test_3pi4_latent_codes)

        ggf_pi4_mse,ggf_pi4_kl,ggf_test_pi4_total_losses,ggf_test_pi4_latent_codes= evaluate(nf = nf,model = model,
                                            dataloader = ggf_test_pi4_dataset)
        ggf_pi2_mse,ggf_pi2_kl,ggf_test_pi2_total_losses,ggf_test_pi2_latent_codes = evaluate(nf = nf,model = model,
                                            dataloader = ggf_test_pi2_dataset)
        ggf_3pi4_mse,ggf_3pi4_kl,ggf_test_3pi4_total_losses,ggf_test_3pi4_latent_codes = evaluate(nf = nf,model = model,
                                            dataloader = ggf_test_3pi4_dataset)

        np.save(file = f"{phi_invariance_study_directory}/ggfpi4totallosses.npy", arr = ggf_test_pi4_total_losses)
        np.save(file = f"{phi_invariance_study_directory}/ggfpi2totallosses.npy", arr = ggf_test_pi2_total_losses)
        np.save(file = f"{phi_invariance_study_directory}/ggf3pi4totallosses.npy", arr = ggf_test_3pi4_total_losses)

        np.save(file = f"{phi_invariance_study_directory}/ggfpi4mse.npy", arr = ggf_pi4_mse)
        np.save(file = f"{phi_invariance_study_directory}/ggfpi2mse.npy", arr = ggf_pi2_mse)
        np.save(file = f"{phi_invariance_study_directory}/ggf3pi4mse.npy", arr = ggf_3pi4_mse)

        np.save(file = f"{phi_invariance_study_directory}/ggfpi4kl.npy", arr = ggf_pi4_kl)
        np.save(file = f"{phi_invariance_study_directory}/ggfpi2kl.npy", arr = ggf_pi2_kl)
        np.save(file = f"{phi_invariance_study_directory}/ggf3pi4kl.npy", arr = ggf_3pi4_kl)


        #np.save(file = f"{latent_code_directory}/ggfpi4kl.npy", arr = ggf_test_pi4_latent_codes)
        #np.save(file = f"{latent_code_directory}/ggf0pi2kl.npy", arr = ggf_test_pi2_latent_codes)
        #np.save(file = f"{latent_code_directory}/ggf3pi4kl.npy", arr = ggf_test_3pi4_latent_codes)

        hs_pi4_mse,hs_pi4_kl,hs_test_pi4_total_losses,hs_test_pi4_latent_codes= evaluate(nf = nf,model = model,
                                            dataloader = hs_test_pi4_dataset)
        hs_pi2_mse,hs_pi2_kl,hs_test_pi2_total_losses,hs_test_pi2_latent_codes = evaluate(nf = nf,model = model,
                                            dataloader = hs_test_pi2_dataset)
        hs_3pi4_mse,hs_3pi4_kl,hs_test_3pi4_total_losses,hs_test_3pi4_latent_codes = evaluate(nf = nf,model = model,
                                            dataloader = hs_test_3pi4_dataset)

        np.save(file = f"{phi_invariance_study_directory}/hspi4totallosses.npy", arr = hs_test_pi4_total_losses)
        np.save(file = f"{phi_invariance_study_directory}/hspi2totallosses.npy", arr = hs_test_pi2_total_losses)
        np.save(file = f"{phi_invariance_study_directory}/hs3pi4totallosses.npy", arr = hs_test_3pi4_total_losses)

        np.save(file = f"{phi_invariance_study_directory}/hspi4mse.npy", arr = hs_pi4_mse)
        np.save(file = f"{phi_invariance_study_directory}/hspi2mse.npy", arr = hs_pi2_mse)
        np.save(file = f"{phi_invariance_study_directory}/hs3pi4mse.npy", arr = hs_3pi4_mse)

        np.save(file = f"{phi_invariance_study_directory}/hspi4kl.npy", arr = hs_pi4_kl)
        np.save(file = f"{phi_invariance_study_directory}/hspi2kl.npy", arr = hs_pi2_kl)
        np.save(file = f"{phi_invariance_study_directory}/hs3pi4kl.npy", arr = hs_3pi4_kl)


        #np.save(file = f"{latent_code_directory}/hspi4kl.npy", arr = hs_test_pi4_latent_codes)
        #np.save(file = f"{latent_code_directory}/hs0pi2kl.npy", arr = hs_test_pi2_latent_codes)
        #np.save(file = f"{latent_code_directory}/hs3pi4kl.npy", arr = hs_test_3pi4_latent_codes)


        fig2,axes2 = plt.subplots(1,1)
        axes2.hist(jz0_test_total_losses, bins = 1000, range = (0,2),
                    histtype= "step", label = r"$\theta = 0$")
        axes2.hist(jz0_test_3pi4_total_losses, bins = 1000, range = (0,2), 
                histtype= "step", label = r"$\theta = \frac{3\pi}{2}$")
        axes2.hist(jz0_test_pi2_total_losses, bins = 1000, range = (0,2),
                    histtype= "step",label = r"$\theta = \frac{\pi}{2}$")
        axes2.hist(jz0_test_pi4_total_losses, bins = 1000, range = (0,2),
                    histtype= "step",label = r"$\theta = \frac{\pi}{4}$")
        axes2.legend()
        axes2.set_xlabel("Reconstruction Error")
        axes2.set_title("Phi Symmetry")
        fig2.savefig(f"{phi_invariance_study_directory}/phi_symmetry_total_loss.png")

        if return_kl_scores:
            return kl_ggf_signal, kl_vbf_signal, kl_hs_signal 
        
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Loading Datasets
        jz0_test_loader = ATLASMapStyleDatasetMaker(filepath =
                    "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/JZ0_no_filter.parquet",
                    lower_row_index = 200, upper_row_index = 248,
                    # See docs for explaination of why upper_row_index = 248 
                    batch_size = 1, shuffle = shuffle,
                    pileup_cutoff = pileup_cutoff,transform1 = transform1, transform2_5=transform2_5,
                    transform2 = transform2, transform3 = transform3, transform4 = transform4) 
        
        ggF_test_loader = ATLASMapStyleDatasetMaker(filepath =
                    "//home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/ggF_SM_HH4b.parquet",
                    upper_row_index= 49,
                    batch_size = 1,shuffle = shuffle,transform1 = transform1, transform2_5=transform2_5,
                    pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3, transform4 = transform4)
        
        vbf_test_loader = ATLASMapStyleDatasetMaker(filepath = 
                    "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/VBF_SM_HH4b.parquet",
                    upper_row_index= 49,
                    batch_size = 1,shuffle = shuffle,transform1 = transform1, transform2_5=transform2_5,
                    pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3, transform4 = transform4)
        
        hs_test_loader = ATLASMapStyleDatasetMaker(filepath =
                    "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/HZ_bbvv.parquet",
                    upper_row_index= 49,
                    batch_size = 1,shuffle = shuffle,transform1 = transform1, transform2_5=transform2_5,
                    pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3, transform4 = transform4)
        
        transform = lambda x: torch.roll(x, shifts = 16, dims = 3)
        vbf_test_pi4_dataset = ATLASMapStyleDatasetMaker(filepath =
                    "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/VBF_SM_HH4b.parquet",
                    lower_row_index = 0,
                    upper_row_index= 40,
                    transform = transform, batch_size = 1,shuffle = shuffle,transform1 = transform1, transform2_5=transform2_5,
                    pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3, transform4 = transform4)
        
        transform = lambda x: torch.roll(x, shifts = 32, dims = 3)
        vbf_test_pi2_dataset = ATLASMapStyleDatasetMaker(filepath =
                    "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/VBF_SM_HH4b.parquet",
                    lower_row_index = 0,
                    upper_row_index= 40,
                    transform = transform, batch_size=1,shuffle = shuffle,transform1 = transform1, transform2_5=transform2_5,
                    pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3, transform4 = transform4)
        
        transform = lambda x: torch.roll(x, shifts = 48, dims = 3)
        vbf_test_3pi4_dataset = ATLASMapStyleDatasetMaker(filepath =
                    "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/VBF_SM_HH4b.parquet",
                    lower_row_index = 0,
                    upper_row_index= 40,
                    transform = transform, batch_size=1,shuffle = shuffle,transform1 = transform1, transform2_5=transform2_5,
                    pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3, transform4 = transform4)
        
        
        transform = lambda x: torch.roll(x, shifts = 16, dims = 3)
        ggf_test_pi4_dataset = ATLASMapStyleDatasetMaker(filepath =
                    "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/ggF_SM_HH4b.parquet",
                    lower_row_index = 0,
                    upper_row_index= 40,
                    transform = transform, batch_size = 1,shuffle = shuffle,transform1 = transform1, transform2_5=transform2_5,
                    pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3, transform4 = transform4)
        
        transform = lambda x: torch.roll(x, shifts = 32, dims = 3)
        ggf_test_pi2_dataset = ATLASMapStyleDatasetMaker(filepath =
                    "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/ggF_SM_HH4b.parquet",
                    lower_row_index = 0,
                    upper_row_index= 40,
                    transform = transform, batch_size=1,shuffle = shuffle,transform1 = transform1, transform2_5=transform2_5,
                    pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3, transform4 = transform4)
        
        transform = lambda x: torch.roll(x, shifts = 48, dims = 3)
        ggf_test_3pi4_dataset = ATLASMapStyleDatasetMaker(filepath =
                    "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/ggF_SM_HH4b.parquet",
                    lower_row_index = 0,
                    upper_row_index= 40,
                    transform = transform, batch_size=1,shuffle = shuffle,transform1 = transform1, transform2_5=transform2_5,
                    pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3, transform4 = transform4)

        transform = lambda x: torch.roll(x, shifts = 16, dims = 3)
        hs_test_pi4_dataset = ATLASMapStyleDatasetMaker(filepath =
                    "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/HZ_bbvv.parquet",
                    lower_row_index = 0,
                    upper_row_index= 40,
                    transform = transform, batch_size = 1,shuffle = shuffle,transform1 = transform1, transform2_5=transform2_5,
                    pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3, transform4 = transform4)
        
        transform = lambda x: torch.roll(x, shifts = 32, dims = 3)
        hs_test_pi2_dataset = ATLASMapStyleDatasetMaker(filepath =
                    "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/HZ_bbvv.parquet",
                    lower_row_index = 0,
                    upper_row_index= 40,
                    transform = transform, batch_size=1,shuffle = shuffle,transform1 = transform1, transform2_5=transform2_5,
                    pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3, transform4 = transform4)
        
        transform = lambda x: torch.roll(x, shifts = 48, dims = 3)
        hs_test_3pi4_dataset = ATLASMapStyleDatasetMaker(filepath =
                    "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/HZ_bbvv.parquet",
                    lower_row_index = 0,
                    upper_row_index= 40,
                    transform = transform, batch_size=1,shuffle = shuffle,transform1 = transform1, transform2_5=transform2_5,
                    pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3, transform4 = transform4)
        
        transform = lambda x: torch.roll(x, shifts = 16, dims = 3)
        jz0_test_pi4_dataset = ATLASMapStyleDatasetMaker(filepath =
                    "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/JZ0_no_filter.parquet",
                    lower_row_index = 200,
                    upper_row_index= 248,
                    transform = transform, batch_size = 1,shuffle = shuffle,transform1 = transform1, transform2_5=transform2_5,
                    pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3, transform4 = transform4)
        
        transform = lambda x: torch.roll(x, shifts = 32, dims = 3)
        jz0_test_pi2_dataset = ATLASMapStyleDatasetMaker(filepath =
                    "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/JZ0_no_filter.parquet",
                    lower_row_index = 200,
                    upper_row_index= 248,
                    transform = transform, batch_size=1,shuffle = shuffle,transform1 = transform1, transform2_5=transform2_5,
                    pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3, transform4 = transform4)
        
        transform = lambda x: torch.roll(x, shifts = 48, dims = 3)
        jz0_test_3pi4_dataset = ATLASMapStyleDatasetMaker(filepath =
                    "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/JZ0_no_filter.parquet",
                    lower_row_index = 200,
                    upper_row_index= 248,
                    transform = transform, batch_size=1,shuffle = shuffle,transform1 = transform1, transform2_5=transform2_5,
                    pileup_cutoff = pileup_cutoff, transform2 = transform2, transform3 = transform3, transform4 = transform4)
        
        ggf_info = ak.from_parquet("//home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/ggF_SM_HH4b.parquet", columns = "AntiKt4TruthDressedWZJets", row_groups = range(50))
        vbf_info = ak.from_parquet("/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/VBF_SM_HH4b.parquet", columns = "AntiKt4TruthDressedWZJets", row_groups = range(50))
        hs_info = ak.from_parquet("/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/HZ_bbvv.parquet", columns = "AntiKt4TruthDressedWZJets", row_groups = range(50))

        # Determining anomaly cutoff scores
        jz0_test_mse, jz0_test_kl, jz0_test_total_losses,jz0_latent_codes = evaluate(model = model,
                                                                                    dataloader = jz0_test_loader)
        mse_cutoff= anomaly_cutoff_score(jz0_test_mse)
        kl_cutoff= anomaly_cutoff_score(jz0_test_kl)
        total_cutoff= anomaly_cutoff_score(jz0_test_total_losses)

        # Signal acceptance rates
        ggF_test_mse, ggF_test_kl, ggF_test_total,ggF_latent_codes = evaluate(model = model,
                                                                                    dataloader = ggF_test_loader)
        mse_ggf_signal = signal_acceptance_rate(ggF_test_mse, mse_cutoff)
        kl_ggf_signal = signal_acceptance_rate(ggF_test_kl, kl_cutoff)
        total_ggf_signal = signal_acceptance_rate(ggF_test_total, total_cutoff)

        vbf_test_mse, vbf_test_kl, vbf_test_total, vbf_latent_codes = evaluate(model = model,
                                                                                    dataloader = vbf_test_loader)
        mse_vbf_signal = signal_acceptance_rate(vbf_test_mse, mse_cutoff)
        kl_vbf_signal = signal_acceptance_rate(vbf_test_kl, kl_cutoff)
        total_vbf_signal = signal_acceptance_rate(vbf_test_total, total_cutoff)

        hs_test_mse, hs_test_kl, hs_test_total, hs_latent_codes = evaluate(model = model,
                                                                                dataloader = hs_test_loader)
        mse_hs_signal = signal_acceptance_rate(hs_test_mse, mse_cutoff)
        kl_hs_signal = signal_acceptance_rate(hs_test_kl, kl_cutoff)
        total_hs_signal = signal_acceptance_rate(hs_test_total, total_cutoff)

        print(f"The mse ggf singal acceptance rate is {mse_ggf_signal}")
        print(f"The kl ggf singal acceptance rate is {kl_ggf_signal}")
        print(f"The recon ggf singal acceptance rate is {total_ggf_signal}")

        print(f"The mse vbf singal acceptance rate is {mse_vbf_signal}")
        print(f"The kl vbf singal acceptance rate is {kl_vbf_signal}")
        print(f"The recon vbf singal acceptance rate is {total_vbf_signal}")

        print(f"The mse hs singal acceptance rate is {mse_hs_signal}")
        print(f"The kl hs singal acceptance rate is {kl_hs_signal}")
        print(f"The recon hs singal acceptance rate is {total_hs_signal}")

        # Missing Transverse Energy (MTE) and Scalar Summed Transverse Energy (HT) Analysis
        
        # MTE
        mte_ggf = np.sum(ggf_info.AntiKt4TruthDressedWZJets[:], axis = 1).rho
        mte_vbf = np.sum(vbf_info.AntiKt4TruthDressedWZJets[:], axis = 1).rho
        mte_hs = np.sum(hs_info.AntiKt4TruthDressedWZJets[:], axis = 1).rho

        # HT
        ht_ggf = np.sum(ggf_info.AntiKt4TruthDressedWZJets.rho, axis = 1)
        ht_vbf = np.sum(vbf_info.AntiKt4TruthDressedWZJets.rho, axis = 1)
        ht_hs = np.sum(hs_info.AntiKt4TruthDressedWZJets.rho, axis = 1)

        # MTE/HT
        mte_ht_ggf = mte_ggf / ht_ggf
        mte_ht_vbf = mte_vbf / ht_vbf
        mte_ht_hs = mte_hs / ht_hs

        # MTE of registered signals
        mte_of_registered_ggf_signals = mte_ggf[ggF_test_kl.detach().cpu().numpy() >= kl_cutoff]
        mte_of_registered_vbf_signals = mte_vbf[vbf_test_kl.detach().cpu().numpy() >= kl_cutoff]
        mte_of_registered_hs_signals = mte_hs[hs_test_kl.detach().cpu().numpy() >= kl_cutoff]

        # MTE of unregistered signals
        mte_of_unregistered_ggf_signals = mte_ggf[ggF_test_kl.detach().cpu().numpy() < kl_cutoff]
        mte_of_unregistered_vbf_signals = mte_vbf[vbf_test_kl.detach().cpu().numpy() < kl_cutoff]
        mte_of_unregistered_hs_signals = mte_hs[hs_test_kl.detach().cpu().numpy() < kl_cutoff]

        # ht of registered signals
        ht_of_registered_ggf_signals = ht_ggf[ggF_test_kl.detach().cpu().numpy() >= kl_cutoff]
        ht_of_registered_vbf_signals = ht_vbf[vbf_test_kl.detach().cpu().numpy() >= kl_cutoff]
        ht_of_registered_hs_signals = ht_hs[hs_test_kl.detach().cpu().numpy() >= kl_cutoff]

        # ht of unregistered signals
        ht_of_unregistered_ggf_signals = ht_ggf[ggF_test_kl.detach().cpu().numpy() < kl_cutoff]
        ht_of_unregistered_vbf_signals = ht_vbf[vbf_test_kl.detach().cpu().numpy() < kl_cutoff]
        ht_of_unregistered_hs_signals = ht_hs[hs_test_kl.detach().cpu().numpy() < kl_cutoff]

        # mte_ht of registered signals
        mte_ht_of_registered_ggf_signals = mte_ht_ggf[ggF_test_kl.detach().cpu().numpy() >= kl_cutoff]
        mte_ht_of_registered_vbf_signals = mte_ht_vbf[vbf_test_kl.detach().cpu().numpy() >= kl_cutoff]
        mte_ht_of_registered_hs_signals = mte_ht_hs[hs_test_kl.detach().cpu().numpy() >= kl_cutoff]

        # mte_ht of unregistered signals
        mte_ht_of_unregistered_ggf_signals = mte_ht_ggf[ggF_test_kl.detach().cpu().numpy() < kl_cutoff]
        mte_ht_of_unregistered_vbf_signals = mte_ht_vbf[vbf_test_kl.detach().cpu().numpy() < kl_cutoff]
        mte_ht_of_unregistered_hs_signals = mte_ht_hs[hs_test_kl.detach().cpu().numpy() < kl_cutoff]
        
        # Test pileup


        # Storing MTE, HT and MTE/HT 
        np.save(file = f"{signal_acceptance_directory}/mte_of_registered_ggf_signals", arr = mte_of_registered_ggf_signals)
        np.save(file = f"{signal_acceptance_directory}/mte_of_registered_vbf_signals", arr = mte_of_registered_vbf_signals)
        np.save(file = f"{signal_acceptance_directory}/mte_of_registered_hs_signals", arr = mte_of_registered_hs_signals)

        np.save(file = f"{signal_acceptance_directory}/mte_of_unregistered_ggf_signals", arr = mte_of_unregistered_ggf_signals)
        np.save(file = f"{signal_acceptance_directory}/mte_of_unregistered_vbf_signals", arr = mte_of_unregistered_vbf_signals)
        np.save(file = f"{signal_acceptance_directory}/mte_of_unregistered_hs_signals", arr = mte_of_unregistered_hs_signals)

        np.save(file = f"{signal_acceptance_directory}/ht_of_registered_ggf_signals", arr = ht_of_registered_ggf_signals)
        np.save(file = f"{signal_acceptance_directory}/ht_of_registered_vbf_signals", arr = ht_of_registered_vbf_signals)
        np.save(file = f"{signal_acceptance_directory}/ht_of_registered_hs_signals", arr = ht_of_registered_hs_signals)

        np.save(file = f"{signal_acceptance_directory}/ht_of_unregistered_ggf_signals", arr = ht_of_unregistered_ggf_signals)
        np.save(file = f"{signal_acceptance_directory}/ht_of_unregistered_vbf_signals", arr = ht_of_unregistered_vbf_signals)
        np.save(file = f"{signal_acceptance_directory}/ht_of_unregistered_hs_signals", arr = ht_of_unregistered_hs_signals)

        np.save(file = f"{signal_acceptance_directory}/mte_ht_of_registered_ggf_signals", arr = mte_ht_of_registered_ggf_signals)
        np.save(file = f"{signal_acceptance_directory}/mte_ht_of_registered_vbf_signals", arr = mte_ht_of_registered_vbf_signals)
        np.save(file = f"{signal_acceptance_directory}/mte_ht_of_registered_hs_signals", arr = mte_ht_of_registered_hs_signals)

        np.save(file = f"{signal_acceptance_directory}/mte_ht_of_unregistered_ggf_signals", arr = mte_ht_of_unregistered_ggf_signals)
        np.save(file = f"{signal_acceptance_directory}/mte_ht_of_unregistered_vbf_signals", arr = mte_ht_of_unregistered_vbf_signals)
        np.save(file = f"{signal_acceptance_directory}/mte_ht_of_unregistered_hs_signals", arr = mte_ht_of_unregistered_hs_signals)

        
        # Storing cutoff scores
        np.save(file = f"{signal_acceptance_directory}/MSECutoff.npy", arr = mse_cutoff)
        np.save(file = f"{signal_acceptance_directory}/KLCutoff.npy", arr = kl_cutoff)
        np.save(file = f"{signal_acceptance_directory}/TotalCutoff.npy", arr = total_cutoff)
        #np.save(file = f"{latent_code_directory}/jz0_latent_codes.npy", arr = jz0_latent_codes)
        np.save(file = f"{testing_losses_directory}/jz0_mse_losses.npy", arr = jz0_test_mse)
        np.save(file = f"{testing_losses_directory}/jz0_kl_losses.npy", arr = jz0_test_kl)
        np.save(file = f"{testing_losses_directory}/jz0_total_losses.npy", arr=jz0_test_total_losses)

        

        # Storing losses
        np.save(file = f"{signal_acceptance_directory}/MSEggFsignalrate.npy", arr = mse_ggf_signal)
        np.save(file = f"{signal_acceptance_directory}/KLggFsignalrate.npy", arr = kl_ggf_signal)
        np.save(file = f"{signal_acceptance_directory}/TotalggFsignalrate.npy", arr = total_ggf_signal)
        #np.save(file = f"{latent_code_directory}/ggF_latent_codes.npy", arr = ggF_latent_codes)
        np.save(file = f"{testing_losses_directory}/ggF_mse_losses.npy", arr = ggF_test_mse)
        np.save(file = f"{testing_losses_directory}/ggF_kl_losses.npy", arr = ggF_test_kl)
        np.save(file = f"{testing_losses_directory}/ggF_total_losses.npy", arr=ggF_test_total)

        np.save(file = f"{signal_acceptance_directory}/MSEvbfsignalrate.npy", arr = mse_vbf_signal)
        np.save(file = f"{signal_acceptance_directory}/KLvbfsignalrate.npy", arr = kl_vbf_signal)
        np.save(file = f"{signal_acceptance_directory}/Totalvbfsignalrate.npy", arr = total_vbf_signal)
        #np.save(file = f"{latent_code_directory}/vbf_latent_codes.npy", arr = vbf_latent_codes)
        np.save(file = f"{testing_losses_directory}/vbf_mse_losses.npy", arr = vbf_test_mse)
        np.save(file = f"{testing_losses_directory}/vbf_kl_losses.npy", arr = vbf_test_kl)
        np.save(file = f"{testing_losses_directory}/vbf_total_losses.npy", arr=vbf_test_total)


        np.save(file = f"{signal_acceptance_directory}/MSEhssignalrate.npy", arr = mse_hs_signal)
        np.save(file = f"{signal_acceptance_directory}/KLhssignalrate.npy", arr = kl_hs_signal)
        np.save(file = f"{signal_acceptance_directory}/Totalhssignalrate.npy", arr = total_hs_signal)
        #np.save(file = f"{latent_code_directory}/hs_latent_codes.npy", arr = hs_latent_codes)
        np.save(file = f"{testing_losses_directory}/hs_mse_losses.npy", arr = hs_test_mse)
        np.save(file = f"{testing_losses_directory}/hs_kl_losses.npy", arr = hs_test_kl)
        np.save(file = f"{testing_losses_directory}/hs_total_losses.npy", arr=hs_test_total)

        # Reconstructions
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))

        for batch_idx, features in enumerate(jz0_test_loader):
            if batch_idx >= 3: break  # Stop after 3 images
        
            input = features[0].to(device)
            recon_image, mu, logvar, z = model(input)
            input = input.cpu().detach().numpy()
            recon_image = recon_image.cpu().detach().numpy()

            im1 = axes[0,batch_idx].imshow(input[0,0])
            fig.colorbar(mappable = im1,ax = axes[0,batch_idx])
            axes[0,batch_idx].set_title(f"JZ0 Image {batch_idx + 1}")
        
    

        
            im2 = axes[1,batch_idx].imshow(recon_image[0,0])
            fig.colorbar(mappable = im2,ax = axes[1,batch_idx])
            axes[1,batch_idx].set_title(f"Reconstructed JZ0 Image {batch_idx + 1}")


        fig.savefig(f"{testing_losses_directory}/JZ0_Reconstructions.png")

        # phi invariance study
        jz0_pi4_mse,jz0_pi4_kl,jz0_test_pi4_total_losses,jz0_test_pi4_latent_codes= evaluate(model = model,
                                            dataloader = jz0_test_pi4_dataset)
        jz0_pi2_mse,jz0_pi2_kl,jz0_test_pi2_total_losses,jz0_test_pi2_latent_codes = evaluate(model = model,
                                            dataloader = jz0_test_pi2_dataset)
        jz0_3pi4_mse,jz0_3pi4_kl,jz0_test_3pi4_total_losses,jz0_test_3pi4_latent_codes = evaluate(model = model,
                                            dataloader = jz0_test_3pi4_dataset)

        np.save(file = f"{phi_invariance_study_directory}/jz0pi4totallosses.npy", arr = jz0_test_pi4_total_losses)
        np.save(file = f"{phi_invariance_study_directory}/jz0pi2totallosses.npy", arr = jz0_test_pi2_total_losses)
        np.save(file = f"{phi_invariance_study_directory}/jz03pi4totallosses.npy", arr = jz0_test_3pi4_total_losses)

        np.save(file = f"{phi_invariance_study_directory}/jz0pi4mse.npy", arr = jz0_pi4_mse)
        np.save(file = f"{phi_invariance_study_directory}/jz0pi2mse.npy", arr = jz0_pi2_mse)
        np.save(file = f"{phi_invariance_study_directory}/jz03pi4mse.npy", arr = jz0_3pi4_mse)

        np.save(file = f"{phi_invariance_study_directory}/jz0pi4kl.npy", arr = jz0_pi4_kl)
        np.save(file = f"{phi_invariance_study_directory}/jz0pi2kl.npy", arr = jz0_pi2_kl)
        np.save(file = f"{phi_invariance_study_directory}/jz03pi4kl.npy", arr = jz0_3pi4_kl)

        #np.save(file = f"{latent_code_directory}/jz0pi4kl.npy", arr = jz0_test_pi4_latent_codes)
        #np.save(file = f"{latent_code_directory}/jz0pi2kl.npy", arr = jz0_test_pi2_latent_codes)
        #np.save(file = f"{latent_code_directory}/jz03pi4kl.npy", arr = jz0_test_3pi4_latent_codes)

        # Signal Phi Invariance

        vbf_pi4_mse,vbf_pi4_kl,vbf_test_pi4_total_losses,vbf_test_pi4_latent_codes= evaluate(nf = nf,model = model,
                                            dataloader = vbf_test_pi4_dataset)
        vbf_pi2_mse,vbf_pi2_kl,vbf_test_pi2_total_losses,vbf_test_pi2_latent_codes = evaluate(nf = nf,model = model,
                                            dataloader = vbf_test_pi2_dataset)
        vbf_3pi4_mse,vbf_3pi4_kl,vbf_test_3pi4_total_losses,vbf_test_3pi4_latent_codes = evaluate(nf = nf,model = model,
                                            dataloader = vbf_test_3pi4_dataset)

        np.save(file = f"{phi_invariance_study_directory}/vbfpi4totallosses.npy", arr = vbf_test_pi4_total_losses)
        np.save(file = f"{phi_invariance_study_directory}/vbfpi2totallosses.npy", arr = vbf_test_pi2_total_losses)
        np.save(file = f"{phi_invariance_study_directory}/vbf3pi4totallosses.npy", arr = vbf_test_3pi4_total_losses)

        np.save(file = f"{phi_invariance_study_directory}/vbfpi4mse.npy", arr = vbf_pi4_mse)
        np.save(file = f"{phi_invariance_study_directory}/vbfpi2mse.npy", arr = vbf_pi2_mse)
        np.save(file = f"{phi_invariance_study_directory}/vbf3pi4mse.npy", arr = vbf_3pi4_mse)

        np.save(file = f"{phi_invariance_study_directory}/vbfpi4kl.npy", arr = vbf_pi4_kl)
        np.save(file = f"{phi_invariance_study_directory}/vbfpi2kl.npy", arr = vbf_pi2_kl)
        np.save(file = f"{phi_invariance_study_directory}/vbf3pi4kl.npy", arr = vbf_3pi4_kl)


        #np.save(file = f"{latent_code_directory}/vbfpi4kl.npy", arr = vbf_test_pi4_latent_codes)
        #np.save(file = f"{latent_code_directory}/ggf0pi2kl.npy", arr = vbf_test_pi2_latent_codes)
        #np.save(file = f"{latent_code_directory}/vbf3pi4kl.npy", arr = vbf_test_3pi4_latent_codes)

        ggf_pi4_mse,ggf_pi4_kl,ggf_test_pi4_total_losses,ggf_test_pi4_latent_codes= evaluate(nf = nf,model = model,
                                            dataloader = ggf_test_pi4_dataset)
        ggf_pi2_mse,ggf_pi2_kl,ggf_test_pi2_total_losses,ggf_test_pi2_latent_codes = evaluate(nf = nf,model = model,
                                            dataloader = ggf_test_pi2_dataset)
        ggf_3pi4_mse,ggf_3pi4_kl,ggf_test_3pi4_total_losses,ggf_test_3pi4_latent_codes = evaluate(nf = nf,model = model,
                                            dataloader = ggf_test_3pi4_dataset)

        np.save(file = f"{phi_invariance_study_directory}/ggfpi4totallosses.npy", arr = ggf_test_pi4_total_losses)
        np.save(file = f"{phi_invariance_study_directory}/ggfpi2totallosses.npy", arr = ggf_test_pi2_total_losses)
        np.save(file = f"{phi_invariance_study_directory}/ggf3pi4totallosses.npy", arr = ggf_test_3pi4_total_losses)

        np.save(file = f"{phi_invariance_study_directory}/ggfpi4mse.npy", arr = ggf_pi4_mse)
        np.save(file = f"{phi_invariance_study_directory}/ggfpi2mse.npy", arr = ggf_pi2_mse)
        np.save(file = f"{phi_invariance_study_directory}/ggf3pi4mse.npy", arr = ggf_3pi4_mse)

        np.save(file = f"{phi_invariance_study_directory}/ggfpi4kl.npy", arr = ggf_pi4_kl)
        np.save(file = f"{phi_invariance_study_directory}/ggfpi2kl.npy", arr = ggf_pi2_kl)
        np.save(file = f"{phi_invariance_study_directory}/ggf3pi4kl.npy", arr = ggf_3pi4_kl)


        #np.save(file = f"{latent_code_directory}/ggfpi4kl.npy", arr = ggf_test_pi4_latent_codes)
        #np.save(file = f"{latent_code_directory}/ggf0pi2kl.npy", arr = ggf_test_pi2_latent_codes)
        #np.save(file = f"{latent_code_directory}/ggf3pi4kl.npy", arr = ggf_test_3pi4_latent_codes)

        hs_pi4_mse,hs_pi4_kl,hs_test_pi4_total_losses,hs_test_pi4_latent_codes= evaluate(nf = nf,model = model,
                                            dataloader = hs_test_pi4_dataset)
        hs_pi2_mse,hs_pi2_kl,hs_test_pi2_total_losses,hs_test_pi2_latent_codes = evaluate(nf = nf,model = model,
                                            dataloader = hs_test_pi2_dataset)
        hs_3pi4_mse,hs_3pi4_kl,hs_test_3pi4_total_losses,hs_test_3pi4_latent_codes = evaluate(nf = nf,model = model,
                                            dataloader = hs_test_3pi4_dataset)

        np.save(file = f"{phi_invariance_study_directory}/hspi4totallosses.npy", arr = hs_test_pi4_total_losses)
        np.save(file = f"{phi_invariance_study_directory}/hspi2totallosses.npy", arr = hs_test_pi2_total_losses)
        np.save(file = f"{phi_invariance_study_directory}/hs3pi4totallosses.npy", arr = hs_test_3pi4_total_losses)

        np.save(file = f"{phi_invariance_study_directory}/hspi4mse.npy", arr = hs_pi4_mse)
        np.save(file = f"{phi_invariance_study_directory}/hspi2mse.npy", arr = hs_pi2_mse)
        np.save(file = f"{phi_invariance_study_directory}/hs3pi4mse.npy", arr = hs_3pi4_mse)

        np.save(file = f"{phi_invariance_study_directory}/hspi4kl.npy", arr = hs_pi4_kl)
        np.save(file = f"{phi_invariance_study_directory}/hspi2kl.npy", arr = hs_pi2_kl)
        np.save(file = f"{phi_invariance_study_directory}/hs3pi4kl.npy", arr = hs_3pi4_kl)


        #np.save(file = f"{latent_code_directory}/hspi4kl.npy", arr = hs_test_pi4_latent_codes)
        #np.save(file = f"{latent_code_directory}/hs0pi2kl.npy", arr = hs_test_pi2_latent_codes)
        #np.save(file = f"{latent_code_directory}/hs3pi4kl.npy", arr = hs_test_3pi4_latent_codes)


        fig2,axes2 = plt.subplots(1,1)
        axes2.hist(jz0_test_total_losses, bins = 1000, range = (0,2),
                    histtype= "step", label = r"$\theta = 0$")
        axes2.hist(jz0_test_3pi4_total_losses, bins = 1000, range = (0,2), 
                histtype= "step", label = r"$\theta = \frac{3\pi}{2}$")
        axes2.hist(jz0_test_pi2_total_losses, bins = 1000, range = (0,2),
                    histtype= "step",label = r"$\theta = \frac{\pi}{2}$")
        axes2.hist(jz0_test_pi4_total_losses, bins = 1000, range = (0,2),
                    histtype= "step",label = r"$\theta = \frac{\pi}{4}$")
        axes2.legend()
        axes2.set_xlabel("Reconstruction Error")
        axes2.set_title("Phi Symmetry")
        fig2.savefig(f"{phi_invariance_study_directory}/phi_symmetry_total_loss.png")

        if return_kl_scores:
            return kl_ggf_signal, kl_vbf_signal, kl_hs_signal

    







#########################################################################################################################
                                                    # ARCHIVED FUNCTIONS
#########################################################################################################################


class IterableDatasetMaker(IterableDataset):
    """
    Makes an iterable dataset out of a parquet file containing Monte-Carlo simulations
    of events at ATLAS experiment at the LHC.

    NB. This is VERY SLOW! Don't use unless dataset is larger than RAM. 
    Pileup supression needs to be optimised to alleviate this to some capacity.
    """

    def __init__(self, filepath, low_row_num = 0, up_row_num = 200, transform = None):
        self.dataset = pa.ParquetFile(filepath)
        self.low_row_num = low_row_num # Lower limit on index of row groups to get items from.
        self.up_row_num = up_row_num # Upper limit on index of row groups to get items from.
        self.transform = transform
    
    def __iter__(self):
        for image in self.dataset.iter_batches(batch_size = 1, columns = ["cell_towers"],row_groups = range(self.low_row_num, self.up_row_num)):
            image = ak.from_arrow(image)
            image = ak.to_torch(image.cell_towers).to(torch.float32)

            # Batch ID X Layer X Eta X Phi
            image = torch.permute(image, dims = (0,3,1,2))                
            
            # Pileup supression
            image = image.view(6,50,64) # Getting rid of excess dimension (caused problems later)
            image_summed = torch.sum(image, dim = 0)
            image_eta, image_phi = torch.where(image_summed < 2)
            for eta,phi in zip(image_eta, image_phi):
                image[:,eta, phi] = 0 
            
            if self.transform:
                yield self.transform(image)
            else:
                yield image 







def old_training_loss(recon_image, input_image, mu, logvar, beta = 1):
    """
    Computes the total weighted training loss function for
    the VAE:
    
    ( (1 - beta) * MSE + beta * KL Divergence).
    """
    
    recon_loss = F.mse_loss(recon_image, input_image, reduction = "sum") 
    kl_loss = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = (1 - beta) * recon_loss + beta * kl_loss
    return recon_loss, kl_loss, total_loss







def old_unadulterated_training_loss(recon_image, input_image, mu, logvar):
    """
    Computes the total training loss function for
    the VAE (MSE + KL Divergence).
    """

    recon_loss = F.mse_loss(recon_image, input_image, reduction = "sum")
    kl_loss = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + kl_loss
    return recon_loss, kl_loss, total_loss







def old_mean_training_loss(recon_image, input_image, mu, logvar):
    """
    Computes the total training loss function for
    the VAE (MSE + KL Divergence).
    """

    recon_loss = F.mse_loss(recon_image, input_image, reduction = "mean")
    kl_loss = - 0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + kl_loss
    return recon_loss, kl_loss, total_loss







def old_mean_testing_loss(recon_image, input_image, mu, logvar):
    """
    Computes the total testing loss function for
    the VAE (MSE + KL Divergence).
    """
    
    recon_loss = F.mse_loss(recon_image, input_image, reduction = "mean")
    kl_loss = - 0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + kl_loss
    return recon_loss, kl_loss, total_loss







def old_testing_loss(recon_image, input_image, mu, logvar):
    """
    Computes the total testing loss function for
    the VAE (MSE + KL Divergence).
    """
    
    recon_loss = F.mse_loss(recon_image, input_image, reduction = "sum")
    kl_loss = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + kl_loss
    return recon_loss, kl_loss, total_loss







def old_train(model, dataloader, device, optimiser, loss_function, num_events = 200000):
    """
    Trains VAE.

    Assuming 200K events are used by default. Change if this isn't true
    """

    
    recon_loss = 0
    kl_loss = 0
    total_loss = 0

    model.train() 
    for images in dataloader:
        optimiser.zero_grad() 
        images = images.to(device)
        
        # Forward pass
        recon_image, mu, logvar = model(images) 
        batch_recon_loss,batch_kl_loss,batch_total_loss = loss_function(recon_image, images, mu, logvar)

        # Backward pass and weight update
        batch_total_loss.backward() 
        optimiser.step() 

        # Cumulative error until current batch
        recon_loss += batch_recon_loss.item()
        kl_loss += batch_kl_loss.item()
        total_loss += batch_total_loss.item() 


    # Average loss of epoch
    avg_recon_loss = recon_loss / num_events
    avg_kl_loss = kl_loss / num_events 
    avg_total_loss = total_loss / num_events

    return avg_recon_loss, avg_kl_loss, avg_total_loss





def old_valid(model, dataloader, device, loss_function, num_events = 5000):
    """
    Validates VAE during training.

    Assuming 5K events used for validation by default. Change if this isn't true 

    Must use testing_loss() here!
    """

    model.eval()
    recon_loss = 0
    kl_loss = 0
    total_loss = 0
    with torch.no_grad(): 
        for images in dataloader:
            images = images.to(device)
            recon_images, mu, logvar = model(images)
            batch_recon_loss, batch_kl_loss, batch_total_loss = loss_function(recon_images, images, mu, logvar)
            
            recon_loss += batch_recon_loss.item()
            kl_loss += batch_kl_loss.item()
            total_loss += batch_total_loss.item()
    
    
    avg_recon_loss = recon_loss / num_events
    avg_kl_loss = kl_loss / num_events
    avg_total_loss = total_loss / num_events 
    return avg_recon_loss, avg_kl_loss, avg_total_loss


def tune(project_name, model, epochs = 100, lr = 1e-4):
    
    """
    Tunes beta term in VAE to achieve maximal signal acceptance rates.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_training_events = 150000
    num_validation_events = 50000
    batch_size = 128

    config = {"epochs": epochs, "lr":lr}

    # Training datasets
    training_data_loader = ATLASMapStyleDatasetMaker(filepath 
                            = "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/JZ0_no_filter.parquet",
                            upper_row_index = 149,
                              batch_size = batch_size)
    validation_data_loader = ATLASMapStyleDatasetMaker(filepath = 
                            "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/JZ0_no_filter.parquet",
                            lower_row_index = 150,
                              upper_row_index = 199,
                                batch_size = batch_size)
    
    # Testing datasets
   
    jz0_test_loader = ATLASMapStyleDatasetMaker(filepath =
            "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/JZ0_no_filter.parquet",
            lower_row_index = 200, upper_row_index = 209,
            # See docs for explaination of why upper_row_index = 248 
            batch_size = 1) 

    
    hs_test_loader = ATLASMapStyleDatasetMaker(filepath=
            "/home/xzcapask/atlas_ad_hllhc/data/atlas_calorimeter_images/HZ_bbvv.parquet",
            upper_row_index= 9,
            batch_size = 1)
  
    
    model = model.to(device) 
    
    # Storing losses from each epoch
    valid_recon_losses = []
    valid_kl_losses = []
    valid_total_losses = []
    train_recon_losses = []
    train_kl_losses = []
    train_total_losses = []

    wandb.login()
    with wandb.init(project = project_name) as run:
        run.name = f"beta_up = {run.config.beta_upper_limit}"
        optimiser = torch.optim.Adam(model.parameters(), lr = lr)

        beta = np.empty(100)
        beta[:50] = run.config["beta_upper_limit"] / 50 * np.linspace(0,49,50)
        beta[50:] = run.config["beta_upper_limit"]
        
        for i in range(epochs):
            print(f"Starting Training Epoch {i + 1}")



            # Training loop =
            train_avg_recon_loss,train_avg_kl_loss, train_avg_total_loss = epoch_train(model = model,
                                                                                        dataloader = training_data_loader, 
                                                                                        device = device, 
                                                                                        optimiser = optimiser, 
                                                                                        beta = beta[i],
                                                                                        num_events = num_training_events)
            train_recon_losses.append(train_avg_recon_loss)
            train_kl_losses.append(train_avg_kl_loss)
            train_total_losses.append(train_avg_total_loss)

            # Validation loop
            valid_avg_recon_loss,valid_avg_kl_loss, valid_avg_total_loss = epoch_valid(model = model,
                                                                                       dataloader = validation_data_loader,
                                                                                       device = device,
                                                                                       num_events = num_validation_events)
            valid_recon_losses.append(valid_avg_recon_loss)
            valid_kl_losses.append(valid_avg_kl_loss)
            valid_total_losses.append(valid_avg_total_loss)
            
            run.log({"Training Total Loss": train_avg_total_loss,"Training Recon Loss": train_avg_recon_loss,
                      "Training KL Loss": train_avg_kl_loss, "Validation Total Loss": valid_avg_total_loss,
                      "Validation Recon Loss": valid_avg_recon_loss, "Validation KL Loss": valid_avg_kl_loss,"beta": beta[i]})
        
        # Anomaly cutoff score determination
        _, jz0_test_kl, _, _ = evaluate(model = model, dataloader = jz0_test_loader)
        
        _, hs_test_kl, _, _ = evaluate(model = model, dataloader = hs_test_loader)
        
        kl_cutoff = anomaly_cutoff_score(jz0_test_kl)
        hs_kl_signal_acceptance_rate = signal_acceptance_rate(signal_test_loss = hs_test_kl, cutoff_score = kl_cutoff)

        run.log({"hs_kl_signal_acceptance_rate" : hs_kl_signal_acceptance_rate})   
    print("Training Complete")