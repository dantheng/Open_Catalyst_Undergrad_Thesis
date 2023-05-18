# Import Libraries
import numpy as np
import pandas as pd

from ocpmodels.datasets import LmdbDataset
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)
from ocpmodels.datasets.embeddings import KHOT_EMBEDDINGS, QMOF_KHOT_EMBEDDINGS
from ocpmodels.models.base import BaseModel

import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool, radius_graph
from torch_geometric.nn.models.schnet import GaussianSmearing

import wandb
import os.path as osp
import sys
import pathlib
import tqdm
from tqdm import trange, tqdm
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.trial import TrialState
import os
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21"

# Prep Data
## Load Data
# train_src = "/home/dantheng/OC20/is2res_train_val_test_lmdbs/data/is2re/all/train/data.lmdb"
# train_dataset = LmdbDataset({"src": train_src})
# len(train_dataset)

## Create Pytorch Geometric Dataset Class
class oc_dataset(torch_geometric.data.Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        #self.lmdbdataset = LmdbDataset({"src": root+"/"+raw_file_names})    
    @property
    def raw_file_names(self):
      return "data.lmdb"

    # def lmdbdataset(self):
    #    return LmdbDataset({"src": self.root+"/"+self.raw_file_names})

    @property
    def processed_file_names(self):
        '''list the files you wanna keep so that if they exist in the processed directory don't need to process again'''
        processed_filenames = []
        for i in range(len(LmdbDataset({"src": self.root+"/"+self.raw_file_names}))):
          processed_filenames.append(f'data_{i}.pt')
        # return ['data_1.pt', 'data_2.pt', ...]
        return processed_filenames
        pass

    def download(self):
        pass

    def process(self):
        idx = 0
        lmdbdataset = LmdbDataset({"src": self.root+"/"+self.raw_file_names}) 
        for i in range(len(LmdbDataset({"src": self.root+"/"+self.raw_file_names}))):
          data = lmdbdataset[i]
          
          if self.pre_filter is not None and not self.pre_filter(data):
                continue
          
          if self.pre_transform is not None:
                data = self.pre_transform(data)
          


          torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
          idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data

## Init Train Dataset
OC_dataset_all = oc_dataset(root = "/home/dantheng/OC20/is2res_train_val_test_lmdbs/data/is2re/100k/train")

## Init Test_id Dataset
OC_dataset_test_id = oc_dataset(root = "/home/dantheng/OC20/is2res_train_val_test_lmdbs/data/is2re/all/val_id")


## Create Dataloaders
train_loader = DataLoader(OC_dataset_all, batch_size=128, shuffle=True)
test_loader = DataLoader(OC_dataset_test_id, batch_size=128, shuffle=False)

# Define Wandb config and callback
wandb_kwargs = {
    "project": "OC_LR_only",                    
    "entity":"dantheng",
    "reinit": True}                                                                                                                                                                                                                                                                                                                                         
wandbc = WeightsAndBiasesCallback(metric_name="MAE", wandb_kwargs=wandb_kwargs, as_multirun=True)

# Training 
from IPython.display import Javascript

#display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

from ocpmodels.models import cgcnn

@wandbc.track_in_wandb()
def objective (trial):
    #Define Model
    model = cgcnn.CGCNN(num_atoms=78, 
                    bond_feat_dim= 50, #2861
                    num_targets=1, 
                    use_pbc=True,
                    regress_forces=False, #True
                    atom_embedding_size=64,
                    num_graph_conv_layers=6,
                    fc_feat_size=128,
                    num_fc_layers=4,
                    otf_graph=True, #False
                    cutoff=6.0,
                    num_gaussians=50, #50
                    embeddings="khot"
                    )

    device = torch.device('cuda:0')
    model = model.to(device)
    LR = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    #hyperparameter tuning: LR, Depth of Graph CONV or FC. Hold steady other and just change 1 variable at a time.  

    criterion = torch.nn.L1Loss()

    #Define Train Function
    def train():
        model.train()
        count = 0
        with tqdm(train_loader, unit="batch") as tepoch:
            for data in tepoch:  # Iterate in batches over the training dataset.
                tepoch.set_description(f"Epoch {epoch}")
                data = data.to(device)
                out = model(data)  # Perform a single forward pass.
                loss = criterion(out, data.y_relaxed.view(-1,1))  # Compute the loss.
                loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
                optimizer.zero_grad(set_to_none=True)  # Clear gradients.
                del loss
                del out
                count += 1

    #Define Test Function
    def test(loader):
        torch.cuda.empty_cache()
        gc.collect()
        abs_error = 0

        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                model.eval()
                out = model(data)
                abs_error += sum(abs(out-data.y_relaxed.view(-1,1))) 
                del out

        num_samples = len(loader.dataset)
        MAE = abs_error/num_samples
        return MAE  # Derive ratio of correct predictions.

    #Prep wandb
    # config = dict(trial.params)
    # config["trial.number"] = trial.number
    # wandb.init(
    #     project="OC_LR_only",
    #     entity="dantheng",  # NOTE: this entity depends on your wandb account.
    #     config=config,
    #     group="OC_project_LR_only",
    #     reinit=True,
    # )  


    for epoch in range(1,26):
        print(f'Epoch: {epoch:03d}')
        train()
        print("FINISHED TRAINING")
        torch.cuda.empty_cache()
        gc.collect()

        train_MAE = test(train_loader)
        print("FINISHED TRAIN TEST!!! - " + f'Train MAE: {train_MAE.item():.4f}')
        test_MAE = test(test_loader)
        print("FINISHED TEST TEST!!! - " + f'Eval MAE: {test_MAE.item():.4f}')
         
        # report validation accuracy to wandb
        wandb.log(data={"training accuracy": train_MAE.item()}, step=epoch)
        wandb.log(data={"validation accuracy": test_MAE.item()}, step=epoch)

        torch.cuda.empty_cache()
        gc.collect()

    torch.save(model.state_dict(), "/home/dantheng/OC20/model_LR_only_ckpt.pt")

    # Handle pruning based on the intermediate value.
    if trial.should_prune():
        wandb.run.summary["state"] = "pruned"
        wandb.finish(quiet=True)
        raise optuna.exceptions.TrialPruned()
    
    # report the final validation accuracy to wandb
    wandb.run.summary["final MAE"] = test_MAE.item()
    wandb.run.summary["state"] = "completed"
    wandb.finish(quiet=True)

    return test_MAE

if __name__ == "__main__":
    study_name = "OC_project_LR_only"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(sampler = optuna.samplers.TPESampler(), 
                                pruner = optuna.pruners.HyperbandPruner(),  
                                direction="minimize",
                                study_name = study_name,
                                storage = storage_name,
                                load_if_exists = True)
    study.optimize(objective, 
                    n_trials=6, 
                    timeout=None,
                    callbacks=[wandbc],
                    gc_after_trial = True,
                    show_progress_bar = True)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    
    print(f"Best value: {study.best_value} (params: {study.best_params})")

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

