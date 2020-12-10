# import packages
import os, gc
import zarr
import numpy as np 
import pandas as pd 
from tqdm import tqdm
from typing import Dict, List, Callable
from collections import Counter
import math

#level5 toolkit
from l5kit.dataset import EgoDataset, AgentDataset
from l5kit.data import ChunkedDataset, LocalDataManager

# level5 toolkit 
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace

# visualization
import matplotlib.pyplot as plt

# deep learning
import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torch.nn import functional as F
import timm

# package
from utils import training_cfg, validate_cfg, pytorch_neg_multi_log_likelihood_single
from model import LyftMixnet, LyftMobile, LyftModel, LyftMixModel


def pytorch_neg_multi_log_likelihood_custom(
    gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor
) -> Tensor:
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert gt.shape == (batch_size, future_len, num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"
    if not torch.allclose(torch.sum(confidences, dim=1), confidences.new_ones((batch_size,))):
        print(confidences)
    assert torch.allclose(torch.sum(confidences, dim=1), confidences.new_ones((batch_size,))), "confidences should sum to 1"
    assert avails.shape == (batch_size, future_len), f"expected 1D (Time) array for gt, got {avails.shape}"
    # assert all data are valid
    assert torch.isfinite(pred).all(), "invalid value found in pred"
    assert torch.isfinite(gt).all(), "invalid value found in gt"
    assert torch.isfinite(confidences).all(), "invalid value found in confidences"
    assert torch.isfinite(avails).all(), "invalid value found in avails"

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one
    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes
    # print("error", error)
    return error


def custom_angle_loss(
    gt: Tensor, pred: Tensor, avails: Tensor, device, penalize=1.25
) -> Tensor:
    """
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: array of shape (bs)x(time), custom parameter to penalize more on corner cases.
    """   
    # pred (bs)x(time)x(2D coords) --> (bs)x(mode=1)x(time)x(2D coords)
    # create confidence (bs)x(mode=1)
    
    
    batch_size, future_len, num_coords = pred.shape
    confidences = pred.new_ones((batch_size, 1))
    
    y_tensor = torch.tensor((), dtype=torch.float64)
    yaws = y_tensor.new_zeros((batch_size,),requires_grad=False)
    for i in range(batch_size):
        # compare first 
        sing_avail = avails[i]
        sing_targets = gt[i]
        ones = (sing_avail == 1).nonzero()
        first,last = ones[0],ones[-1]
        # print(first, last)
        dx = sing_targets[last][0][0].item() - sing_targets[first][0][0].item()
        dy = sing_targets[last][0][1].item() - sing_targets[first][0][1].item()

        yaw_agent = math.acos(abs(dx) / (math.sqrt(dx**2 + dy**2))) * penalize
        yaw_agent = 0.5 if yaw_agent < 0.5 else yaw_agent
        yaws[i] = yaw_agent
        
    yaws.to(device).detach()

    return (pytorch_neg_multi_log_likelihood_custom(gt, pred.unsqueeze(1), confidences, avails) * yaws).mean()
    




######################## Configurations ################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# choose parameter conf
conf = 2
# root directory
DIR_INPUT = "PATH_TO_DATASET"
# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
dm = LocalDataManager(None)
model_name = "model_state_mixnet_l.pth"
######################## Configurations ################################


# training cfg
train_cfg = training_cfg["train_data_loader"]

# rasterizer
rasterizer = build_rasterizer(training_cfg, dm)

# dataloader
train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
train_dataset = AgentDataset(training_cfg, train_zarr, rasterizer)
train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
                             num_workers=train_cfg["num_workers"])
print(train_dataset)



if conf == 1:
    # resnet stuff
    model = LyftModel(training_cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=2e-6)
    criterion = nn.SmoothL1Loss()

if conf == 2:
    # mixnet_large

    model = LyftMixnet(training_cfg, 'mixnet_l').to(device)
    
    WEIGHT_FILE = '../data_model/pth/model_state_mixnetl_25000_17000_nll.pth'
    model_state = torch.load(WEIGHT_FILE, map_location=device)
    model.load_state_dict(model_state)
    
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=2e-6)
    criterion = pytorch_neg_multi_log_likelihood_single
    
if conf == 3:
    # mixnet_medium
    model = LyftMixnet(training_cfg, 'mixnet_m').to(device)
    optimizer = optim.Adam(model.parameters(), lr=4e-4, weight_decay=2e-6)
    criterion = nn.SmoothL1Loss()

if conf == 4:
    model = LyftMobile(training_cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=2e-6)
    criterion = nn.SmoothL1Loss()

############################# Train Model ####################################

tr_it = iter(train_dataloader)
progress_bar = tqdm(range(training_cfg["train_params"]["max_num_steps"]))

losses_train = []
likelihood_valid = []

for iter_step in progress_bar:
    try:
        data = next(tr_it)
    except StopIteration:
        tr_it = iter(train_dataloader)
        data = next(tr_it)
    model.train()
    torch.set_grad_enabled(True)
    
    # forward pass
    inputs = data["image"].to(device)
    target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
    targets = data["target_positions"].to(device)
    
    outputs = model(inputs).reshape(targets.shape).to(device)
    # use Negative Multi Log Likelihood

    if conf == 2:
        loss = criterion(targets, outputs, target_availabilities.squeeze(-1))
    else:
        loss = criterion(outputs, targets)
        # not all the output steps are valid, but we can filter them out from the loss using availabilities
        loss = loss * target_availabilities
        loss = loss.mean()

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses_train.append(loss.item())
        
    progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")
    


################################# Loss Plot & NLL Plot ###################################
# Training Loss plot
l = [i for i in range(1, len(losses_train)+1)]
plt.plot(l, losses_train,label="Training Loss")
plt.title("Loss Performance Versus Steps")
plt.rcParams['figure.figsize'] = 10, 10
plt.xlabel("Scenes")
plt.ylabel("Loss")
plt.savefig("{model_name}.png")
plt.show()


# save full trained model
torch.save(model.state_dict(), model_name)