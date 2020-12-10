# import packages
import os
import zarr
import numpy as np 
import pandas as pd 
from tqdm import tqdm
from typing import Dict, List, Callable
from collections import OrderedDict
import math

#level5 toolkit
from l5kit.dataset import EgoDataset, AgentDataset
from l5kit.data import ChunkedDataset, LocalDataManager

# level5 toolkit 
from l5kit.geometry import transform_points
from l5kit.rasterization import build_rasterizer
from l5kit.visualization import draw_trajectory, draw_reference_trajectory, TARGET_POINTS_COLOR, PREDICTED_POINTS_COLOR, write_gif
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset, export_zarr_to_csv, write_gt_csv
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace

# visualization
import matplotlib.pyplot as plt

# deep learning
import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18, resnet50, resnet34
from torchvision.models.mobilenet import mobilenet_v2
from torch.nn import functional as F
import timm

# ensemble learning
from sklearn.cluster import KMeans

# package
from utils import training_cfg, validate_cfg, pytorch_neg_multi_log_likelihood_single
from model import LyftMixnet, LyftMobile, LyftModel, LyftMixModel



def kmeans_ensemble(models:List[Callable], data:Dict, num_cluster=3) -> np.ndarray:
    """

    Args:
        models (Iterable): list of models
        data (Dict): data from dataloader
    Returns:
        np.ndarray: array of shape (modes)x(timesteps)x(2D coords), predicted tractories modeled by clustering
    """
    assert len(models) >= num_cluster
    
    inputs = data["image"].to(device)
    target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
    targets = data["target_positions"].to(device)
    clusters = []
    
    batch_size = targets.shape[0]
    num_model = len(models)
    
    for model in models:
        outputs = model(inputs)
        clusters.append(outputs.cpu().numpy().copy())
    
    num_outputs = outputs.shape[1] # 100

    arr = np.empty((num_model, num_outputs) )
    batch_centers = np.empty((batch_size, num_cluster, num_outputs//2, 2))
    for i in range(batch_size):
        for j in range(num_model):
            
            assert clusters[j][i].shape == arr[j].shape
            arr[j] = clusters[j][i]
            
        km = KMeans(n_clusters=num_cluster).fit(arr)
        centers = km.cluster_centers_
        centers = centers.reshape(num_cluster,50,2)
        batch_centers[i] = centers
    return batch_centers

    



def multiple(models:List[Callable], data:Dict) -> np.ndarray:
    """

    Args:
        models (Iterable): list of models
        data (Dict): data from dataloader
    Returns:
        np.ndarray: array of shape (modes)x(timesteps)x(2D coords), predicted tractories modeled by clustering
    """
    assert len(models) > 0
    
    inputs = data["image"].to(device)
    target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
    targets = data["target_positions"].to(device)
    res = []
    
    batch_size = targets.shape[0]
    num_model = len(models)
    
    for i, model in enumerate(models):
        outputs = model(inputs)
        res.append(outputs.cpu().numpy().copy().reshape((batch_size, 50, 2)))
        
        
    batch_output = np.empty((batch_size, num_model, 50, 2))
    
    for i in range(batch_size):
        for j in range(num_model):
            
            batch_output[i][j] = res[j][i]
            
    return batch_output


def load_model(model_paths):
    models = []
    for i, path in enumerate(model_paths):
        if i == 0:
            model = LyftMixnet(validate_cfg, 'mixnet_m').to(device)
        elif i == 1 or i == 2:
            model = LyftMixModel(validate_cfg).to(device)

        else:
            model = LyftMixnet(validate_cfg, 'mixnet_l').to(device)
        model_state = torch.load(path, map_location=device)
        model.load_state_dict(model_state)
        models.append(model)
    return models



######################## Configurations ################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# choose parameter conf
TEST_CONF = ["Kmeans", "Ensemble", "Single"]
conf = 2
test = "Single"
eval_gt_path = "valid_gt_7000.csv"
pred_path = 'submission_mixnetl_ensemble.csv'
metric_path = 'metric_mixnetl_kmeans.npy'
# root directory
DIR_INPUT = "PATH_TO_DATASET"
# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
dm = LocalDataManager(None)
######################## Configurations ################################


if conf == 1:
    # resnet stuff
    model = LyftModel(training_cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=2e-6)
    criterion = nn.SmoothL1Loss()
    # criterion = nn.MSELoss(reduction="none")

if conf == 2:
    # mixnet_large

    model = LyftMixnet(validate_cfg, 'mixnet_l').to(device)
    
    WEIGHT_FILE = '../input/resnet-34-pth/model_state_mixnetl_25000_17000_nll.pth'
    model_state = torch.load(WEIGHT_FILE, map_location=device)
    model.load_state_dict(model_state)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=2e-6)
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

if conf == 5:
    model_paths = [
        "../data_model/pth/mixnetm_35000.pth",
        ".../data_model/pth/model_state_mixnet_xl_12000.pth",
        "../data_model/pth/model_state_mixnet_xl_nll_8000.pth",
        "../data_model/pth/model_state_mixnetl_25000_17000_nll.pth",
        "../data_model/pth/model_state_mixnetl_25000.pth",
    ]

    models = load_model(model_paths)




# validation configuration
valid_cfg = validate_cfg["validate_data_loader"]

# Rasterizer
rasterizer = build_rasterizer(validate_cfg, dm)

# Validation dataset/dataloader
valid_zarr = ChunkedDataset(dm.require(valid_cfg["key"])).open()
valid_dataset = AgentDataset(validate_cfg, valid_zarr, rasterizer)
whole_size = valid_dataset.__len__()
valid_dataset_use, valid_dataset_valid, _ = torch.utils.data.random_split(valid_dataset, [7000, 2000, whole_size-9000], generator=torch.Generator().manual_seed(42))

valid_dataloader = DataLoader(valid_dataset_use,
                             shuffle=valid_cfg["shuffle"],
                             batch_size=valid_cfg["batch_size"],
                             num_workers=valid_cfg["num_workers"])

valid_dataloader_valid = DataLoader(valid_dataset_valid,
                             shuffle=valid_cfg["shuffle"],
                             batch_size=valid_cfg["batch_size"],
                             num_workers=valid_cfg["num_workers"])

print("Validate Dataset Size:", valid_dataloader.dataset.__len__())




##################### Evaluation ################################

# store information for evaluation
future_coords_offsets_pd = []
timestamps = []
# coordinates ground truth
valid_coords_gts = []
# target avalabilities
target_avail_pd = []
agent_ids = []


if test == TEST_CONF[2]:
    # Final - Evaluate Validation Dataset 
    model.eval()
    torch.set_grad_enabled(False)


    progress_bar = tqdm(valid_dataloader)
    for data in progress_bar:
        
        inputs = data["image"].to(device)
        target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
        targets = data["target_positions"].to(device)
        

        outputs = model(inputs).reshape(targets.shape)
        
        future_coords_offsets_pd.append(outputs.cpu().numpy().copy())
        timestamps.append(data["timestamp"].numpy().copy())
        agent_ids.append(data["track_id"].numpy().copy())
        valid_coords_gts.append(data["target_positions"].numpy().copy())
        target_avail_pd.append(target_availabilities.cpu().numpy().copy())


if test == TEST_CONF[1]:
    for model in models:
        model.eval()
    torch.set_grad_enabled(False)

    progress_bar = tqdm(valid_dataloader)
    for data in progress_bar:
        
        outputs = kmeans_ensemble(models, data)
        future_coords_offsets_pd.append(outputs.copy())
        timestamps.append(data["timestamp"].numpy().copy())
        agent_ids.append(data["track_id"].numpy().copy())
        valid_coords_gts.append(data["target_positions"].numpy().copy())
        target_avail_pd.append(data["target_availabilities"].unsqueeze(-1).numpy().copy())


if test == TEST_CONF[0]:
    # # Final Evaluation -- Kmeans
    # Final - Evaluate Validation Dataset 
    for model in models:
        model.eval()
    torch.set_grad_enabled(False)

    progress_bar = tqdm(valid_dataloader)
    for data in progress_bar:
        
        outputs = kmeans_ensemble(models, data)
        future_coords_offsets_pd.append(outputs.copy())
        timestamps.append(data["timestamp"].numpy().copy())
        agent_ids.append(data["track_id"].numpy().copy())
        valid_coords_gts.append(data["target_positions"].numpy().copy())
        target_avail_pd.append(data["target_availabilities"].unsqueeze(-1).numpy().copy())



timestamps_concat = np.concatenate(timestamps)
track_ids_concat = np.concatenate(agent_ids)
coords_concat = np.concatenate(future_coords_offsets_pd)
gt_valid_final = np.concatenate(valid_coords_gts)
target_avail_concat = np.concatenate(target_avail_pd)




if test == TEST_CONF[0] or test == TEST_CONF[1]:

    # generate ground truth csv
    write_gt_csv(
        csv_path=eval_gt_path, 
        timestamps=timestamps_concat, 
        track_ids=track_ids_concat, 
        coords=gt_valid_final, 
        avails=target_avail_concat.squeeze(-1)
    )

    num_examples = gt_valid_final.shape[0]
    confidence = np.array([0.33,0.33,0.34])
    confidences= np.empty((num_examples, 3))
    for i in range(num_examples):
        confidences[i] = confidence
        
        
    # submission.csv
    write_pred_csv(pred_path,
                timestamps=timestamps_concat,
                track_ids=track_ids_concat,
                coords=coords_concat,
                confs=confidences
                )

    metrics = compute_metrics_csv(eval_gt_path, pred_path, [neg_multi_log_likelihood, time_displace])
    for metric_name, metric_mean in metrics.items():
        print(metric_name, metric_mean)
    
    # Save Metric
    np.save(metric_path,metrics)


if test == TEST_CONF[2]:

    # generate ground truth csv
    write_gt_csv(
        csv_path=eval_gt_path, 
        timestamps=timestamps_concat, 
        track_ids=track_ids_concat, 
        coords=gt_valid_final, 
        avails=target_avail_concat.squeeze(-1)
    )

        
        
    # submission.csv
    write_pred_csv(pred_path,
                timestamps=timestamps_concat,
                track_ids=track_ids_concat,
                coords=coords_concat,
                # confs=confidences
                )

    metrics = compute_metrics_csv(eval_gt_path, pred_path, [neg_multi_log_likelihood, time_displace])
    for metric_name, metric_mean in metrics.items():
        print(metric_name, metric_mean)
        
    # Save Metric
    np.save(metric_path,metrics)



######################## Plot Prediction Tractories ##################################

model.eval()
torch.set_grad_enabled(False)

# Uncomment to choose satelliter or semantic rasterizer
# validate_cfg["raster_params"]["map_type"] = "py_satellite"
validate_cfg["raster_params"]["map_type"] = "py_semantic"

rast = build_rasterizer(validate_cfg, dm)

eval_ego_dataset = EgoDataset(validate_cfg, valid_dataset.dataset, rast)
num_frames = 2 # randomly pick _ frames
random_frames = np.random.randint(0,len(eval_ego_dataset)-1, (num_frames,))

for frame_number in random_frames:  
    agent_indices = valid_dataset.get_frame_indices(frame_number) 
    if not len(agent_indices):
        continue

    # get AV point-of-view frame
    data_ego = eval_ego_dataset[frame_number]
    im_ego = rasterizer.to_rgb(data_ego["image"].transpose(1, 2, 0))
    center = np.asarray(validate_cfg["raster_params"]["ego_center"]) * validate_cfg["raster_params"]["raster_size"]
    
    predicted_positions = []
    target_positions = []

    for v_index in agent_indices:
        data_agent = valid_dataset[v_index]

        out_net = model(torch.from_numpy(data_agent["image"]).unsqueeze(0).to(device))
        out_pos = out_net[0].reshape(-1, 2).detach().cpu().numpy()
        # store absolute world coordinates
        predicted_positions.append(transform_points(out_pos, data_agent["world_from_agent"]))
        # retrieve target positions from the GT and store as absolute coordinates
        track_id, timestamp = data_agent["track_id"], data_agent["timestamp"]
        target_positions.append(transform_points(data_agent["target_positions"], data_agent["world_from_agent"]) )

    # convert coordinates to AV point-of-view so we can draw them
    predicted_positions = transform_points(np.concatenate(predicted_positions), data_ego["raster_from_world"])
    target_positions = transform_points(np.concatenate(target_positions), data_ego["raster_from_world"])
    
    # make sure ground truth and prediction have the same data size
    assert len(target_positions) == len(predicted_positions)
    
    # draw_trajectory(im_ego, predicted_positions, PREDICTED_POINTS_COLOR)
    draw_trajectory(im_ego, target_positions, TARGET_POINTS_COLOR)
    
    plt.rcParams['figure.figsize'] = 6, 6
    plt.imshow(im_ego[::-1])
    plt.show()
    
    draw_trajectory(im_ego, predicted_positions, PREDICTED_POINTS_COLOR)
    
    plt.rcParams['figure.figsize'] = 6, 6
    plt.imshow(im_ego[::-1])
    plt.show()
