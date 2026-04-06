import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys

root = os.path.abspath(os.path.dirname(__file__))
repo_root = os.path.dirname(root)
sys.path.insert(0, os.path.join(repo_root, "src_interpolation"))

from config_trAISformer import Config
import datasets
import models
import trainers
import utils

cf = Config()
cf.min_gap_points = 10
cf.max_gap_points = 30
cf.min_past_points = 5
cf.max_past_points = 40
cf.min_future_points = 5
cf.max_future_points = 40
cf.edge_case_prob = 0.2
cf.batch_size = 32
cf.use_port_context = True
cf.use_land_context = True
cf.device = torch.device('cpu')
cf.savedir = os.path.join(repo_root, "results_interpolation", "ct_dma-pos-global_roi-interp-gap-10-30-past-5-40-future-5-40-edge-0.2-samples-4-1-data_size-80-80-10-24-embd_size-128-128-64-64-port-1-3-120-land-1-80-head-4-4-bs-128-lr-0.0006-seqlen-120")
cf.ckpt_path = os.path.join(cf.savedir, "model_010.pt")

print('Checkpoint path:', cf.ckpt_path)
if not os.path.exists(cf.ckpt_path):
    raise FileNotFoundError(cf.ckpt_path)

with open(os.path.join(cf.datadir, cf.testset_name), 'rb') as f:
    l_pred_errors = pickle.load(f)

moving_threshold = 0.05
phase_data = []
for vessel in l_pred_errors:
    try:
        moving_idx = np.where(vessel['traj'][:, 2] > moving_threshold)[0][0]
    except Exception:
        moving_idx = len(vessel['traj']) - 1
    vessel['traj'] = vessel['traj'][moving_idx:, :]
    if not np.isnan(vessel['traj']).any() and len(vessel['traj']) >= max(cf.min_seqlen, cf.min_past_points + cf.min_gap_points + cf.min_future_points):
        phase_data.append(vessel)

print('Test tracks in phase_data:', len(phase_data))

test_dataset = datasets.AISInterpolationDataset(
    phase_data,
    max_seqlen=cf.max_seqlen,
    min_past_points=cf.min_past_points,
    max_past_points=cf.max_past_points,
    min_future_points=cf.min_future_points,
    max_future_points=cf.max_future_points,
    min_gap_points=cf.min_gap_points,
    max_gap_points=cf.max_gap_points,
    edge_case_prob=cf.edge_case_prob,
    samples_per_track=cf.eval_samples_per_track,
    seed=cf.seed + 20000,
    config=cf,
)
print('Test dataset size:', len(test_dataset))

model = models.TrAISformerInterpolation(cf)
state = torch.load(cf.ckpt_path, map_location='cpu')
model.load_state_dict(state)
model = model.to(cf.device)
model.eval()

loader = DataLoader(test_dataset, batch_size=cf.batch_size, shuffle=False, num_workers=0)

thresholds = [0.5, 1.0, 2.0, 5.0]
counts_under = {thr: 0 for thr in thresholds}
all_errors = []

step_error_sum = np.zeros(cf.max_gap_points, dtype=np.float64)
step_counts = np.zeros(cf.max_gap_points, dtype=np.int64)

gap_len_sum = np.zeros(cf.max_gap_points + 1, dtype=np.float64)
gap_len_counts = np.zeros(cf.max_gap_points + 1, dtype=np.int64)

for batch in loader:
    seqs, token_types, valid_masks, target_masks, seqlens, past_lens, gap_lens, future_lens, mmsis, time_seqs, port_features, land_features = batch
    seqs = seqs.to(cf.device)
    token_types = token_types.to(cf.device)
    valid_masks = valid_masks.to(cf.device)
    target_masks = target_masks.to(cf.device)
    port_features = port_features.to(cf.device)
    land_features = land_features.to(cf.device)

    preds = trainers.predict_gap(
        model,
        seqs,
        token_types,
        valid_masks,
        port_context=port_features,
        land_context=land_features,
        sample=False,
        temperature=cf.temperature,
        top_k=cf.top_k,
    )

    input_coords = seqs[:, :, :2].clone()
    input_coords[..., 0] = input_coords[..., 0] * (cf.lat_max - cf.lat_min) + cf.lat_min
    input_coords[..., 1] = input_coords[..., 1] * (cf.lon_max - cf.lon_min) + cf.lon_min
    input_coords = input_coords * torch.pi / 180.0

    pred_coords = preds[:, :, :2].clone()
    pred_coords[..., 0] = pred_coords[..., 0] * (cf.lat_max - cf.lat_min) + cf.lat_min
    pred_coords[..., 1] = pred_coords[..., 1] * (cf.lon_max - cf.lon_min) + cf.lon_min
    pred_coords = pred_coords * torch.pi / 180.0

    d = utils.haversine(input_coords, pred_coords)
    d_np = d.detach().cpu().numpy()
    target_np = target_masks.detach().cpu().numpy()
    past_np = past_lens.numpy()
    gap_np = gap_lens.numpy()

    point_errors = d_np[target_np == 1].reshape(-1)
    all_errors.extend(point_errors.tolist())
    for thr in thresholds:
        counts_under[thr] += int((point_errors <= thr).sum())

    for batch_idx in range(d_np.shape[0]):
        start = int(past_np[batch_idx])
        gap_len = int(gap_np[batch_idx])
        gap_errors = d_np[batch_idx, start:start + gap_len]
        if gap_errors.size == 0:
            continue
        for step, err in enumerate(gap_errors, start=1):
            step_error_sum[step - 1] += err
            step_counts[step - 1] += 1
        gap_len_sum[gap_len] += gap_errors.mean()
        gap_len_counts[gap_len] += 1

all_errors = np.asarray(all_errors, dtype=np.float64)
mean_error = float(np.mean(all_errors)) if all_errors.size else float('nan')
rmse_error = float(np.sqrt(np.mean(all_errors ** 2))) if all_errors.size else float('nan')
median_error = float(np.median(all_errors)) if all_errors.size else float('nan')

print('Total gap points:', all_errors.size)
print('Mean error (km):', mean_error)
print('RMSE error (km):', rmse_error)
print('Median error (km):', median_error)
for thr in thresholds:
    pct = 100.0 * counts_under[thr] / max(all_errors.size, 1)
    print(f'Pct <= {thr} km: {pct:.2f}% ({counts_under[thr]}/{all_errors.size})')

print('\nMean error by gap step:')
for step in range(cf.max_gap_points):
    if step_counts[step] > 0:
        print(f' step {step + 1}: {step_error_sum[step] / step_counts[step]:.3f} km ({step_counts[step]} points)')

print('\nMean error by gap length:')
for gap_len in range(cf.min_gap_points, cf.max_gap_points + 1):
    if gap_len_counts[gap_len] > 0:
        print(f' gap_len {gap_len}: {gap_len_sum[gap_len] / gap_len_counts[gap_len]:.3f} km ({gap_len_counts[gap_len]} sequences)')
