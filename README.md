# TrAISformer

Pytorch implementation of TrAISformer---A generative transformer for AIS trajectory prediction (https://arxiv.org/abs/2109.03958).

The transformer part is adapted from: https://github.com/karpathy/minGPT

---
<p align="center">
  <img width="600" height="450" src="./figures/t18_3.png">
</p>


#### Requirements: 
See requirements.yml

### Datasets:

The data used in this paper are provided by the [Danish Maritime Authority (DMA)](https://dma.dk/safety-at-sea/navigational-information/ais-data). 
Please refer to [the paper](https://arxiv.org/abs/2109.03958) for the details of the pre-processing step. The code is available here: https://github.com/CIA-Oceanix/GeoTrackNet/blob/master/data/csv2pkl.py

A processed dataset can be found in `./data/ct_dma/`
(the format is `[lat, log, sog, cog, unix_timestamp, mmsi]`).

### Run

Run `trAISformer.py` to train and evaluate the model.
(Please note that the values given by the code are in km, while the values presented in the paper were converted to nautical mile.)

## Interpolation evaluation metrics

The interpolation model is evaluated on held-out gap points. The primary metric is the Haversine distance between predicted and true gap points, measured in kilometers. Threshold-based accuracy is reported as the percentage of gap points within a fixed distance.

For the port+land context checkpoint in `results_interpolation/ct_dma-pos-global_roi-interp-gap-1-6-past-1-10-future-1-10-edge-0.2-samples-4-1-data_size-80-80-10-24-embd_size-128-128-64-64-port-1-3-120-land-1-80-head-4-4-bs-128-lr-0.0006-seqlen-120/model.pt` the test set metrics are:

- Total gap points: `2409`
- Mean gap error: `1.543 km`
- RMSE gap error: `4.392 km`
- Median gap error: `1.287 km`
- Threshold accuracy:
  - `<= 0.5 km`: `9.17%`
  - `<= 1.0 km`: `33.91%`
  - `<= 2.0 km`: `82.11%`
  - `<= 5.0 km`: `99.38%`

### Larger-gap evaluation (10–30 point gap model)

A separate model checkpoint was evaluated for larger interpolation gaps using the saved port+land context checkpoint at `results_interpolation/ct_dma-pos-global_roi-interp-gap-10-30-past-5-40-future-5-40-edge-0.2-samples-4-1-data_size-80-80-10-24-embd_size-128-128-64-64-port-1-3-120-land-1-80-head-4-4-bs-128-lr-0.0006-seqlen-120/model_010.pt`.

The test set metrics for this larger-gap model are:

- Total gap points: `14086`
- Mean gap error: `1.643 km`
- RMSE gap error: `2.670 km`
- Median gap error: `1.423 km`
- Threshold accuracy:
  - `<= 0.5 km`: `7.78%`
  - `<= 1.0 km`: `29.49%`
  - `<= 2.0 km`: `72.97%`
  - `<= 5.0 km`: `98.64%`

This model was trained for gap lengths between 10 and 30 points; the evaluation set realized gap lengths from 10 to 15 points in this run.

## ONNX export and inference

### Exporting model to ONNX (interactive)

A script is available in `export_scripts/export_onnx.py`.

- Run from repo root:
  ```bash
  python export_scripts/export_onnx.py
  ```
- The script scans for `.pt` checkpoint files in the repository tree.
- If multiple checkpoints are found, it presents an interactive selection (arrow keys + Enter when `InquirerPy` is installed, fallback numeric selection otherwise).
- The selected checkpoint is converted to ONNX (`model.onnx` by default).

Optional arguments:
- `--checkpoint <path>`: explicitly set checkpoint
- `--onnx <path>`: target ONNX output path (default `model.onnx`)
- `--seq-len <N>`: sequence length for export (e.g. 18)

### Expected model input (ONNX)

- input name: `x`
- shape: `(B, T, 4)`
  - `B` batch size
  - `T` sequence length (e.g. 18, max depends on config)
  - features: `[lat_norm, lon_norm, sog_norm, cog_norm]` in [0,1)
- dtype: `float32`

### Model output (ONNX) and decoding

- output shape: `(B, T, full_size)` where `full_size` = `lat_size + lon_size + sog_size + cog_size` (e.g. 622)
- to decode:
  - `lat_logits = out[..., :lat_size]`
  - `lon_logits = out[..., lat_size:lat_size+lon_size]`
  - `sog_logits = out[..., ...]`
  - `cog_logits = out[..., -cog_size:]`
- take `argmax` along dimension -1 per block to get discrete bin index.
- fractional value conversion:
  - `lat_norm = (lat_idx + 0.5) / lat_size`
  - `lon_norm = (lon_idx + 0.5) / lon_size`
  - `sog_norm = (sog_idx + 0.5) / sog_size`
  - `cog_norm = (cog_idx + 0.5) / cog_size`
- coordinate conversion into real values:
  - `lat = lat_norm * (lat_max - lat_min) + lat_min`
  - `lon = lon_norm * (lon_max - lon_min) + lon_min`
  - `sog = sog_norm * sog_range` (depends on dataset scaling)
  - `cog = cog_norm * 360`

### Contents returned by datasets

`datasets.AISDataset` returns, for each sample:

1. `seq` (torch.float32, shape `(max_seqlen,4)`) with `[lat, lon, sog, cog]` normalized to [0,1)
2. `mask` (torch.float32, shape `(max_seqlen,)`, 0/1 padding mask)
3. `seqlen` (torch.int): length of valid sequence
4. `mmsi` (torch.int): vessel identifier (metadata only, not model input)
5. `time_start` (torch.int): start timestamp (metadata only)

### ONNX runtime inference example

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession('model.onnx')
input_data = np.random.rand(1, 18, 4).astype(np.float32)
output = sess.run(None, {'x': input_data})[0]
print(output.shape)  # expected (1, 18, full_size)
```

### License

See `LICENSE`, as this code was forked and modified later on. Since Git LFS was not enabled on that repo, I had to upload with preicos git info into a new repository.

