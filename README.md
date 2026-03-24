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

