# TrAISformer Interpolation Training

## Overview
This module trains a transformer-based model to interpolate missing segments (gaps) in vessel trajectories. The model is designed to predict the most likely path a vessel took during periods where AIS (Automatic Identification System) data is missing.

## How Training Works
1. **Data Preparation**: Vessel trajectories are loaded from preprocessed pickle files. Each trajectory is a sequence of points with features (latitude, longitude, speed over ground, course over ground, and optionally time).
2. **Window Sampling**: For each vessel, random windows are sampled consisting of:
   - **Past**: Observed points before the gap
   - **Gap**: The missing segment to be predicted
   - **Future**: Observed points after the gap
   The lengths of past, gap, and future are randomly sampled within configured bounds.
3. **Input Construction**: Each window is padded to a fixed length (`max_seqlen`). Each point is assigned a token type: past (1), gap (2), or future (3). Optionally, port context features are added for each point.
4. **Model**: The model is a bidirectional transformer (`TrAISformerInterpolation`) that takes the sequence of points (with masked gap) and predicts the missing gap segment.
5. **Training Loop**: The model is trained to minimize the cross-entropy loss over the gap segment, using the true values as targets. Training uses AdamW optimizer, learning rate scheduling, and gradient clipping.
6. **Validation & Checkpointing**: After each epoch, the model is evaluated on a validation set. The best model is saved based on validation loss. Plots of predicted vs. true trajectories are generated for qualitative assessment.

## What Happens to Trajectories
- Each trajectory is split into windows with a random gap.
- The model sees the past and future, but the gap is masked (hidden).
- The model learns to "infill" the gap, reconstructing the vessel's likely path.
- During evaluation, the predicted gap is compared to the true (held-out) gap for error analysis.

## Input Format
- **Input to the model**: A tensor of shape `(batch_size, max_seqlen, 4)` representing normalized [lat, lon, sog, cog] for each point, with the gap segment masked.
- **Token types**: A tensor indicating which points are past, gap, or future.
- **Port context (optional)**: Additional features encoding proximity to nearby ports.

## Output Format
- **Model output**: For each point in the gap, the model predicts a probability distribution over possible values for each feature (lat, lon, sog, cog).
- **During training**: The loss is computed only over the gap segment.
- **During inference**: The predicted gap is decoded and can be visualized or compared to ground truth.

## Example Workflow
1. Prepare data in the expected format (see `data/` directory).
2. Configure training parameters in `config_trAISformer.py`.
3. Run the training script (`trAISformer.py`).
4. Monitor training/validation loss and qualitative plots in `results_interpolation/`.
5. Use the trained model to interpolate gaps in new trajectories.

## References
- See `datasets.py`, `models.py`, and `trainers.py` for implementation details.
- The model is inspired by sequence modeling techniques for time series and NLP.

## Architecture Diagram

see ./architecture.mermaid

## Computational Complexity: Land and Port Context

The computational cost of adding land and port context features depends on the number of trajectory points, ports, and land polygons, as well as the use of spatial indexing. Below is a summary of the Big O complexity for different scenarios:

### Port Context Encoder
- For each trajectory point, distances to all filtered ports are computed and the k-nearest are selected.
- Let $N$ = number of trajectory points per batch, $P$ = number of ports, $k$ = nearest ports used (small).
- **Complexity:**
  - Per point: $O(P)$
  - All points: $O(NP)$

### Land Context Encoder
- For each trajectory point, the distance to the nearest land/coastline is computed.
- Let $L$ = number of land polygons/segments.
- **Naive:** $O(NL)$
- **With spatial index (e.g., R-tree):** $O(N \log L)$

### Combined Context
- **Total:** $O(NP + NL)$ (naive), $O(NP + N \log L)$ (with spatial index)

### Scenarios Table
| Scenario                        | Port Context | Land Context (naive) | Land Context (spatial index) | Combined                           |
| ------------------------------- | ------------ | -------------------- | ---------------------------- | ---------------------------------- |
| Small $P$, small $L$            | $O(N)$       | $O(N)$               | $O(N)$                       | $O(N)$                             |
| Large $P$, small $L$            | $O(NP)$      | $O(N)$               | $O(N)$                       | $O(NP)$                            |
| Small $P$, large $L$ (no index) | $O(N)$       | $O(NL)$              | $O(N \log L)$                | $O(NL)$                            |
| Large $P$, large $L$ (no index) | $O(NP)$      | $O(NL)$              | $O(N \log L)$                | $O(NP + NL)$ or $O(NP + N \log L)$ |

**Key Points:**
- The main cost is linear in the number of trajectory points and the number of ports/land polygons.
- Using spatial indexing for land context is highly recommended for scalability.
- For typical $k$ (nearest ports) and batch sizes, the cost is dominated by $P$ and $L$.

# Profiling

commit hash: eeac1e758d9bcb8c946b491a62da47bde723cfe5
```
TrAISformer  cmd /c "cd /d c:\Users\BBBS-AI-01\d\anomaly\TrAISformer && python scripts\profile_infer.py"
C:\ProgramData\anaconda3\Lib\site-packages\torch\nn\modules\transformer.py:392: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(
device cuda:0
model params 57424384
case past=1 gap=1 future=1 -> avg=5.301 ms (total 265.053 ms over 50)
case past=5 gap=5 future=5 -> avg=4.828 ms (total 241.402 ms over 50)
case past=10 gap=10 future=10 -> avg=5.806 ms (total 290.282 ms over 50)
case past=20 gap=20 future=20 -> avg=5.980 ms (total 298.989 ms over 50)
case past=40 gap=40 future=40 -> avg=5.548 ms (total 277.387 ms over 50)
```