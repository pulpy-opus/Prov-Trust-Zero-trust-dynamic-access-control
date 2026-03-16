# Prov-Trust: A Dynamic and Context-Aware Trust Evaluation Model for Zero Trust Architecture

This repository contains the core implementation of **Prov-Trust**, a provenance-graph-based dynamic trust evaluation system designed to detect Advanced Persistent Threat (APT) attacks under a Zero Trust Architecture (ZTA).

The system combines a Temporal Graph Attention Network (TGAT) for anomaly detection on system provenance graphs with a dynamic IP trust scoring model that continuously evaluates entity trustworthiness based on behavioral context.

---

## System Overview

```
Raw Audit Logs (CADETS E3)
        │
        ▼
 create_database.py       ← Parse logs → PostgreSQL
        │
        ▼
   embedding.py           ← Node feature extraction + graph vectorization
        │
        ▼
    train.py              ← Train TGAT model (GraphAttentionEmbedding + LinkPredictor)
        │
        ▼
     test.py              ← Graph reconstruction inference → per-edge anomaly loss
        │
        ▼
anomalous_queue_construction.py  ← IDF-based anomalous queue construction
        │
        ▼
  ip_trust_manager.py     ← Dynamic IP trust scoring (Z-score + EWMA + Mann-Kendall)
        │
        ▼
   evaluation.py          ← Precision / Recall / F1 / AUC
```

---

## Key Components

### 1. TGAT-based Anomaly Detection (`model.py`, `train.py`, `test.py`)
- **TimeEncode**: Cosine-based temporal encoding for edge timestamps.
- **GraphAttentionEmbedding**: Two-layer TransformerConv with temporal edge attributes.
- **LinkPredictor**: MLP classifier predicting edge relation types.
- Training uses days 2–4 of the CADETS E3 dataset; inference runs on days 3–7.
- Each edge is assigned a reconstruction loss; high-loss edges indicate anomalies.

### 2. Anomalous Queue Construction (`anomalous_queue_construction.py`)
- Computes per-node IDF scores across training time windows to filter common nodes.
- Incrementally builds anomalous queues by linking time windows that share rare suspicious nodes.
- Simultaneously extracts per-IP netflow behavior logs for trust evaluation.

### 3. Dynamic IP Trust Evaluation (`ip_trust_manager.py`)
- **`compute_anomaly_score`**: Z-score normalization + severity aggregation over edge losses.
- **`IPBehaviorQueue`**: Maintains a sliding window of trust values per IP; applies reward/decay based on activity.
- **`calculate_final_trust`**: Exponentially weighted rolling average with a cumulative penalty factor β.
- **`calculate_permission_threshold`**: Dynamically adjusts access thresholds θ_high / θ_low using Mann-Kendall trend test and linear regression slope.

---

## Dataset

This implementation is evaluated on the **DARPA Transparent Computing CADETS E3** dataset.

- Dataset: [DARPA TC CADETS E3](https://github.com/darpa-i2o/Transparent-Computing)
- Raw log files expected at: `config.py → raw_dir` (default: `/root/autodl-tmp/cadets_e3/`)
- Expected files:
  ```
  ta1-cadets-e3-official.json
  ta1-cadets-e3-official.json.1
  ta1-cadets-e3-official.json.2
  ta1-cadets-e3-official-1.json  ... (json.1 ~ json.4)
  ta1-cadets-e3-official-2.json
  ta1-cadets-e3-official-2.json.1
  ```

---

## Environment Setup

### Requirements
- Linux (tested on Ubuntu 20.04)
- CUDA 12.1
- Conda

### Install

```bash
conda env create -f environment.yml
conda activate prov-trust
```

### PostgreSQL Setup

Install PostgreSQL and create the database:

```sql
CREATE DATABASE tc_cadet_dataset_db;
```

Create the required tables:

```sql
CREATE TABLE netflow_node_table (uuid TEXT, hash TEXT, src_addr TEXT, src_port TEXT, dst_addr TEXT, dst_port TEXT);
CREATE TABLE subject_node_table (uuid TEXT, hash TEXT, exec TEXT);
CREATE TABLE file_node_table    (uuid TEXT, hash TEXT, path TEXT);
CREATE TABLE node2id            (hash TEXT, type TEXT, label TEXT, index_id INT);
CREATE TABLE event_table        (src_hash TEXT, src_label TEXT, rel_type TEXT, dst_hash TEXT, dst_label TEXT, timestamp_rec BIGINT);
```

Update database credentials in `config.py`:

```python
database = 'tc_cadet_dataset_db'
host     = 'localhost'
user     = 'postgres'
password = 'your_password'
port     = '5432'
```

---

## Usage

Run the pipeline in order:

```bash
# Step 1: Parse raw logs into PostgreSQL
python create_database.py

# Step 2: Extract node features and vectorize temporal graphs
python embedding.py

# Step 3: Train the TGAT model
python train.py

# Step 4: Run graph reconstruction inference (generates per-edge loss files)
python test.py

# Step 5: Build anomalous queues and extract IP behavior logs
python anomalous_queue_construction.py

# Step 6: Compute dynamic IP trust scores
python ip_trust_manager.py

# Step 7: Evaluate detection performance
python evaluation.py
```

All intermediate artifacts are saved under `./artifact/` (configurable in `config.py`).

---

## Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `time_window_size` | `60000000000 * 15` | Time window size (15 min in nanoseconds) |
| `node_embedding_dim` | `16` | Node feature hash dimension |
| `node_state_dim` | `100` | Node state dimension |
| `edge_dim` | `100` | Edge embedding dimension |
| `neighbor_size` | `20` | Neighborhood sampling size |
| `BATCH` | `1024` | Training/inference batch size |
| `epoch_num` | `50` | Training epochs |
| `lr` | `0.00005` | Adam learning rate |
| `beta_day6/7` | `100` | Anomaly score threshold for detection |

---

## Citation

If you use this code in your research, please cite:

```
@article{provtrust2025,
  title   = {Implementing A Dynamic and Context-Aware Trust Evaluation Model for Zero Trust Architecture},
  year    = {2025}
}
```

---

## Acknowledgements

Part of the TGAT implementation is adapted from:
- [pytorch_geometric TGN example](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py)
