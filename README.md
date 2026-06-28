# Crys_JEPA
This is the official implement of paper [Crys-JEPA: Accelerating Crystal Discovery via Embedding Screening and Generative Refinement](https://arxiv.org/pdf/2605.14759).

<img src="https://github.com/liun-online/Crys_JEPA/blob/main/model.png" width="800">

## Citation
```
@article{liu2026crysjepa,
  title         = {Crys-JEPA: Accelerating Crystal Discovery via Embedding Screening and Generative Refinement},
  author        = {Liu, Nian and Kazeev, Nikita and Dale, Stephen Gregory and Maevskiy, Artem and Zeng, Yuwei and Kubo, Ryoji and Huang, Pengru and Laurent, Thomas and LeCun, Yann and Novoselov, Kostya S. and Bresson, Xavier},
  journal       = {arXiv preprint},
  archivePrefix = {arXiv},
  eprint        = {2605.14759},
  year          = {2026}
}
```

## Python environment setup with uv
```
git clone https://github.com/liun-online/Crys_JEPA.git
cd Crys_JEPA/

uv sync
```

The project uses Python 3.12.
`pyproject.toml` is configured to use PyTorch 2.6.0 CUDA 12.4 wheels on Linux/Windows, and default
PyPI wheels on macOS.

Install `torch_scatter` according to your platform:

- Linux/Windows with CUDA 12.4:
```
uv pip install torch_scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

- macOS (CPU/MPS):
```
uv pip install torch_scatter
```

MatterSim is only needed for the relaxation step (`IV_relax.py`) and is optional:
```
uv sync --extra relaxation
```

On Windows, MatterSim 1.2.0 does not publish a prebuilt wheel, so `uv` must compile it from source.
Install Microsoft C++ Build Tools first, or run the relaxation step from Linux/WSL where MatterSim
provides a wheel.

## Download and Reproduce
Firstly, install [Hugging Face](https://huggingface.co/) and login
```
uv run hf auth login
```
Then, download the folders, i.e., `./data`, `./ref_dataset`, and `exp_logs.zip` by running the following commands:
```
hf download liun-online/Crys_JEPA --repo-type dataset --local-dir ./
mv ./ref_dataset ./eval/vsun/
```
The `exp_logs.zip` includes the checkpoints and intermediate data for reproducing the results in paper.

To reproduce V.S.U.N.
- MP-20: 44.9
- Alex-MP-20: 64.6
```
unzip ./exp_logs.zip
mv exp_logs logs
cd ./data
unzip ./exp_finetune.zip
mv exp_finetune finetune
cd ..

## MP-20
uv run python VII_final_gen.py
uv run python VIII_final_eval.py

## Alex-MP-20
uv run python VII_final_gen.py --dataset alex_mp_20
uv run python VIII_final_eval.py --dataset alex_mp_20

## [Optional] Run the commands below to prevent name clashes when starting a new training run
mv logs exp_logs
cd ./data
mv finetune exp_finetune
cd ..
```
Remarks
> 1. `python VII_final_gen.py` generates 10 batches of 1,000 crystals, stored in `./logs/finetune/mp_20/gen` and `./logs/finetune/alex_mp_20/gen`
> 2. Find results at `./logs/finetune/mp_20/eval/metrics_summary.log` and `./logs/finetune/alex_mp_20/eval/metrics_summary.log`

## Training Pipeline
### a. Train JEPA and base generative model
```
uv run python I_train_jepa.py
uv run python II_train_base.py
```

### b. Use base model to generate candidates, relax and screen
```
uv run python III_base_gen.py
uv sync --extra relaxation
uv run python IV_relax.py
uv run python V_screen.py
```

### c. Fine-tune the base model, and evaluate
```
uv run python VI_ft_base.py
uv run python VII_final_gen.py
uv run python VIII_final_eval.py
```
Remarks
> 1. Add `--dataset alex_mp_20` behind commands `II ~ VIII` to use Alex-MP-20 dataset.
> 2. Change hyper-parameters, e.g., ```uv run python II_train_base.py --conf_new training.batch_size=256 training.lr=0.0001```.
> 3. Use partial of GPUs, e.g., ```CUDA_VISIBLE_DEVICES=0,1 uv run python II_train_base.py```. Default: use all GPUs.
> 4. The evaluation code is mainly adapted from [MatterGen](https://github.com/microsoft/mattergen) and [FlowMM](https://github.com/facebookresearch/flowmm).

## 3DSC Superconductivity MVP

This repository now includes a supervised prototype for testing whether frozen Crys-JEPA crystal embeddings are useful for superconductivity prediction on 3DSC/3DSCMP.

### Goal

The MVP maps a CIF crystal structure to `C = (X, A, L)`, encodes it with a frozen Crys-JEPA-compatible encoder, and trains supervised MLP heads for:

- binary superconductivity classification, using `label_supra = 1` when `Tc > 0` and `0` when `Tc = 0`;
- critical-temperature regression in Kelvin;
- optional Tc uncertainty through a positive Softplus head.

The code is designed so the encoder can be swapped: set `model.crys_jepa_checkpoint` in `configs/default.yaml` to use a pretrained Crys-JEPA checkpoint, or leave it null to use `PlaceholderCrysJEPAEncoder` for pipeline tests.

### Dataset Layout

Place the 3DSC metadata and CIF files like this:

```text
data/raw/3DSC_MP.csv
data/raw/cifs/*.cif
```

The CSV must contain at least:

- `formula`
- `Tc`
- `cif_path`

Optional Materials Project or DFT columns can be kept in the CSV for later extensions. The current MVP ignores them unless new heads/features are added.

### Training

```bash
python scripts/train_mvp.py --config configs/default.yaml
```

The best checkpoint is saved to `checkpoints/best.pt` by default, selected by validation MAE on Tc.

### Evaluation

```bash
python scripts/evaluate_mvp.py --config configs/default.yaml --checkpoint checkpoints/best.pt --predictions-csv predictions.csv
```

Evaluation reports classification metrics, Tc MAE/RMSE, MAE on superconductors only, MAE on high-Tc materials above 77 K, and a binary confusion matrix.

### CIF Inference

```bash
python scripts/infer_cif.py --checkpoint checkpoints/best.pt --cif path/to/material.cif
```

Example output:

```text
Material: YBa2Cu3O7
P(superconductor): 0.8421
Predicted Tc: 86.30 K
Uncertainty: 7.40 K
```

### Main Files

- `src/datasets/threedsc_dataset.py`: 3DSC CSV/CIF dataset, labels, reproducible splits, and padding collate function.
- `src/models/crys_jepa_wrapper.py`: pretrained Crys-JEPA adapter plus placeholder encoder.
- `src/models/superconductivity_heads.py`: classifier, Tc regressor, and optional uncertainty head.
- `src/training/train.py`: training loop and best-checkpoint saving.
- `src/training/evaluate.py`: prediction collection and metrics.
- `src/infer.py`: single-CIF inference.

### Current Scientific Limits

- `Tc = 0` rows are used as negatives for the MVP, but they may mean "not known to superconduct" rather than experimentally confirmed non-superconductors.
- 3DSC structures can be approximate and may not capture pressure, doping, disorder, or synthesis conditions.
- The default MVP freezes Crys-JEPA and trains only supervised heads.
- DFT features such as `band_gap`, `energy_above_hull`, `fermi_energy`, and magnetization are reserved for later integration.
- Real discovery workflows still need stability, synthesizability, and energy-above-hull filtering.

### Planned Extensions

1. Partial Crys-JEPA fine-tuning.
2. Pressure and doping inputs.
3. DFT feature fusion.
4. Ensemble uncertainty.
5. Materials Project candidate screening.
6. Energy-above-hull filtering.
7. High-Tc-oriented objectives instead of plain classification.

### DFT Feature Ablation

The supervised MVP supports three input modes for a clean scientific ablation:

- `crys_jepa`: structure embedding only.
- `dft`: numerical DFT features only.
- `crys_jepa_dft`: late fusion of the structure embedding and a DFT MLP.

The default config now uses `crys_jepa_dft`. To run the three comparable variants on the same deterministic split:

```bash
python scripts/train_mvp.py --config configs/ablation/crys_jepa.yaml
python scripts/train_mvp.py --config configs/ablation/dft.yaml
python scripts/train_mvp.py --config configs/ablation/crys_jepa_dft.yaml
```

Or run the full ablation table in one command:

```bash
python scripts/run_dft_ablation.py --output ablation_results.csv
```

DFT columns are read from the 3DSC CSV, imputed with train-split medians, and standardized with train-split mean/std only. The scaler statistics are saved in the checkpoint config so validation, test, and later evaluation use the same transform.


### Partial Crys-JEPA Fine-Tuning

When a real pretrained Crys-JEPA checkpoint is available, the supervised DFT-JEPA model can compare frozen and partially fine-tuned encoders:

- `frozen`: pretrained Crys-JEPA kept frozen.
- `last1`: only the last transformer block plus final norm are trainable.
- `last2`: last two transformer blocks plus final norm are trainable with a smaller encoder LR.

Run all three variants with:

```bash
python scripts/run_partial_finetune.py --checkpoint path/to/crys_jepa_checkpoint.pt --matrix-scaler data/jepa/mean_std_scaler.pt --output partial_finetune_results.csv
```

The optimizer uses separate parameter groups: `training.learning_rate` for heads/fusion layers and `training.encoder_learning_rate` for trainable Crys-JEPA layers. The default fine-tuning configs live in `configs/finetune_dft_jepa/`.
