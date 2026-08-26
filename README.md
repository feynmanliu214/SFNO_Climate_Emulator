# SFNO_Climate_Emulator

Training and inference for **SFNO** (Spherical Fourier Neural Operator) emulators
of atmospheric and climate model output, built on
[NVIDIA Makani](https://github.com/NVIDIA/makani).

SFNO learns dynamics directly on the sphere using a spherical harmonic transform
in place of the planar FFT used by earlier Fourier neural operators, which keeps
long autoregressive rollouts stable and free of the polar artefacts that
equirectangular models accumulate. This repository packages the Makani training
framework together with the configuration and launcher layer used to train and
roll out emulators.

## Repository layout

| Path                  | Contents                                                              |
| --------------------- | --------------------------------------------------------------------- |
| `makani/`             | The Makani training framework (vendored, Apache-2.0 — see `NOTICE`).   |
| `configs/`            | Training configurations.                                              |
| `src/sfno_emulator/`  | Command-line launcher for training, inference, and ensemble runs.      |
| `requirements.txt`    | Runtime dependencies.                                                 |

## Installation

Requires Python >= 3.10 and a CUDA-capable PyTorch build. Install PyTorch first,
matched to your CUDA version, following the
[official instructions](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/feynmanliu214/SFNO_Climate_Emulator.git
cd SFNO_Climate_Emulator

python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

pip install -e ./makani        # training framework
pip install -r requirements.txt
pip install -e .               # launcher
```

Verify the install:

```bash
python -c "import makani; print(makani.__version__)"
sfno-emulator --version
```

## Configuration

A configuration file holds one or more named blocks; you select a block by name
at launch. `configs/example_sfno.yaml` documents the full key set — data and
statistics paths, the SFNO architecture, the loss, the optimizer and schedule,
and the dataloader.

Before training, point the data keys at your own packaged dataset:

```yaml
metadata_json_path: "/path/to/dataset/metadata/data.json"
train_data_path:    "/path/to/dataset/train"
valid_data_path:    "/path/to/dataset/valid"
exp_dir:            "/path/to/runs"
```

Makani reads HDF5 shards plus a metadata JSON describing the channel set, and
expects normalization statistics generated from the training split. The
`makani/data_process/` directory contains the tooling for building these.

## Training

```bash
sfno-emulator train \
    --config configs/example_sfno.yaml \
    --config-name example_sfno \
    --run-name 01
```

Multi-GPU runs launch under MPI, which is how Makani wires up its communicators:

```bash
sfno-emulator train \
    --config configs/example_sfno.yaml \
    --config-name example_sfno \
    --run-name 01 \
    --nproc 8 \
    --amp_mode bf16 \
    --batch_size 64
```

Checkpoints and logs are written under the `exp_dir` set in the config.

## Inference

```bash
sfno-emulator infer \
    --config configs/example_sfno.yaml \
    --config-name example_sfno \
    --checkpoint_path /path/to/runs/01/training_checkpoints/best_ckpt_mp0.tar
```

Ensemble forecasts use the same interface:

```bash
sfno-emulator ensemble \
    --config configs/example_sfno.yaml \
    --config-name example_sfno \
    --ensemble_size 16
```

## Passing Makani options through

The launcher owns only `--config`, `--config-name`, `--run-name`, `--nproc`, and
`--dry-run`. Every other argument is forwarded verbatim to the underlying Makani
entry point, so the full upstream option surface — model and spatial parallelism,
mixed precision, gradient checkpointing, multistep rollout, profiling — remains
available. `--dry-run` prints the assembled command without running it:

```bash
sfno-emulator train --config configs/example_sfno.yaml \
    --config-name example_sfno --h_parallel_size 4 --dry-run
```

See `makani/README.md` for the complete option reference.

## Citing

If you use this code, please cite Makani and the SFNO paper:

- **Makani** — NVIDIA, https://github.com/NVIDIA/makani
- **Bonev, B., Kurth, T., Hundt, C., Pathak, J., Baust, M., Kashinath, K., &
  Anandkumar, A. (2023).** *Spherical Fourier Neural Operators: Learning
  Stable Dynamics on the Sphere.* ICML 2023. arXiv:2306.03838

## License

Apache License 2.0 — see [`LICENSE`](LICENSE). Attribution for the vendored
Makani framework is in [`NOTICE`](NOTICE).
