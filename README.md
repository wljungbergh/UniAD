# Inference using UniAD
This repository is a fork of the [UniAD](https://github.com/OpenDriveLab/UniAD), used to showcase how to implment a model to be evaluated according to the [NeuroNCAP](https://github.com/atonderski/neuro-ncap) evaluation framework.

## Changes
This repository differs from the original UniAD repository in the following ways:
- Added a config file at `projects/configs/stage2_e2e/inference_e2e.py` to limit the operations applied to the input.
- Added inference functionality in the `inference` folder. This includes two files:
    - `runner.py` which wraps the original `UniAD` model to be able to run in inference mode (original can only be ran in training or validation/testing mode).
    - `server.py` which is a simple FastAPI server that opens endpoints to run inference using the model. The endpoints follow the NeuroNCAP API specification.
- Added a `Dockerfile` that was used to build the `.sif` file that the model can run in.

## How to use
1. Download the weights:
```bash
mkdir checkpoints
wget "https://github.com/OpenDriveLab/UniAD/releases/download/v1.0.1/uniad_base_e2e.pth" -P checkpoints
wget https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/motion_anchor_infos_mode6.pkl -P checkpoints
```

2. Build the `.sif` file:

```bash
docker build -t uniad:latest -f docker/Dockerfile .
singularity build uniad.sif docker-daemon://uniad:latest
```
Links:
- [How to install docker](https://docs.docker.com/get-docker/)
- [How to install singularity](https://docs.sylabs.io/guides/3.0/user-guide/index.html)

3. Follow the instructions in the [NeuroNCAP](https://github.com/wljungbergh/neuro-ncap/blob/release/docs/how-to-run.md) repository.


### [Original UniAD README.md](ORIGINAL_README.md)
