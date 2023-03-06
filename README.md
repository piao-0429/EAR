# Emergent Action Representation (ICLR2023)

The codebase for ICLR2023 paper: Simple Emergent Action Representation from Multi-Task Policy Training.

Our project page is at: https://sites.google.com/view/emergent-action-representation/

## Environment Requirements

- Python 3.8
- PyTorch 1.8
- gym 0.24.0
- MuJoCo 2.1.0
- mujoco_py 2.1.2.14
- posix_ipc
- tensorboardX
- tabulate
- seaborn

## Training

```
# train EAR in HalfCheetah-Vel
python starter/ear_train.py --config config/train/halfcheetah-vel.json --id EAR_SAC --seed 0 --worker_nums 10 --eval_worker_nums 10
```
The training curves can be plotted following:

```
python torchrl/utils/plot_csv.py --id EXP_NAMES --env_name HalfCheetah-Vel --entry "Running_Average_Rewards" --add_tag POSTFIX_FOR_OUTPUT_FILES --seed SEEDS
```
You can replace "Running_Average_Rewards" with different entry to see different curve for different entry.
## Task Adaptation

```
# adapt to new tasks in HalfCheetah-Vel
python starter/ear_adapt.py --config config/adapt/halfcheetah-vel.json --id EAR_SAC --seed 0
```

## Task Interpolation & Composition
```
# interpolate two tasks in HalfCheetah-Vel
python starter/ear_interpolate.py --config config/interpolate/halfcheetah-vel.json --id EAR_SAC --seed 0

# compose two tasks in HalfCheetah-RunJump
python starter/ear_interpolate.py --config config/interpolate/halfcheetah-runjump.json --id EAR_SAC --seed 0
```
You should first initialize LTE_1 and LTE_2 in [ear_interpolate.py](starter/ear_interpolate.py) with the two task embeddings to be interpolated(or composed). 