# 3D Group Activity Generation

We provide the 3D group activity dataset, **M3Act3D**, as well as relevant supporting scripts for data visualization, **MDM+IFormer** baseline, and evaluation.

## Requirements

- Python 3.9
- FFMPEG

Install packages:
```bash
pip install -r requirements.txt
```

## Preparation

1. Prepare 3D dataset:

    - Download [M3Act3D](https://tally.so/r/wvLN60) dataset and put all *.h5 files under current directory (`gag/`).

2. Download model checkpoints: (TBD)

    - Download MDM+IFormer [checkpoint](https://github.com/cjerry1243/M3Act/tree/master/gag). (TBD)

    - Download Evaluation Model Checkpoint, [Composer3D](https://github.com/cjerry1243/M3Act/tree/master/gag). (TBD)


## M3Act3D Dataset

The h5 data contains simulations of all 6 group activities. The length of each simulation clip is 150 frames, in 30 FPS.

To load the motions given a h5file  and the clip index (idx), use the following code snippet:

```python
import h5py

with h5py.File(path_to_h5_file, "r") as h5:
    keys = list(h5.keys())
    idx = 0  # ID of simulation clips

    rot6d = h5[keys[idx]]["6d_rotations"][:]  # 6d rotation representation
    # shape: (150, num_people, num_joints, 6)

    quat = h5[keys[idx]]["quaternions"][:]  # quaternions representation
    # shape: (150, num_people, num_joints, 4)

    wpos = h5[keys[idx]]["w_positions"][:]  # world-space positions
    # shape: (150, num_people, num_joints, 3)

    rot6d = h5[keys[idx]]["bone_lengths"][:]  # bone lengths
    # shape: (num_people, num_bones)

    rot6d = h5[keys[idx]]["group_id"][()]  # group class (int)
    rot6d = h5[keys[idx]]["group_name"][()]  # group name (str)

    rot6d = h5[keys[idx]]["action_id"][:]  # action classes
    # shape: (150, num_people)

    ACTIONS = { 0: 'Idle',
                1: 'Walk',
                2: 'Text',
                3: 'Talk',
                4: 'Wave',
                5: 'Point',
                6: 'Dance',
                7: 'Run',
                8: 'Sit',
                9: 'Fight',
                10: 'Box',
                11: 'Salute',
                12: 'Handshake',
                }
```


### Stick Figure Visualization  

```bash
python skeleton_visualize.py
```

Resulting videos will be saved to `results/` folder.


### SMPL Visualization  

TBD

## Inference

TBD

## Training

TBD
