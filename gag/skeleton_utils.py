import h5py
import numpy as np
from common.quaternion import *
from common.skeleton import Skeleton


ACTIONS = {0: 'Idle',
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
           12: 'Handshake'}


# the first 22 joints follow t2m skeleton, we use 25 joints
jointname_order = ["hips",
                   "left_hip", "right_hip",
                   "spine",
                   "left_knee", "right_knee",
                   "spine1",
                   "left_ankle", "right_ankle",
                   "spine2",
                   "left_toe_base", "right_toe_base",
                   "neck",
                   "left_shoulder", "right_shoulder",
                   "head",
                   "left_upperarm", "right_upperarm", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                   "head_end", "lefttoed_end", "righttoe_end",
                   "nose", "left_eye", "right_eye", "left_ear", "right_ear"]


NUM_JOINTS_USED = 25


KINEMATIC_FULLCHAIN = [ [0, 2, 5, 8, 11, 24], # right hip
                        [0, 1, 4, 7, 10, 23],
                        [0, 3, 6, 9, 12, 15, 22], # spine
                        [9, 14, 17, 19, 21], # right arm
                        [9, 13, 16, 18, 20],
                        [15, 25], [15, 26], [15, 27], [15, 28], [15, 29]
                      ]

KINEMATIC_CHAIN = [ [0, 2, 5, 8, 11, 24], # right hip
                    [0, 1, 4, 7, 10, 23],
                    [0, 3, 6, 9, 12, 15, 22], # spine
                    [9, 14, 17, 19, 21], # right arm
                    [9, 13, 16, 18, 20],
                  ]

COMMON_KINEMATIC_CHAIN = [[0, 2, 5, 8, 11], # right hip
                          [0, 1, 4, 7, 10],
                          [0, 3, 6, 9, 12, 15], # spine
                          [9, 14, 17, 19, 21], # right arm
                          [9, 13, 16, 18, 20],
                          ]

RAW_OFFSETS = np.array([[0, 0, 0],   # "hips"
                        [1, 0, 0],   # "left_hip"
                        [-1, 0, 0],  # "right_hip"
                        [0, 1, 0],   # "spine"
                        [0, -1, 0],  # "left_knee"
                        [0, -1, 0],  # "right_knee"
                        [0, 1, 0],   # "spine1"
                        [0, -1, 0],  # "left_ankle"
                        [0, -1, 0],  # "right_ankle"
                        [0, 1, 0],   # "spine2"
                        [0, 0, 1],   # "left_toe_base"
                        [0, 0, 1],   # "right_toe_base"
                        [0, 1, 0],   # "neck"
                        [1, 0, 0],   # "left_shoulder"
                        [-1, 0, 0],  # "right_shoulder"
                        [0, 1, 0],   # "head"
                        [1, 0, 0],   # "left_upperarm"
                        [-1, 0, 0],  # "right_upperarm"
                        [0, -1, 0],  # "left_elbow"
                        [0, -1, 0],  # "right_elbow"
                        [0, -1, 0],  # "left_wrist"
                        [0, -1, 0],  # "right_wrist"
                        [0, 1, 0],   # "head_end"
                        [0, 0, 1],   # "lefttoed_end"
                        [0, 0, 1]    # "righttoe_end"
                        ])


def get_bone_lengths(motion):
    # motion: (T, num_avatars, J, 3) or (T, J, 3)
    bone_lengths = np.zeros([RAW_OFFSETS.shape[0],])
    for chain in KINEMATIC_CHAIN:
        for parent, child in zip(chain[:-1], chain[1:]):
            bone_length = np.mean(np.sqrt(np.sum(np.square(motion[..., parent, :] - motion[..., child, :]), axis=-1)))
            bone_lengths[child] = bone_length
            # print(parent, child, bone_length)
    return bone_lengths


def get_skeleton_hierarchy():
    skeleton_hierarchy = {}
    child2parent = {}
    for chain in KINEMATIC_CHAIN:
        for parent, child in zip(chain[:-1], chain[1:]):
            if parent not in skeleton_hierarchy.keys():
                skeleton_hierarchy[parent] = [child]
            else:
                skeleton_hierarchy[parent].append(child)
            child2parent[child] = parent

    return skeleton_hierarchy, child2parent


def fit_wpos(gt):
    bone_lengths = get_bone_lengths(gt)[:, np.newaxis]
    skeleton = Skeleton(torch.tensor(RAW_OFFSETS*bone_lengths), kinematic_tree=KINEMATIC_CHAIN, hipID=0)
    quat = skeleton.inverse_kinematics_np(gt, [2, 1, 14, 13])
    pos = skeleton.forward_kinematics_np(quat, gt[..., 0, :], skel_joints=None)
    return pos


def get_motion_representations(wpos):
    bone_lengths = get_bone_lengths(wpos)
    skeleton = Skeleton(torch.tensor(RAW_OFFSETS*bone_lengths[:, np.newaxis]), kinematic_tree=KINEMATIC_CHAIN, hipID=0)
    quat = skeleton.inverse_kinematics_np(wpos, [2, 1, 14, 13])
    rot6d = quaternion_to_cont6d_np(quat)
    _wpos = skeleton.forward_kinematics_np(quat, wpos[..., 0, :], skel_joints=None)
    # bone_lengths.shape: (25,)
    # quat.shape: (150, 25, 4)
    # rot6d.shape: (150, 25, 6)
    # wpos.shape: (150, 25, 3)
    return quat, rot6d, _wpos, bone_lengths


def recover_wpos_from_root_and_rot6d(root_pos, rot6d):
    # rot_pos: (T, 3)
    # rot6d: (T, 25, 6)
    # wpos: (T, 25, 3)
    wpos = REFERENCE_SKELETON.forward_kinematics_cont6d_np(rot6d, root_pos)
    return wpos


with h5py.File("test.h5", "r") as h5:
    keys = list(h5.keys())

    REFERENCE_WPOS = h5[keys[0]]["w_positions"][:, 0, ...]
    PERSON_WIDTH = np.mean(np.sqrt(np.sum((REFERENCE_WPOS[:, 16] - REFERENCE_WPOS[:, 17]) ** 2, axis=-1)))
    REFERENCE_BONE_LENGTHS = get_bone_lengths(REFERENCE_WPOS)
    REFERENCE_SKELETON = Skeleton(torch.tensor(RAW_OFFSETS*REFERENCE_BONE_LENGTHS[:, np.newaxis]),
                                    kinematic_tree=KINEMATIC_CHAIN)
    

if __name__ == "__main__":

    pass