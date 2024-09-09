import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from textwrap import wrap
from skeleton_utils import *


def plot_motion(save_path, motion, kinematic_tree=KINEMATIC_CHAIN, title="", figsize=(5, 3), fps=30, radius=3):
    """
    Creates a 3D animation of a single motion sequence and saves it as a video file.

    Parameters:
    - save_path: str, path to save the animation video.
    - motion: numpy array, 3D array of shape (frames, joints, coordinates) representing the motion data.
    - kinematic_tree: list, kinematic chain defining the connections between joints.
    - title: str, optional, title for the animation plot.
    - figsize: tuple, optional, size of the figure (width, height).
    - fps: int, optional, frames per second for the animation.
    - radius: float, optional, defines the scale of the 3D plot's axes.
    """

    matplotlib.use('Agg')
    title = '\n'.join(wrap(title, 40))

    def init():
        ax.set_xlim3d([-radius/1.5, radius/1.5])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    colors = ["#4D84AA", "#5B9965", "#5B9965", "#61CEB9", "#61CEB9", "#34C1E2", "#80B79A", "#DD5A37", "#DD5A37", "#D69E00", "#D69E00"]  # GT color

    # motion *= 0.8
    MINS = motion.min(axis=0).min(axis=0)
    MAXS = motion.max(axis=0).max(axis=0)

    frame_number = motion.shape[0]

    # Normalize height and xz positions
    height_offset = MINS[1]
    motion[:, :, 1] -= height_offset
    trajec = motion[:, 0, [0, 2]]
    motion[..., 0] -= motion[:, 0:1, 0]
    motion[..., 2] -= motion[:, 0:1, 2]

    print("distance traveled:", np.sum((trajec[-1] - trajec[0])**2) ** 0.5)
    print("avatar height:", np.mean(np.max(motion[..., 1], axis=1) - np.min(motion[..., 1], axis=1)))

    def update(index):
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                          MAXS[2] - trajec[index, 1])
        # ax.scatter(data[index, :, 0], data[index, :, 1], data[index, :, 2], color='black', s=1)

        # ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
        #           trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
        #           color='black', linestyle='dashed')

        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            if i < 5:
                linewidth = 1.5
            else:
                linewidth = 0.5
            ax.plot3D(motion[index, chain, 0], motion[index, chain, 1], motion[index, chain, 2],
                      linewidth=linewidth, color=color)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)
    ani.save(save_path, fps=fps)
    plt.close()
    return


def plot_group_activity(save_path, motions, kinematic_tree=KINEMATIC_CHAIN, title="", figsize=(5, 3), fps=30, radius=5):
    """
    Creates a 3D animation of multiple motion sequences (group activity) and saves it as a video file.

    Parameters:
    - save_path: str, path to save the animation video.
    - motions: numpy array, 4D array of shape (frames, avatars, joints, coordinates) representing group motion data.
    - kinematic_tree: list, kinematic chain defining the connections between joints.
    - title: str, optional, title for the animation plot.
    - figsize: tuple, optional, size of the figure (width, height).
    - fps: int, optional, frames per second for the animation.
    - radius: float, optional, defines the scale of the 3D plot's axes.
    """

    matplotlib.use('Agg')
    title = '\n'.join(wrap(title, 40))

    def init():
        ax.set_xlim3d([-radius/1.5, radius/1.5])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    colors = ["#4D84AA", "#5B9965", "#5B9965", "#61CEB9", "#61CEB9", "#34C1E2", "#80B79A", "#DD5A37", "#DD5A37", "#D69E00", "#D69E00"]  # GT color

    group_MINS = motions.min(axis=0).min(axis=0).min(axis=0)
    group_MAXS = motions.max(axis=0).max(axis=0).max(axis=0)
    group_motion = np.mean(motions, axis=1)

    # MINS = motions.min(axis=0).min(axis=0)
    # MAXS = motions.max(axis=0).max(axis=0)

    frame_number = motions.shape[0]
    num_avatars = motions.shape[1]

    # Normalize height and xz positions
    height_offset = group_MINS[1]
    motions[..., 1] -= height_offset
    trajec = group_motion[:, 0, [0, 2]]
    motions[..., 0] -= group_motion[:, 0:1, 0][:, np.newaxis, :]
    motions[..., 2] -= group_motion[:, 0:1, 2][:, np.newaxis, :]

    def update(index):
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(group_MINS[0] - trajec[index, 0], group_MAXS[0] - trajec[index, 0], 0,
                     group_MINS[2] - trajec[index, 1], group_MAXS[2] - trajec[index, 1])
        # ax.scatter(data[index, :, 0], data[index, :, 1], data[index, :, 2], color='black', s=1)

        ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                  trajec[:index, 1] - trajec[index, 1], linewidth=1.2,
                  color='black', linestyle='dashed')

        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            if i < 5:
                linewidth = 1.5
            else:
                linewidth = 0.5
            for avatar_id in range(num_avatars):
                motion = motions[:, avatar_id]
                ax.plot3D(motion[index, chain, 0], motion[index, chain, 1], motion[index, chain, 2],
                          linewidth=linewidth, color=color)

                # motion_traj = motion[:, 0, [0, 2]] #+ group_motion[:, 0, [0, 2]]
                # ax.plot3D(motion_traj[:index, 0] - motion_traj[index, 0], np.zeros_like(motion_traj[:index, 0]),
                #           motion_traj[:index, 1] - motion_traj[index, 1], linewidth=0.25,
                #           color='orange', linestyle='dashed')

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)
    ani.save(save_path, fps=fps)
    plt.close()
    return


if __name__ == "__main__":
    with h5py.File("test.h5", "r") as h5:
        keys = list(h5.keys())
        n_clips = len(keys)
        
        np.random.seed(1234)
        ID = np.random.randint(0, n_clips)

        num_avatars = h5[keys[ID]]["w_positions"].shape[1]
        motion_data = h5[keys[ID]]["w_positions"][..., :NUM_JOINTS_USED, :3]
    
        print("---------------------------------")
        print("Total Simulation:", n_clips)
        print("Simulation ID:", ID, "Name:", keys[ID])
        print("Simulation Data:", h5[keys[ID]].keys())
        print("Shape of 6d rotations:", h5[keys[ID]]['6d_rotations'].shape)
        print("Shape of quaternions:", h5[keys[ID]]['quaternions'].shape)
        print("Shape of world positions:", h5[keys[ID]]['w_positions'].shape)
        print("Shape of bone lengths:", h5[keys[ID]]['bone_lengths'].shape)
        print("Group class:", h5[keys[ID]]['group_id'][()])
        print("Group name:", h5[keys[ID]]['group_name'][()])
        print("Shape of action classes:", h5[keys[ID]]['action_id'].shape)
        print("Num of People:", num_avatars)
        print("---------------------------------")

        print("Plot Stick Figure Motion.")
        os.makedirs("results", exist_ok=True)

        # ## Visualize with 25 joints
        # plot_group_activity("results/ID{}-{}.mp4".format(ID, keys[ID]), motion_data, fps=30, title="INDEX:{}-{}".format(ID, keys[ID]))
        
        ## Visualize with 22 joints
        plot_group_activity("results/ID{}-{}.mp4".format(ID, keys[ID]), motion_data[..., :22, :3], fps=30, title="INDEX:{}-{}".format(ID, keys[ID]), kinematic_tree=COMMON_KINEMATIC_CHAIN)

        # ## Forward Kinematics (With Average Bone Lengths)
        # _motion_data = np.zeros_like(motion_data)
        # for avatar_id in range(num_avatars):
        #     _motion_data[:, avatar_id] = recover_wpos_from_root_and_rot6d(motion_data[:, avatar_id, 0, :3], h5[keys[ID]]["6d_rotations"][:, avatar_id, :NUM_JOINTS_USED, :])
        # print(np.max(np.abs(motion_data - _motion_data)))

        # ## Inverse + Forward Kinematics
        # _motion_data = np.zeros_like(motion_data)
        # for avatar_id in range(num_avatars):
        #     _quat, _rot6d, _wpos, _bone_lengths = get_motion_representations(motion_data[:, avatar_id])
        #     _motion_data[:, avatar_id] = _wpos

        # print(np.max(np.abs(motion_data - _motion_data)))