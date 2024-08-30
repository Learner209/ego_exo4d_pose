import json
import os

import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
from projectaria_tools.core import calibration
from utils.utils import (
	aria_landscape_to_portrait,
	cam_to_img,
	get_ego_aria_cam_name,
	get_ego_pose_takes_from_splits,
	get_interested_take,
	HAND_ORDER,
	pad_bbox_from_kpts,
	rand_bbox_from_kpts,
	hand_jnts_dist_angle_check,
	reproj_error_check,
	world_to_cam,
)
import copy
import torch

from egoego.utils.smpl_to_openpose.mapping import EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS, EGOEXO4D_EGOPOSE_HANDPOSE_MAPPINGS, USED_SMPLH_JOINT_NAMES
from egoego.utils.smpl_to_openpose.mapping import EGOEXO4D_BODYPOSE_TO_SMPLH_INDICES, EGOEXO4D_HANDPOSE_TO_SMPLH_w_HAND_INDICES, SMPLH_BODY_JOINTS, SMPLH_HAND_JOINTS
from egoego.utils.time_utils import linear_interpolate_missing_vals, spline_interpolate_missing_vals
from sklearn.cluster import DBSCAN
from egoego.utils.setup_logger import setup_logger
from egoego.simple_ik import simple_ik_solver_w_smplh
from egoego.smplx_utils import SMPLXUtils
from egoego.vis.utils import gen_full_body_vis
from egoego.utils.geom import align_to_reference_pose, rotate_at_frame_smplh
from egoego.config import default_cfg as CFG

from egoego.utils.aria.mps import AriaMPSService
from egoego.utils.egoexo.egoexo_utils import EgoExoUtils
from projectaria_tools.utils.vrs_to_mp4_utils import get_timestamp_from_mp4, convert_vrs_to_mp4

import matplotlib.pyplot as plt
from human_body_prior.body_model.body_model import BodyModel

VIZ_PLOTS = False

logger = setup_logger(__file__)

BODY_JOINTS = EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS
HAND_JOINTS = EGOEXO4D_EGOPOSE_HANDPOSE_MAPPINGS
SMPLH_JOINT_NAMES = USED_SMPLH_JOINT_NAMES


class EgoPoseDataPreparation:
	def __init__(self, args):
		self.args = args
		self.egoexo_root_path = args.ego4d_data_dir
		self.gt_output_dir = args.gt_output_dir
		self.portrait_view = args.portrait_view
		self.valid_kpts_num_thresh = args.valid_kpts_num_thresh
		self.bbox_padding = args.bbox_padding
		self.reproj_error_threshold = args.reproj_error_threshold
		self.hand_order = HAND_ORDER

		# Load calibration
		# self.calib = calibration.Calibration(self.ego4d_data_dir)

	def align_exported_anno_to_slam_traj(self, take_uid, egoexo_util_inst: EgoExoUtils, this_take_ego_cam_traj, this_take_ego_cam_intr, this_take_anno_3d, this_take_anno_3d_valid_flag):
		"""
		Aligns exported annotations to SLAM traj data for a given take.

		Parameters
		----------
		take_uid : str
			Unique identifier for the take.
		egoexo_util_inst : EgoExoUtils
			An instance of the EgoExoUtils class, which provides utility functions and data related to the take.
		this_take_ego_cam_traj : ndarray, shape (N, 7)
		this_take_ego_cam_intr : ndarray of shape (N, 3, 3)
		this_take_anno_3d : ndarray, shape (T, J, 3)
			The 3D annotations for the take, where T is the number of frames, J is the number of joints.
		this_take_anno_3d_valid_flag : ndarray, shape (T, J)
			A boolean array indicating the validity of each 3D annotation.
		reference_slam_traj : ndarray of shape (T', 7)

		Returns
		-------
		this_take_aligned_3d : ndarray, shape (T, J, 3)
		this_take_aligned_ego_cam_traj : ndarray, shape (T, 7)
		
		"""

		take_name = egoexo_util_inst.take_uid_to_take_names[take_uid]
		exported_mp4_file_path = egoexo_util_inst.get_exported_mp4_path_from_take_name(take_name=take_name, gt_output_dir=self.gt_output_dir)

		_, take_name, take, open_loop_traj_path, close_loop_traj_path, gopro_calibs_path, cam_pose_anno_path, vrs_path, vrs_noimagestreams_path, online_calib_json_path = egoexo_util_inst.get_take_metadata_from_take_uid(take_uid)
		aria_mps_serv = AriaMPSService(vrs_file_path=vrs_path,
								vrs_exported_mp4_path = exported_mp4_file_path,
								take=take,
								open_loop_traj_path=open_loop_traj_path,
								close_loop_traj_path=close_loop_traj_path,
								gopro_calibs_path=gopro_calibs_path,
								cam_pose_anno_path=cam_pose_anno_path,
								generalized_eye_gaze_path=None,
								calibrated_eye_gaze_path=None,
								online_calib_json_path=online_calib_json_path,
								wrist_and_palm_poses_path=None)

	
		exported_vrs_timestamps = aria_mps_serv.get_timestamps_from_exported_mp4()
		
		sampled_mps_close_traj_pose, T_world_imus_w_close_traj, tracking_timestamp_close_traj, utc_close_traj = aria_mps_serv.sample_close_loop_traj_from_timestamps(timestamps=exported_vrs_timestamps)
		sampled_mps_open_traj_pose, T_world_imus_w_open_traj, tracking_timestamp_open_traj, utc_open_traj = aria_mps_serv.sample_open_loop_traj_from_timestamps(timestamps=exported_vrs_timestamps)

		aligned_traj_trans, aligned_traj_rot_mat, aligned_traj_quat_wxyz, to_align2ref_rot_seq, move_to_ref_trans = align_to_reference_pose(this_take_ego_cam_traj[None,], sampled_mps_close_traj_pose[None,])
		aligned_traj_trans, aligned_traj_rot_mat, aligned_traj_quat_wxyz = aligned_traj_trans[0], aligned_traj_rot_mat[0], aligned_traj_quat_wxyz[0]
		this_take_aligned_ego_cam_traj = np.concatenate([aligned_traj_trans, aligned_traj_quat_wxyz], axis=1) # T x 7

		this_take_aligned_3d = (to_align2ref_rot_seq @ this_take_anno_3d.transpose(1,2)).transpose(1,2) # T x J x 3
		this_take_aligned_3d -= move_to_ref_trans[:,None] # T x J x 3

        # Make sure the aligned 3D annotations has their ankle heights on the z=0 plane.
		this_take_root_height_3d, has_left_ankle_flag, has_right_ankle_flag = self.extract_ankle_height(this_take_aligned_3d, this_take_anno_3d_valid_flag)
		this_take_floor_height_3d = this_take_root_height_3d - CFG.empirical_val.ankle_floor_height_offset
		this_take_aligned_3d[:,:,2] -= this_take_floor_height_3d[:,None] # T x J x 3

		return this_take_aligned_3d, this_take_aligned_ego_cam_traj
	
	def predict_pelvis_origin(self, this_take_anno_3d, this_take_anno_3d_valid_flag):
		"""
		Predict the pelvis origin for each frame based on the 3D annotations of body joints(in CoCo25 convention). The method 
		processes annotations to determine the probable pelvis origin using velocity clustering of joint movements 
		and height calculations based on ankle positions.

		Parameters
		----------
		this_take_anno_3d : ndarray of shape (T, J, D)
			The input 3D annotation array with shape (T, J, D), where T is the number of frames, 
			J is the number of joints, and D is the number of dimensions for each joint's coordinates (typically 3D).

		this_take_anno_3d_valid_flag : ndarray of shape (T, J)
			The validity flags for the input 3D annotations with shape (T, J), where T is the number of frames 
			and J is the number of joints. A true value in this array indicates that the corresponding joint's 
			annotation is valid and can be used for calculations.

		Returns
		-------
		pred_pelvis_origins: ndarray of shape (T, 3)
			An array with the predicted pelvis 3d coordinate directions having shape (T, 3), where T is the number of frames. There should be no NaN values in this output.

		Notes
		-----
		The function assumes that the joint annotations include specific indices for the left and right ankles. It handles frames where ankle data 
		may be missing and uses DBSCAN clustering to analyze joint velocity patterns to infer the most probable 
		pelvis position.

		The method is sensitive to the accuracy and completeness of the input data, particularly the validity of 
		ankle positions and the overall movement dynamics represented in the velocity of joint movements.

		Examples
		--------
		>>> this_take_anno_3d = np.random.rand(100, 42+17, 3)
		>>> this_take_anno_3d_valid_flag = np.random.rand(100, 42+17) > 0.3
		>>> predicted_origins = predict_pelvis_origin(this_take_anno_3d, this_take_anno_3d_valid_flag)
		>>> print(predicted_origins.shape)
		(100, 3)

		"""

		T, J, D = this_take_anno_3d.shape

		assert J == len(BODY_JOINTS) + len(HAND_JOINTS), f"The input 3D annotation should have {len(BODY_JOINTS) + len(HAND_JOINTS)} joints"
		assert D == 3, "The input 3D annotation should have 3D coordinates"

		this_take_body_anno_3d, this_take_hand_anno_3d = this_take_anno_3d[:len(BODY_JOINTS)], this_take_anno_3d[len(BODY_JOINTS):]
		this_take_body_anno_3d_valid_flag, this_take_hand_anno_3d_valid_flag = this_take_anno_3d_valid_flag[:len(BODY_JOINTS)], this_take_anno_3d_valid_flag[len(BODY_JOINTS):]
		# T x (body_jnts), T x (hand_jnts)

		this_take_prev_body_anno_3d = this_take_body_anno_3d.copy() # T x (body_jnts+hand_jnts) x 3
		this_take_prev_body_anno_3d = np.concatenate([np.zeros((1,J,3)), this_take_prev_body_anno_3d[:-1]], axis=0) # T x (body_jnts+hand_jnts) x 3
		this_take_anno_3d_vel = this_take_body_anno_3d - this_take_prev_body_anno_3d # T x (body_jnts+hand_jnts) x 3
		# Assuming the first frame is consistent with the second frame.
		this_take_anno_3d_vel[0] = this_take_anno_3d_vel[1] # T x (body_jnts+hand_jnts) x 3 

		# Skip this body if left-ankle and right-ankle are both None
		left_ankle_idx = BODY_JOINTS.index("left-ankle")
		right_ankle_idx = BODY_JOINTS.index("right-ankle")

		miss_left_ankle_flag = (~this_take_body_anno_3d_valid_flag)[:,left_ankle_idx]
		miss_right_ankle_flag = (~this_take_body_anno_3d_valid_flag)[:,right_ankle_idx]

		assert np.any(np.logical_and(miss_left_ankle_flag, miss_right_ankle_flag)), "Both left ankle and right ankle are missing, skip this body"
		assert np.any(np.isnan(this_take_body_anno_3d_valid_flag.mean(axis=1))), f"{sum(np.isnan(this_take_anno_3d_valid_flag.mean(axis=1)))} joints are missing, skip this body"

		tmp_body_anno_3d_w_left_ankle = this_take_body_anno_3d.copy()
		tmp_body_anno_3d_w_right_ankle = this_take_body_anno_3d.copy()
		# A little hack since both the left and right ankle could not be missing at the same time.
		tmp_body_anno_3d_w_left_ankle[miss_left_ankle_flag,left_ankle_idx] = tmp_body_anno_3d_w_right_ankle[miss_left_ankle_flag,left_ankle_idx]
		tmp_body_anno_3d_w_right_ankle[miss_right_ankle_flag,right_ankle_idx] = tmp_body_anno_3d_w_left_ankle[miss_right_ankle_flag,right_ankle_idx]
		this_take_root_height_3d = np.mean(np.concatenate([tmp_body_anno_3d_w_left_ankle[:,left_ankle_idx],tmp_body_anno_3d_w_right_ankle[:,right_ankle_idx]]), axis=1) # T


		pred_pelvis_origin_valid = []

		for frame_ind in tqdm(range(T), total=T, desc="Predicting pelvis origin", ascii=' >='):
			# ! Perform DBSCAN clustering on the num_of_jnts vels between the current frame and the prev frame to `find the most likely global orient`
	
			in_cluster_inds = this_take_body_anno_3d_valid_flag[frame_ind] # (body_jnts+hand_jnts)
			in_anno_3d = this_take_body_anno_3d[frame_ind][in_cluster_inds] # J' x 3
			in_anno_3d_vel = this_take_anno_3d_vel[frame_ind][in_cluster_inds] # J' x 3
			in_anno_root_heights = this_take_root_height_3d[frame_ind][in_cluster_inds] # J'

			# NOTE: this clustering assumes the `pelvis` facing direction is aligned to human-body-movement to perform balanced and prioceptive actions. 
			cluster_vels = []
			cluster_sizes = []

			# cluster foot heights and find one with smallest median
			clustering = DBSCAN(eps=0.005, min_samples=3).fit(in_anno_3d_vel) # J' x 3
			all_labels = np.unique(clustering.labels_)
			all_static_inds = np.arange(sum(in_cluster_inds))

			min_median = min_root_median = float('inf')
			for cur_label in all_labels:
				cur_clust = in_anno_3d_vel[clustering.labels_ == cur_label]
				cur_clust_inds = np.unique(all_static_inds[clustering.labels_ == cur_label]) # inds in the original sequence that correspond to this cluster
				if VIZ_PLOTS:
					plt.scatter(cur_clust, np.zeros_like(cur_clust), label='foot %d' % (cur_label))
				# get median foot height and use this as height
				cur_median = np.median(cur_clust)
				# cur_mean = np.mean(cur_clust)
				cluster_vels.append(cur_median)
				cluster_sizes.append(cur_clust.shape[0])

				# update min info
				if cur_median < min_median:
					min_median = cur_median

			cluster_size_acc = np.sum(cluster_sizes)
			cluster_weight_acc = np.zeros(1,3)
			for cluster_ind, (cluster_size, cluster_vel) in enumerate(zip(cluster_sizes, cluster_vels)):
				cluster_weight_acc += cluster_size * cluster_vel
			weighted_cluster_vel = cluster_weight_acc / cluster_size_acc
			pred_pelvis_origin_valid.append(weighted_cluster_vel)

		pred_pelvis_origin_valid = np.stack(pred_pelvis_origin_valid, axis=0) # T x 3

		return pred_pelvis_origin_valid
	
	def extract_ankle_height(self, this_take_anno_3d, this_take_anno_3d_valid_flag):
		"""
		Extracts the average height of the left and right ankles for each frame in a dataset.

		Parameters
		----------
		this_take_anno_3d : np.ndarray
			3D annotations for each joint per timestep. Shape (T, J, D) where T is the number of timesteps, J is the number of joints, and D is the dimension of the coordinates (3 for x, y, z).
		this_take_anno_3d_valid_flag : np.ndarray
			Validity flags for the annotations, indicating whether a joint's data is valid (1) or missing (0). Shape matches `this_take_anno_3d` in the first two dimensions (T, J).

		Returns
		-------
		this_take_root_height_3d : np.ndarray
			The average z-coordinate (height) of the left and right ankles for each timestep. Shape (T,).
		has_left_ankle_flag : np.ndarray
			Boolean array indicating presence of left ankle data per timestep. Shape (T,).
		has_right_ankle_flag : np.ndarray
			Boolean array indicating presence of right ankle data per timestep. Shape (T,).

		Raises
		------
		AssertionError
			If both left and right ankles are missing in any timestep or if any required joint data is missing entirely.

		"""

		T, J, D = this_take_anno_3d.shape

		assert J == len(BODY_JOINTS) + len(HAND_JOINTS), f"The input 3D annotation should have {len(BODY_JOINTS) + len(HAND_JOINTS)} joints"
		assert D == 3, "The input 3D annotation should have 3D coordinates"

		this_take_body_anno_3d, this_take_hand_anno_3d = this_take_anno_3d[:len(BODY_JOINTS)], this_take_anno_3d[len(BODY_JOINTS):]
		this_take_body_anno_3d_valid_flag, this_take_hand_anno_3d_valid_flag = this_take_anno_3d_valid_flag[:len(BODY_JOINTS)], this_take_anno_3d_valid_flag[len(BODY_JOINTS):]
		# T x (body_jnts), T x (hand_jnts)

		left_ankle_idx = BODY_JOINTS.index("left-ankle")
		right_ankle_idx = BODY_JOINTS.index("right-ankle")

		miss_left_ankle_flag = (~this_take_body_anno_3d_valid_flag)[:,left_ankle_idx]
		miss_right_ankle_flag = (~this_take_body_anno_3d_valid_flag)[:,right_ankle_idx]

		assert np.any(np.logical_and(miss_left_ankle_flag, miss_right_ankle_flag)), "Both left ankle and right ankle are missing, skip this body"
		assert np.any(np.isnan(this_take_body_anno_3d_valid_flag.mean(axis=1))), f"{sum(np.isnan(this_take_anno_3d_valid_flag.mean(axis=1)))} joints are missing, skip this body"

		tmp_body_anno_3d_w_left_ankle = this_take_body_anno_3d.copy()
		tmp_body_anno_3d_w_right_ankle = this_take_body_anno_3d.copy()
		# A little hack since both the left and right ankle could not be missing at the same time.
		tmp_body_anno_3d_w_left_ankle[miss_left_ankle_flag,left_ankle_idx] = tmp_body_anno_3d_w_right_ankle[miss_left_ankle_flag,left_ankle_idx]
		tmp_body_anno_3d_w_right_ankle[miss_right_ankle_flag,right_ankle_idx] = tmp_body_anno_3d_w_left_ankle[miss_right_ankle_flag,right_ankle_idx]
		
		this_take_root_height_3d = np.mean(np.concatenate([tmp_body_anno_3d_w_left_ankle[:,left_ankle_idx],tmp_body_anno_3d_w_right_ankle[:,right_ankle_idx]]), axis=1) # T
		has_left_ankle_flag = ~miss_left_ankle_flag
		has_right_ankle_flag = ~miss_left_ankle_flag
		return this_take_root_height_3d, has_left_ankle_flag, has_right_ankle_flag



	def predict_hip_trans(self, this_take_anno_3d, this_take_anno_3d_valid_flag):
		"""
		Predict the average translation vector for the hip joints using 3D annotations of left and right hips,
		interpolating missing values where necessary.

		Parameters
		----------
		this_take_anno_3d : np.ndarray
			A 3D numpy array of shape (T, J, D) containing the 3D joint coordinates,
			where T is the number of frames, J is the number of joints, and D is the dimensionality (always 3).
		this_take_anno_3d_valid_flag : np.ndarray
			A 2D boolean numpy array of shape (T, J) indicating the validity of each joint annotation per frame.

		Returns
		-------
		np.ndarray
			A numpy array of shape (T, 3), representing the interpolated translation vectors for the hip
			across T frames.

		"""
		T, J, D = this_take_anno_3d.shape

		assert J == len(BODY_JOINTS) + len(HAND_JOINTS), f"The input 3D annotation should have {len(BODY_JOINTS) + len(HAND_JOINTS)} joints"
		assert D == 3, "The input 3D annotation should have 3D coordinates"

		this_take_body_anno_3d, this_take_hand_anno_3d = this_take_anno_3d[:len(BODY_JOINTS)], this_take_anno_3d[len(BODY_JOINTS):]
		this_take_body_anno_3d_valid_flag, this_take_hand_anno_3d_valid_flag = this_take_anno_3d_valid_flag[:len(BODY_JOINTS)], this_take_anno_3d_valid_flag[len(BODY_JOINTS):]
		# T x (body_jnts), T x (hand_jnts)

		left_hip_ind = BODY_JOINTS.index("left_hip")
		right_hip_ind = BODY_JOINTS.index("right_hip")

		# TODO; consider scenario where left ankle and right ankle doesn't has valid annotations.
		# TODO: the current impl only uses mean of left ankle and right ankle as pesudo transl, and gloal_orient is set to all zeros.
		this_take_trans_left_hip = this_take_body_anno_3d.copy()[:,left_hip_ind] # T x 3
		this_take_trans_right_hip = this_take_body_anno_3d.copy()[:,right_hip_ind] # T x 3

		miss_left_hip_flag = np.isnan(np.mean(this_take_trans_left_hip,axis=1)) # T
		miss_right_hip_flag = np.isnan(np.mean(this_take_trans_right_hip,axis=1)) # T
		miss_both_hips_flag = np.logical_and(miss_left_hip_flag, miss_right_hip_flag) # T

		this_take_trans_left_hip[miss_left_hip_flag] = this_take_trans_right_hip[miss_left_hip_flag]
		this_take_trans_right_hip[miss_right_hip_flag] = this_take_trans_left_hip[miss_right_hip_flag]

		this_take_trans_hip = np.mean(np.stack([this_take_trans_left_hip, this_take_trans_right_hip]),axis=0) # T x 3
		this_take_trans_hip[miss_both_hips_flag] = np.nan # T x 3

		this_take_trans_hip = linear_interpolate_missing_vals(this_take_trans_hip)

		# TODO: the pesudo transl may have conflicts with the optimization process, so detract a small value.
		this_take_trans_hip[:,2] += 0.05 # T x 3
		return this_take_trans_hip

	def generate_smpl_converted_anno(self, this_take_anno_3d, this_take_anno_3d_valid_flag, seq_name):
		"""
		Converts 3D annotations to the SMPL-H format, computes transformations, and visualizes the results.

		Parameters
		----------
		this_take_anno_3d : ndarray
			The 3D annotations for the take with shape (T, J, 3), where T is the number of frames, J is the number of joints.
		this_take_anno_3d_valid_flag : ndarray
			A boolean array indicating the validity of each 3D annotation with shape (T, J).
		seq_name : str
			The sequence name used for visualization and file naming.

		Returns
		-------
		tuple
			A tuple containing:
			- ndarray: Converted SMPL-H 3D annotations (T, 52, 3).
			- ndarray: SMPL-H vertex coordinates (T, 6890, 3).
			- ndarray: SMPL-H face indices (T, 13776, 3).

		Raises
		------
		AssertionError
			If the input joint count does not match expected counts or the annotation is not in 3D format.

		Notes
		-----
		This function uses an IK solver and the SMPL-H model to align and convert the input 3D annotations. Visualization
		of the results is optionally provided based on configuration settings.
		"""
		T, J, D = this_take_anno_3d.shape

		assert J == len(BODY_JOINTS) + len(HAND_JOINTS), f"The input 3D annotation should have {len(BODY_JOINTS) + len(HAND_JOINTS)} joints"
		assert D == 3, "The input 3D annotation should have 3D coordinates"

		this_take_body_anno_3d, this_take_hand_anno_3d = this_take_anno_3d[:len(BODY_JOINTS)], this_take_anno_3d[len(BODY_JOINTS):]
		this_take_body_anno_3d_valid_flag, this_take_hand_anno_3d_valid_flag = this_take_anno_3d_valid_flag[:len(BODY_JOINTS)], this_take_anno_3d_valid_flag[len(BODY_JOINTS):]
		# T x (body_jnts), T x (hand_jnts)

		this_take_hip_trans = self.predict_hip_trans(this_take_anno_3d, this_take_anno_3d_valid_flag) # T x 3
		this_take_pelvis_origin = self.predict_pelvis_origin(this_take_anno_3d, this_take_anno_3d_valid_flag) # T x 3

		this_take_smplh_anno_3d, this_take_smplh_anno_3d_valid_flag = self.convert_to_smplh_convention(this_take_anno_3d, this_take_anno_3d_valid_flag)
		# T x 52 x 3, T x 52

		if VIZ_PLOTS:
			fig = plt.figure()
			ax = fig.add_subplot(projection='3d')
			RADIUS = 1.0
			xroot, yroot, zroot = this_take_hip_trans[0,
												0], this_take_hip_trans[0, 1], this_take_hip_trans[0, 2]
			ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
			ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
			ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
			print(f"the x-mean is {np.mean(this_take_hip_trans[:,0])}, the y-mean is {np.mean(this_take_hip_trans[:,1])}, the z-mean is{np.mean(this_take_hip_trans[:,2])}")

			ax.scatter(this_take_hip_trans[:, 0], this_take_hip_trans[:, 1],
					this_take_hip_trans[:, 2], c='b', marker='x')
			# ax.scatter(opt_joints[:, 0], opt_joints[:, 1],
			#            opt_joints[:, 2], c='r', marker='o')
			ax.grid(True)
			ax.set_xlabel('X')
			ax.set_ylabel('Y')
			ax.set_zlabel('Z')

			plt.show()

		smplx_utils = SMPLXUtils(CFG.smplh.smplh_root_path)

		opt_this_take_smplh_anno_3d = []  # jnts
		opt_this_take_smplh_verts = []
		opt_this_take_smplh_faces = []

		for frame_ind in tqdm(range(T), total=T, desc="Geenrating smpl converted annotation using simple ik sovler", ascii=' >='):
			
			this_frame_smplh_anno_3d_valid_flag = this_take_smplh_anno_3d_valid_flag[frame_ind] # 52
			this_frame_smplh_anno_3d = this_take_smplh_anno_3d[frame_ind] # 52 x 3
			this_frame_hip_trans = this_take_hip_trans[frame_ind] # 3
			this_frame_pelvis_origin = this_take_pelvis_origin[frame_ind] # 3

			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

			this_frame_hip_trans = torch.from_numpy(this_frame_hip_trans).to(device)
			this_frame_pelvis_origin = torch.from_numpy(this_frame_pelvis_origin).to(device)

			opt_local_aa_rep = simple_ik_solver_w_smplh(root_smplh_path = "/mnt/homes/minghao/robotflow/egoego/assets/smpl_based_model/smpl+h/male/model.npz", 
									target=torch.from_numpy(this_frame_smplh_anno_3d).to(device),
									target_mask=torch.from_numpy(this_frame_smplh_anno_3d_valid_flag).to(device),
									transl=this_frame_hip_trans,
									global_orient=this_frame_pelvis_origin,
									device=device) # 52 x 3

			this_frame_hip_trans = this_frame_hip_trans[None, None] # BS(1) x T x 3
			opt_local_aa_rep  = opt_local_aa_rep[None, None] # BS(1) x T x 52 x 3
			smplh_betas = torch.zeros(1, 16)
			smplh_betas = smplh_betas.unsqueeze(0) # BS(1) x 16

			opt_pred_smplh_jnts, opt_pred_smplh_verts, opt_pred_smplh_faces = smplx_utils.run_smpl_model(this_frame_hip_trans, opt_local_aa_rep, smplh_betas, smplx_utils.bm_dict[CFG.smplh.smplh_model])
			# B(1) x T(1) x 52 x 3, B(1) x T(1) x 6890 x 3, 13776 x 3

			dest_mesh_vis_folder = CFG.io.save_mesh_vis_folder
			os.makedirs(dest_mesh_vis_folder, exist_ok=True)
			gen_full_body_vis(opt_pred_smplh_verts[0], opt_pred_smplh_faces, dest_mesh_vis_folder, seq_name, vis_gt=False)

			opt_this_take_smplh_anno_3d.append(opt_pred_smplh_jnts[0,0].detach().cpu().numpy())
			opt_pred_smplh_verts.append(opt_pred_smplh_jnts[0,0].detach().cpu().numpy())
			opt_pred_smplh_faces.append(opt_pred_smplh_faces.detach().cpu().numpy())

			if VIZ_PLOTS:
				fig = plt.figure()
				ax = fig.add_subplot(projection='3d')
				RADIUS = 1.0
				opt_pred_smplh_jnts_vis = opt_pred_smplh_jnts[0].detach().cpu().numpy()
				xroot, yroot, zroot = opt_pred_smplh_jnts_vis[0,
													0], opt_pred_smplh_jnts_vis[0, 1], opt_pred_smplh_jnts_vis[0, 2]
				ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
				ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
				ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

				ax.scatter(opt_pred_smplh_jnts_vis[:, 0], opt_pred_smplh_jnts_vis[:, 1],
						opt_pred_smplh_jnts_vis[:, 2], c='b', marker='x')
				# ax.scatter(opt_joints[:, 0], opt_joints[:, 1],
				#            opt_joints[:, 2], c='r', marker='o')
				ax.grid(True)
				ax.set_xlabel('X')
				ax.set_ylabel('Y')
				ax.set_zlabel('Z')

				plt.show()
			
		opt_this_take_smplh_anno_3d = np.stack(opt_this_take_smplh_anno_3d, axis=0) # T x 52 x 3
		opt_this_take_smplh_verts = np.stack(opt_this_take_smplh_verts, axis=0) # T x 6890 x 3
		opt_this_take_smplh_faces = np.stack(opt_this_take_smplh_faces, axis=0) #  T x 13776 x 3

		return opt_this_take_smplh_anno_3d, opt_this_take_smplh_verts, opt_this_take_smplh_faces
			
	def convert_to_smplh_convention(self, this_take_anno_3d, this_take_anno_3d_valid_flag):
		"""
		Converts 3D joint annotations into the SMPL-H convention, which is suitable for human body models that
		include hand and body joints. This function adjusts the annotations to align with the joint indices used in
		the SMPL-H model.

		Parameters
		----------
		this_take_anno_3d : ndarray
			The input 3D annotation array with shape (T, J, D), where T is the number of frames,
			J is the number of joints (body and hand joints combined), and D is the dimensionality (typically 3D).

		this_take_anno_3d_valid_flag : ndarray
			A boolean array with shape (T, J) indicating the validity of each joint's annotation per frame.

		Returns
		-------
		tuple
			A tuple containing two ndarrays:
			- The first ndarray is the converted 3D annotations with shape (T, K, D), where K is the number of SMPL-H joints.
			- The second ndarray is a boolean array with shape (T, K) representing the validity of each converted joint's annotation.

		Examples
		--------
		>>> this_take_anno_3d = np.random.rand(100, 42+17, 3)
		>>> this_take_anno_3d_valid_flag = np.random.rand(100, 42+17) > 0.3
		>>> smplh_anno, smplh_valid_flags = convert_to_smplh_convention(this_take_anno_3d, this_take_anno_3d_valid_flag)
		>>> print(smplh_anno.shape, smplh_valid_flags.shape)
		(100, 52, 3), (100, 52)
		
		"""
		T, J, D = this_take_anno_3d.shape

		assert J == len(BODY_JOINTS) + len(HAND_JOINTS), f"The input 3D annotation should have {len(BODY_JOINTS) + len(HAND_JOINTS)} joints"
		assert D == 3, "The input 3D annotation should have 3D coordinates"

		this_take_body_anno_3d, this_take_hand_anno_3d = this_take_anno_3d[:len(BODY_JOINTS)], this_take_anno_3d[len(BODY_JOINTS):]
		this_take_body_anno_3d_valid_flag, this_take_hand_anno_3d_valid_flag = this_take_anno_3d_valid_flag[:len(BODY_JOINTS)], this_take_anno_3d_valid_flag[len(BODY_JOINTS):]
		# T x (body_jnts), T x (hand_jnts)

		this_take_smplh_anno_3d = np.full((T, len(SMPLH_JOINT_NAMES), 3), np.nan)
		this_take_smplh_body_anno_3d, this_take_smplh_hand_anno_3d = this_take_smplh_anno_3d[:,:len(SMPLH_BODY_JOINTS)], this_take_smplh_anno_3d[:,len(SMPLH_BODY_JOINTS):]

		body_pose_valid_flag = np.where(np.asarray(EGOEXO4D_BODYPOSE_TO_SMPLH_INDICES) != -1)[0]
		hand_pose_valid_flag = np.where(np.asarray(EGOEXO4D_HANDPOSE_TO_SMPLH_w_HAND_INDICES) != -1)[0]

		this_take_smplh_body_anno_3d[body_pose_valid_flag] = this_take_body_anno_3d[EGOEXO4D_BODYPOSE_TO_SMPLH_INDICES[body_pose_valid_flag]]
		this_take_smplh_hand_anno_3d[hand_pose_valid_flag] = this_take_hand_anno_3d[EGOEXO4D_HANDPOSE_TO_SMPLH_w_HAND_INDICES[hand_pose_valid_flag]]
		 
		this_take_smplh_body_anno_3d_valid_flag = np.isnan(np.mean(this_take_smplh_body_anno_3d, axis=2))
		this_take_smplh_hand_anno_3d_valid_flag = np.isnan(np.mean(this_take_smplh_hand_anno_3d, axis=2))

		this_take_smplh_anno_3d_valid_flag = np.concatenate([this_take_smplh_body_anno_3d_valid_flag, this_take_smplh_hand_anno_3d_valid_flag], axis=1)
		return this_take_smplh_anno_3d, this_take_smplh_anno_3d_valid_flag
