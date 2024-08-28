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
from egoego.config import default_cfg as CFG

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
		self.ego4d_data_dir = args.ego4d_data_dir
		self.gt_output_dir = args.gt_output_dir
		self.portrait_view = args.portrait_view
		self.valid_kpts_num_thresh = args.valid_kpts_num_thresh
		self.bbox_padding = args.bbox_padding
		self.reproj_error_threshold = args.reproj_error_threshold
		self.hand_order = HAND_ORDER

		# Load calibration
		# self.calib = calibration.Calibration(self.ego4d_data_dir)

		# Load interested takes
		self.interested_takes = get_interested_take(self.ego4d_data_dir, self.ego_pose_takes)

	def align_exported_anno_to_slam_traj(self, exported_annotation, reference_slam_traj):
		pass
	
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
		all_pred_pelvis_origins = np.full((T, 3), np.nan)

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

	def generate_smpl_converted_anno(self, this_take_anno_3d, this_take_anno_3d_valid_flag, pred_pelvis_origins):
	


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
									device=device)

			this_frame_hip_trans = this_frame_hip_trans.unsqueeze(0)
			opt_local_aa_rep  = opt_local_aa_rep.unsqueeze(0)
			smplh_betas = torch.zeros(1, 10)
			smplh_betas = smplh_betas.unsqueeze(0)

			opt_pred_smplh_jnts, opt_pred_smplh_verts, opt_pred_smplh_faces = smplx_utils.run_smpl_model(this_frame_hip_trans, opt_local_aa_rep, smplh_betas, smplx_utils.bm_dict[CFG.smplh.smplh_model])
			# B(1) x T x 52 x 3, B(1) x T x 6890 x 3, 13776 x 3
			dest_mesh_vis_folder = CFG.io.save_mesh_vis_folder
			os.makedirs(dest_mesh_vis_folder, exist_ok=True)
			gen_full_body_vis(opt_pred_smplh_verts[0], opt_pred_smplh_faces, dest_mesh_vis_folder, seq_name, vis_gt=False)

		import matplotlib.pyplot as plt

		fig = plt.figure()
		ax = fig.add_subplot(projection='3d')
		RADIUS = 1.0
		target_joints = target_joints.detach().cpu().numpy()
		xroot, yroot, zroot = target_joints[0,
											0], target_joints[0, 1], target_joints[0, 2]
		ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
		ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
		ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

		ax.scatter(target_joints[:, 0], target_joints[:, 1],
				target_joints[:, 2], c='b', marker='x')
		# ax.scatter(opt_joints[:, 0], opt_joints[:, 1],
		#            opt_joints[:, 2], c='r', marker='o')
		ax.grid(True)
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')

		plt.show()
			
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
