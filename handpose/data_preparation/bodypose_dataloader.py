import json
import os

import cv2
from collections import defaultdict
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
	body_jnts_dist_angle_check,
	reproj_error_check,
	world_to_cam,
)
from egoego.utils.smpl_to_openpose.mapping import EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS
from egoego.utils.egoexo.egoexo_utils import EgoExoUtils
from egoego.utils.aria.aria_calib import CalibrationUtilities
from egoego.utils.setup_logger import setup_logger
import math

logger = setup_logger(__name__)

# Loads dataframe at target path to csv
def load_csv_to_df(filepath: str) -> pd.DataFrame:
	with open(filepath, "r") as csv_file:
		return pd.read_csv(csv_file)

class body_pose_anno_loader:
	"""
	Load Ego4D data and create ground truth annotation JSON file for egoexo4d body-pose baseline model
	"""

	def __init__(self, args, split, anno_type):
		# Set dataloader parameters
		self.dataset_root = args.ego4d_data_dir
		self.anno_type = anno_type
		self.split = split
		self.num_joints = len(EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS)  # Number of joints for single body
		self.undist_img_dim = (2160, 3840)  # Dimension of undistorted aria image [H, W]
		self.valid_kpts_threshold = (
			args.valid_kpts_num_thresh
		)  # Threshold of minimum number of valid kpts in single body
		self.bbox_padding = (
			args.bbox_padding
		)  # Amount of pixels to pad around kpts to find bbox
		self.reproj_error_threshold = args.reproj_error_threshold
		self.portrait_view = (
			args.portrait_view
		)  # Whether use portrait view (Default is landscape view)
		self.aria_calib_dir = os.path.join(args.gt_output_dir, "aria_calib_json")
		self.takes = json.load(open(os.path.join(self.dataset_root, "takes.json")))
		self.splits = json.load(
			open(os.path.join(self.dataset_root, "annotations/splits.json"))
		)

		# Determine annotation and camera pose directory
		anno_type_dir_dict = {"manual": "annotation", "auto": "automatic"}
		self.body_anno_dir = (
			os.path.join(
				self.dataset_root, "annotations/ego_pose/test/body/annotation"
			)
			if self.split == "test"
			else os.path.join(
				self.dataset_root,
				f"annotations/ego_pose/{split}/body",
				anno_type_dir_dict[self.anno_type],
			)
		)
		self.cam_pose_dir = os.path.join(
			self.dataset_root, f"annotations/ego_pose/{split}/camera_pose"
		)

		# Load dataset
		self.db = self.load_raw_data()

	def load_raw_data(self):
		gt_db = {}

		# Find all annotation takes from local direcctory by splits
		# Check test anno availability. No gt-anno will be generated for public.
		if not os.path.exists(self.body_anno_dir):
			assert (
				self.split == "test"
			), f"No annotation found for {self.split} split at {self.body_anno_dir}.\
				Make sure you follow step 0 to download data first."
			return gt_db
		# Get all local annotation takes for train/val split
		split_all_local_takes = [
			k.split(".")[0] for k in os.listdir(self.body_anno_dir)
		]
		# take to uid dict
		take_to_uid = {
			t["take_name"]: t["take_uid"]
			for t in self.takes
			if t["take_uid"] in split_all_local_takes
		}
		uid_to_take = {uid: take for take, uid in take_to_uid.items()}

		# Get all valid local take uids that are used in current split
		# 1. Filter takes based on split (train/val/test)
		curr_split_uid = self.splits["split_to_take_uids"][self.split]
		# 2. Filter takes based on benchmark (ego_pose)
		ego_pose_uid = get_ego_pose_takes_from_splits(self.splits)
		curr_split_ego_pose_uid = list(set(curr_split_uid) & set(ego_pose_uid))
		# 3. Filter common takes
		common_take_uid = list(
			set(split_all_local_takes) & set(curr_split_ego_pose_uid)
		)
		# 4. Filter takes with available camera pose file
		available_cam_pose_uid = [
			k.split(".")[0] for k in os.listdir(self.cam_pose_dir)
		]
		comm_take_w_cam_pose = sorted(list(set(common_take_uid) & set(available_cam_pose_uid)))
		print(
			f"Find {len(comm_take_w_cam_pose)} takes in {self.split} ({self.anno_type}) dataset. Start data processing..."
		)

		overall_frames_num = 0
		overall_valid_frames_num = 0

		# Iterate through all takes from annotation directory and check
		for curr_take_uid in comm_take_w_cam_pose:
			curr_take_name = uid_to_take[curr_take_uid]
			# Load annotation, camera pose JSON and image directory
			curr_take_anno_path = os.path.join(
				self.body_anno_dir, f"{curr_take_uid}.json"
			)
			curr_take_cam_pose_path = os.path.join(
				self.cam_pose_dir, f"{curr_take_uid}.json"
			)
			curr_take = [t for t in self.takes if t["take_name"] == curr_take_name][0]
			traj_dir = os.path.join(self.dataset_root, curr_take["root_dir"], "trajectory")
			exo_traj_path = os.path.join(traj_dir, "gopro_calibs.csv")

			assert os.path.exists(exo_traj_path), f"Exo trajectory file not found at {exo_traj_path}."
			assert os.path.exists(curr_take_anno_path), f"Annotation file not found at {curr_take_anno_path}."
			assert os.path.exists(curr_take_cam_pose_path), f"Camera pose file not found at {curr_take_cam_pose_path}."

			exo_traj_df = load_csv_to_df(exo_traj_path)
			# print(f"Processing take {exo_traj_path}...")

			# Load in annotation JSON and image directory
			curr_take_anno = json.load(open(curr_take_anno_path))
			curr_take_cam_pose = json.load(open(curr_take_cam_pose_path))

			# Get valid takes info for all frames
			if len(curr_take_anno) > 0:
				# aria_mask, aria_cam_name = self.load_aria_calib(curr_take_name)
				egoexo_cam_masks, egoexo_cam_names = CalibrationUtilities.get_exo_cam_masks(curr_take, exo_traj_df, portrait_view=self.portrait_view, dimension=self.undist_img_dim[::-1])
				# egoexo_cam_names = EgoExoUtils.get_exo_cam_names(curr_take)
				if egoexo_cam_names is not None:
					curr_take_data = self.load_take_raw_data(
						curr_take_name,
						curr_take_uid,
						curr_take_anno,
						curr_take_cam_pose,
						egoexo_cam_names,
						egoexo_cam_masks,
					)
					# Append into dataset if has at least valid annotation
					if len(curr_take_data) > 0:
						gt_db[curr_take_uid] = curr_take_data
						overall_frames_num += len(curr_take_anno.values())
						overall_valid_frames_num += len(curr_take_data)
					else:
						pass
						# logger.warning(f"Take {curr_take_name} has no valid annotation. Skipped this take.")
		logger.warning(f"Egoexo4d dataset achieves {overall_valid_frames_num}/{overall_frames_num} valid frames.")
		return gt_db

	def load_take_raw_data(
		self,
		take_name,
		take_uid,
		anno,
		cam_pose,
		egoexo_cam_names,
		egoexo_cam_masks,
	):
		curr_take_db = {}

		for frame_idx, curr_frame_anno in anno.items():
			# Load in current frame's 2D & 3D annotation and camera parameter
			curr_body_2d_kpts, curr_body_3d_kpts, _ = self.load_frame_body_2d_3d_kpts(
				curr_frame_anno, egoexo_cam_names
			)

			miss_left_ankle_flag = False
			miss_right_ankle_flag = False

			curr_intrs, curr_extrs = self.load_static_ego_exo_cam_poses(
				cam_pose, egoexo_cam_names
			)
			# Skip this frame if missing valid data
			if curr_body_3d_kpts is None or curr_body_2d_kpts is None or curr_extrs is None or curr_intrs is None:
				continue
			in_miss_anno_3d_flag = np.any(np.isnan(curr_body_3d_kpts), axis=1).astype(bool)

			# body biomechanical structure check
			curr_body_3d_kpts, _ = body_jnts_dist_angle_check(curr_body_3d_kpts)
			body_jnts_dist_angle_check_flag = np.logical_xor(np.any(np.isnan(curr_body_3d_kpts), axis=1).astype(bool),in_miss_anno_3d_flag.astype(bool))
			logger.info(f"Frame {frame_idx} miss {sum(in_miss_anno_3d_flag)}/{self.num_joints} valid 3d kpts, body biochemical not passed: {sum(body_jnts_dist_angle_check_flag)}.")

			all_body_annot_valid = np.ones((len(egoexo_cam_names),), dtype=bool)

			for curr_ind, (curr_intr, curr_extr, curr_egoexo_cam_mask, curr_ego_exo_cam_name) in zip(curr_intrs.values(), curr_extrs.values(), egoexo_cam_masks.values(), egoexo_cam_names):
				# Look at each body in current frame
				curr_frame_anno = defaultdict(dict)
				this_body_anno_valid = False
				valid_3d_kpts_flag = np.ones((self.num_joints,), dtype=bool)

				# Get current body's 2D kpts and 3D world kpts
				body_idx = 0
				start_idx, end_idx = self.num_joints * body_idx, self.num_joints * (
					body_idx + 1
				)
				this_cam_body_2d_kpts = curr_body_2d_kpts[curr_ego_exo_cam_name][start_idx:end_idx].copy()

				in_miss_anno_2d_flag = np.any(np.isnan(this_cam_body_2d_kpts), axis=1).astype(bool)

				# Transform annotation 2d kpts if in portrait view
				if self.portrait_view:
					this_cam_body_2d_kpts = aria_landscape_to_portrait(
						this_cam_body_2d_kpts, self.undist_img_dim
					)
				this_body_3d_kpts_world = curr_body_3d_kpts[start_idx:end_idx].copy()
				# Skip this body if left-ankle and right-ankle are both None
				left_ankle_idx = EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS.index("left-ankle")
				right_ankle_idx = EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS.index("right-ankle")

				miss_left_ankle_flag = np.any(np.isnan(this_body_3d_kpts_world[left_ankle_idx]))
				miss_right_ankle_flag = np.any(np.isnan(this_body_3d_kpts_world[right_ankle_idx]))

				if np.any(np.isnan(this_body_3d_kpts_world[left_ankle_idx])) and np.any(np.isnan(this_body_3d_kpts_world[right_ankle_idx])):
					this_body_3d_kpts_world[:, :] = None

				# 3D world to camera
				this_body_3d_kpts_cam = world_to_cam(this_body_3d_kpts_world, curr_extr)
				# Camera to image plane
				this_cam_body_proj_2d_kpts = cam_to_img(this_body_3d_kpts_cam, curr_intr)
				# Transform projected 2d kpts if in portrait view
				if self.portrait_view:
					this_cam_body_proj_2d_kpts = aria_landscape_to_portrait(
						this_cam_body_proj_2d_kpts, self.undist_img_dim
					)

				# Filter projected 2D kpts
				(
					this_cam_body_filtered_proj_2d_kpts,
					valid_proj_2d_flag,
				) = self.body_kpts_valid_check(this_cam_body_proj_2d_kpts, curr_egoexo_cam_mask)

				# Filter 2D annotation kpts
				this_body_filtered_anno_2d_kpts, _ = self.body_kpts_valid_check(
					this_cam_body_2d_kpts, curr_egoexo_cam_mask
				)

				# Filter 3D anno by checking reprojection error with 2D anno (which is usually better)
				valid_reproj_flag = reproj_error_check(
					this_cam_body_filtered_proj_2d_kpts,
					this_body_filtered_anno_2d_kpts,
					self.reproj_error_threshold,
				)
				valid_3d_kpts_flag = valid_proj_2d_flag * valid_reproj_flag

				# Prepare 2d kpts, 3d kpts, bbox and flag data based on number of valid 3D kpts
				if sum(valid_3d_kpts_flag) >= self.valid_kpts_threshold:
					this_body_anno_valid = True
					# Generate body bbox based on 2D GT kpts
					if self.split == "test":
						this_body_bbox = rand_bbox_from_kpts(
							this_cam_body_filtered_proj_2d_kpts[valid_3d_kpts_flag],
							self.undist_img_dim,
						)
					else:
						# For train and val, generate body bbox with padding
						this_body_bbox = pad_bbox_from_kpts(
							this_cam_body_filtered_proj_2d_kpts[valid_3d_kpts_flag],
							self.undist_img_dim,
							self.bbox_padding,
						)
					# logger.info(f"Frame successes! [{frame_idx}]/[{curr_ego_exo_cam_name}] achieved {sum(valid_3d_kpts_flag)}/{self.num_joints} valid 3d kpts. \
					# 			fail 2d anno kpts: {sum(in_miss_anno_2d_flag)}/{self.num_joints} \
					# 			passed 2d proj in frames: {sum(valid_proj_2d_flag)}/{self.num_joints}. \
					# 			passed 3d-2d reproj err tests: {sum(valid_reproj_flag)}/{self.num_joints}")
				# If no valid annotation for current body, assign empty bbox, anno and valid flag only for this cam.
				else:
					# logger.info(f"Frame fails! [{frame_idx}]/[{curr_ego_exo_cam_name}] achieved {sum(valid_3d_kpts_flag)}/{self.num_joints} valid 3d kpts. \
					# 			fail 2d anno kpts: {sum(in_miss_anno_2d_flag)}/{self.num_joints} \
					# 			passed 2d proj in frames: {sum(valid_proj_2d_flag)}/{self.num_joints}. \
					# 			passed 3d-2d reproj err tests: {sum(valid_reproj_flag)}/{self.num_joints}, missing left-ankle: {miss_left_ankle_flag}, missing right-ankle: {miss_right_ankle_flag}.")
					this_body_bbox = np.array([])
					this_body_filtered_anno_2d_kpts = np.array([])

				curr_frame_anno["body_2d"][curr_ego_exo_cam_name] = this_body_filtered_anno_2d_kpts.tolist()
				curr_frame_anno["body_bbox"][curr_ego_exo_cam_name] = this_body_bbox.tolist()
				all_body_annot_valid[curr_ind] = this_body_anno_valid

			# Compose current body GT info in current frame
			# Assign original hand left/right ankle 3d kpts back (needed for offset left/right ankle to determine floor heights)
			body_filtered_3d_kpts_world = curr_body_3d_kpts.copy()
			body_filtered_3d_kpts_world[~valid_3d_kpts_flag] = None
			body_filtered_3d_kpts_world[left_ankle_idx] = curr_body_3d_kpts[left_ankle_idx]
			body_filtered_3d_kpts_world[right_ankle_idx] = curr_body_3d_kpts[right_ankle_idx]

			assert sum(np.isnan(body_filtered_3d_kpts_world)) == sum(~valid_3d_kpts_flag), f"Missing 3d kpts count mismatch: {sum(np.isnan(body_filtered_3d_kpts_world))} vs {sum(~valid_3d_kpts_flag)}"

			curr_frame_anno[
				"body_3d"
			] = body_filtered_3d_kpts_world.tolist()
			curr_frame_anno[
				"body_valid_3d"
			] = valid_3d_kpts_flag.tolist()

			# Append current frame into GT JSON if at least one valid body exists
			minimum_body_annot_valid = math.ceil(len(egoexo_cam_names)*0.6)
			if sum(all_body_annot_valid)>=minimum_body_annot_valid:
				metadata = {
					"take_uid": take_uid,
					"take_name": take_name,
					"frame_number": int(frame_idx),
					"exo_cam_names": egoexo_cam_names,
					"camera_intrinsics": {k: v.tolist() for k, v in curr_intrs.items()},
					"camera_extrinsics": {k: v.tolist() for k, v in curr_extrs.items()},
				}
				curr_frame_anno["metadata"] = metadata
				curr_take_db[frame_idx] = curr_frame_anno
			else:
				logger.warning(f"Frame {frame_idx} has no valid annotation, achieved {sum(all_body_annot_valid)}/{len(egoexo_cam_names)} cmaera checks. Skipped this frame.")
				pass
		logger.info(f"Take {take_name} has {len(curr_take_db)}/{len(anno.items())} valid frames.")

		return curr_take_db

	def load_aria_calib(self, curr_take_name):
		# Find aria names
		take = [t for t in self.takes if t["take_name"] == curr_take_name]
		take = take[0]
		aria_cam_name = get_ego_aria_cam_name(take)

		# Load aria calibration model
		curr_aria_calib_json_path = os.path.join(
			self.aria_calib_dir, f"{curr_take_name}.json"
		)
		if not os.path.exists(curr_aria_calib_json_path):
			print(
				f"[Warning] No Aria calibration JSON file found at {curr_aria_calib_json_path}. Skipped this take."
			)
			return None, None
		aria_rgb_calib = calibration.device_calibration_from_json(
			curr_aria_calib_json_path
		).get_camera_calib("camera-rgb")
		dst_cam_calib = calibration.get_linear_camera_calibration(512, 512, 150)
		# Generate mask in undistorted aria view
		mask = np.full((1408, 1408), 255, dtype=np.uint8)
		undistorted_mask = calibration.distort_by_calibration(
			mask, dst_cam_calib, aria_rgb_calib
		)
		undistorted_mask = (
			cv2.rotate(undistorted_mask, cv2.ROTATE_90_CLOCKWISE)
			if self.portrait_view
			else undistorted_mask
		)
		undistorted_mask = undistorted_mask / 255
		return undistorted_mask, aria_cam_name

	def load_frame_body_2d_3d_kpts(self, frame_anno, egoexo_cam_names):
		"""
		load frame body 2d and 3d kpts for this frame.

		Parameters
		----------
		frame_anno : dict, annotation for current frame
		egoexo_cam_names : list,  egoexo camera names

		Returns
		-------
		curr_frame_2d_kpts : dict of numpy array of shape (17,2) 
			each key being the egoexo cam name, with corresponding value being the 2D body keypoints in original frame
		curr_frame_3d_kpts : (17,3) 3D body keypoints in world coordinate system
		joints_view_stat : (17,) Number of triangulation views for each 3D body keypoints

		"""
		### Load 2D GT body kpts ###
		# Return NaN if no annotation exists
		all_cam_curr_frame_2d_kpts = {}
		for egoexo_cam_name in egoexo_cam_names:
			if (
				len(frame_anno) == 0
				or "annotation2D" not in frame_anno[0].keys()
				or egoexo_cam_name not in frame_anno[0]["annotation2D"].keys()
				or len(frame_anno[0]["annotation2D"][egoexo_cam_name]) == 0
			):
				curr_frame_2d_kpts = [[None, None] for _ in range(self.num_joints)]
			else:
				curr_frame_2d_anno = frame_anno[0]["annotation2D"][egoexo_cam_name]
				curr_frame_2d_kpts = []
				# Load 3D annotation for both bodys
				for body_jnt in EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS:
					if body_jnt in curr_frame_2d_anno.keys():
						curr_frame_2d_kpts.append(
							[
								curr_frame_2d_anno[body_jnt]["x"],
								curr_frame_2d_anno[body_jnt]["y"],
							]
						)
					else:
						curr_frame_2d_kpts.append([None, None])
			all_cam_curr_frame_2d_kpts[egoexo_cam_name] = np.array(curr_frame_2d_kpts).astype(np.float32)

		### Load 3D GT body kpts ###
		# Return NaN if no annotation exists
		if (
			len(frame_anno) == 0
			or "annotation3D" not in frame_anno[0].keys()
			or len(frame_anno[0]["annotation3D"]) == 0
		):
			return None, None, None
		else:
			curr_frame_3d_anno = frame_anno[0]["annotation3D"]
			curr_frame_3d_kpts = []
			joints_view_stat = []
			# Load 3D annotation for both bodys
			for body_jnt in EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS:
				if body_jnt in curr_frame_3d_anno.keys() and (
					curr_frame_3d_anno[body_jnt]["num_views_for_3d"]
					>= 2
					or self.anno_type == "auto"
				):
					curr_frame_3d_kpts.append(
						[
							curr_frame_3d_anno[body_jnt]["x"],
							curr_frame_3d_anno[body_jnt]["y"],
							curr_frame_3d_anno[body_jnt]["z"],
						]
					)
					joints_view_stat.append(
						curr_frame_3d_anno[body_jnt][
							"num_views_for_3d"
						]
					)
				else:
					curr_frame_3d_kpts.append([None, None, None])
					joints_view_stat.append(None)
					
		return (
			all_cam_curr_frame_2d_kpts,
			np.array(curr_frame_3d_kpts).astype(np.float32),
			np.array(joints_view_stat).astype(np.float32),
		)

	def load_static_ego_exo_cam_poses(self, cam_pose, egoexo_cam_names):
		"""
		Load static camera poses for ego and exo cameras for current frame
		Retrun a dict for each key being the egoexo_cam_name
		NOTE: intr is 3x3 matrix, extr is 3x4 matrix"""
		# Check if current frame has corresponding camera pose
		curr_cam_intrs = {}
		curr_cam_extrs = {}
		for egoexo_cam_name in egoexo_cam_names:
			if (
				egoexo_cam_name not in cam_pose.keys()
				or "camera_intrinsics" not in cam_pose[egoexo_cam_name].keys()
				or "camera_extrinsics" not in cam_pose[egoexo_cam_name].keys()
			):
				curr_cam_intrinsics, curr_cam_extrinsics = None, None
			else:
				# Build camera projection matrix
				curr_cam_intrinsics = np.array(
					cam_pose[egoexo_cam_name]["camera_intrinsics"]
				).astype(np.float32)
				curr_cam_extrinsics = np.array(
					cam_pose[egoexo_cam_name]["camera_extrinsics"]
				).astype(np.float32)
			curr_cam_intrs[egoexo_cam_name] = curr_cam_intrinsics
			curr_cam_extrs[egoexo_cam_name] = curr_cam_extrinsics
		return curr_cam_intrs, curr_cam_extrs

	def body_kpts_valid_check(self, kpts, egoexo_cam_mask):
		"""
		Return valid kpts with three checks:
			- Has valid kpts
			- Within image bound
			- Visible within aria mask
		Input:
			kpts: (17,2) raw single 2D body kpts
			egoexo_cam_masks: (H,W) binary mask that has same shape as undistorted aria image
		Output:
			new_kpts: (17,2)
			flag: (17,)
		"""
		new_kpts = kpts.copy()
		# 1. Check missing annotation kpts
		miss_anno_flag = np.any(np.isnan(kpts), axis=1)
		new_kpts[miss_anno_flag] = 0
		# 2. Check out-bound annotation kpts
		# Width
		x_out_bound = np.logical_or(
			new_kpts[:, 0] < 0, new_kpts[:, 0] >= self.undist_img_dim[1]
		)
		# Height
		y_out_bound = np.logical_or(
			new_kpts[:, 1] < 0, new_kpts[:, 1] >= self.undist_img_dim[0]
		)
		out_bound_flag = np.logical_or(x_out_bound, y_out_bound)
		new_kpts[out_bound_flag] = 0
		# 3. Check in-bound but invisible kpts
		invis_flag = (
			egoexo_cam_mask[new_kpts[:, 1].astype(np.int64), new_kpts[:, 0].astype(np.int64)]
			== 0
		)
		# 4. Get valid flag
		invalid_flag = miss_anno_flag + out_bound_flag + invis_flag
		valid_flag = ~invalid_flag
		# 5. Assign invalid kpts as None
		new_kpts[invalid_flag] = None

		return new_kpts, valid_flag

