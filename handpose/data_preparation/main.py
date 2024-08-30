import copy
import json
import os
import os.path as osp
from glob import glob

import cv2
import numpy as np
from handpose_dataloader import hand_pose_anno_loader
from bodypose_dataloader import body_pose_anno_loader
from collections import defaultdict
from PIL import Image
from projectaria_tools.core import calibration
from scripts.download import find_annotated_takes
from tqdm import tqdm
from utils.config import create_arg_parse
from utils.reader import PyAvReader
from utils.utils import extract_aria_calib_to_json, get_ego_aria_cam_name
from third_party.ego_exo4d_egopose.handpose.data_preparation.utils.utils import HAND_ORDER
from egoego.utils.setup_logger import setup_logger
from egoego.utils.geom import pose_to_T, T_to_pose
from egoego.utils.egoexo.egoexo_utils import EgoExoUtils
from egoego.utils.transform import aria_camera_device2opengl_pose
from egoego.utils.aria.mps import AriaMPSService
from projectaria_tools.utils.vrs_to_mp4_utils import get_timestamp_from_mp4, convert_vrs_to_mp4
from third_party.ego_exo4d_egopose.handpose.data_preparation.data_conversion import EgoPoseDataPreparation

logger = setup_logger(__name__)

from egoego.utils.smpl_to_openpose.mapping import EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS, EGOEXO4D_EGOPOSE_HANDPOSE_MAPPINGS, USED_SMPLH_JOINT_NAMES
# from egoego.smplx_utils import NUM_OF_BODY_JOINTS as NUM_OF_SMPLH_BODY_JOINTS
# from egoego.smplx_utils import NUM_OF_HAND_JOINTS as NUM_OF_SMPLH_HAND_JOINTS

DEFICIENT_TAKE_NAMES = ["upenn_0722_Violin_1_3", "cmu_bike09_5", "georgiatech_bike_01_6"]

BODY_JOINTS = EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS
HAND_JOINTS = EGOEXO4D_EGOPOSE_HANDPOSE_MAPPINGS
NUM_OF_HAND_JOINTS = len(HAND_JOINTS) // 2
NUM_OF_BODY_JOINTS = len(BODY_JOINTS)  
NUM_OF_JOINTS = NUM_OF_BODY_JOINTS + NUM_OF_HAND_JOINTS * 2

# region
def undistort_aria_img(args):
	# Load all takes metadata
	takes = json.load(open(os.path.join(args.ego4d_data_dir, "takes.json")))

	for anno_type in args.anno_types:
		for split in args.splits:
			# Load GT annotation
			gt_anno_path = os.path.join(
				args.gt_output_dir,
				"annotation",
				anno_type,
				f"ego_pose_gt_anno_{split}_public.json",
			)
			# Check gt-anno file existence
			if not os.path.exists(gt_anno_path):
				print(
					f"[Warning] Undistortion of aria raw image fails for split={split}({anno_type}). Invalid path: {gt_anno_path}. Skipped for now."
				)
				continue
			gt_anno = json.load(open(gt_anno_path))
			# Input and output root path
			img_view_prefix = "image_portrait_view" if args.portrait_view else "image"
			dist_img_root = os.path.join(
				args.gt_output_dir, img_view_prefix, "distorted", split
			)
			undist_img_root = os.path.join(
				args.gt_output_dir, img_view_prefix, "undistorted", split
			)
			# Extract frames with annotations for all takes
			print("Undisorting Aria images...")
			for i, (take_uid, take_anno) in enumerate(gt_anno.items()):
				# Get current take's metadata
				take = [t for t in takes if t["take_uid"] == take_uid]
				assert len(take) == 1, f"Take: {take_uid} does not exist"
				take = take[0]
				# Get current take's name and aria camera name
				take_name = take["take_name"]
				print(f"[{i+1}/{len(gt_anno)}] processing {take_name}")
				# Get aria calibration model and pinhole camera model
				curr_aria_calib_json_path = os.path.join(
					args.gt_output_dir, "aria_calib_json", f"{take_name}.json"
				)
				if not os.path.exists(curr_aria_calib_json_path):
					print(f"No Aria calib json for {take_name}. Skipped.")
					continue
				aria_rgb_calib = calibration.device_calibration_from_json(
					curr_aria_calib_json_path
				).get_camera_calib("camera-rgb")
				pinhole = calibration.get_linear_camera_calibration(512, 512, 150)
				# Input and output directory
				curr_dist_img_dir = os.path.join(dist_img_root, take_name)
				if not os.path.exists(curr_dist_img_dir):
					print(
						f"[Warning] No extracted raw aria images found at {curr_dist_img_dir}. Skipped take {take_name}."
					)
					continue
				curr_undist_img_dir = os.path.join(undist_img_root, take_name)
				# if take_name in DEFICIENT_TAKE_NAMES:
				# 	print(
				# 		f"[Warning] Undistorted aria images found at {curr_undist_img_dir}. Skipped take {take_name}."
				# 	)
				# 	continue
				os.makedirs(curr_undist_img_dir, exist_ok=True)
				# Extract undistorted aria images
				num_of_imgs = len(glob(os.path.join(curr_dist_img_dir, "*.jpg")))
				assert num_of_imgs == len(take_anno.keys()), f"Number of images {num_of_imgs} does not match number of annotations {len(take_anno)}"
				for frame_number in tqdm(take_anno.keys(), total=len(take_anno.keys())):
					f_idx = int(frame_number)
					curr_undist_img_path = os.path.join(
						curr_undist_img_dir, f"{f_idx:06d}.jpg"
					)
					# Avoid repetitive generation by checking file existence
					if not os.path.exists(curr_undist_img_path):
						# Load in distorted images
						curr_dist_img_path = os.path.join(
							curr_dist_img_dir, f"{f_idx:06d}.jpg"
						)
						assert os.path.exists(
							curr_dist_img_path
						), f"No distorted images found at {curr_dist_img_path}. Please extract images with steps=raw_images first."
						curr_dist_image = np.array(Image.open(curr_dist_img_path))
						curr_dist_image = (
							cv2.rotate(curr_dist_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
							if args.portrait_view
							else curr_dist_image
						)
						# Undistortion
						undistorted_image = calibration.distort_by_calibration(
							curr_dist_image, pinhole, aria_rgb_calib
						)
						undistorted_image = (
							cv2.rotate(undistorted_image, cv2.ROTATE_90_CLOCKWISE)
							if args.portrait_view
							else undistorted_image
						)
						# Save undistorted image
						assert cv2.imwrite(
							curr_undist_img_path, undistorted_image[:, :, ::-1]
						), curr_undist_img_path
# endregion

# region
def extract_aria_img(args):
	# Load all takes metadata
	takes = json.load(open(os.path.join(args.ego4d_data_dir, "takes.json")))

	for anno_type in args.anno_types:
		for split in args.splits:
			# Load GT annotation
			gt_anno_path = os.path.join(
				args.gt_output_dir,
				"annotation",
				anno_type,
				f"ego_pose_gt_anno_{split}_public.json",
			)
			# Check gt-anno file existence
			if not os.path.exists(gt_anno_path):
				print(
					f"[Warning] Extraction of aria raw image fails for split={split}({anno_type}). Invalid path: {gt_anno_path}. Skipped for now."
				)
				continue
			gt_anno = json.load(open(gt_anno_path))
			# Input and output root path
			take_video_dir = os.path.join(args.ego4d_data_dir, "takes")
			img_view_prefix = "image_portrait_view" if args.portrait_view else "image"
			img_output_root = os.path.join(
				args.gt_output_dir, img_view_prefix, "distorted", split
			)
			os.makedirs(img_output_root, exist_ok=True)
			# Extract frames with annotations for all takes
			print("Extracting Aria images...")
			for i, (take_uid, take_anno) in enumerate(gt_anno.items()):
				# Get current take's metadata
				take = [t for t in takes if t["take_uid"] == take_uid]
				assert len(take) == 1, f"Take: {take_uid} does not exist"
				take = take[0]
				# Get current take's name and aria camera name
				take_name = take["take_name"]
				print(f"[{i+1}/{len(gt_anno)}] processing {take_name}")
				ego_aria_cam_name = get_ego_aria_cam_name(take)
				# Load current take's aria video
				curr_take_video_path = os.path.join(
					take_video_dir,
					take_name,
					"frame_aligned_videos",
					f"{ego_aria_cam_name}_214-1.mp4",
				)
				if not os.path.exists(curr_take_video_path):
					print(
						f"[Warning] No frame aligned videos found at {curr_take_video_path}. Skipped take {take_name}."
					)
					continue
				curr_take_img_output_path = os.path.join(img_output_root, take_name)
				# if osp.exists(curr_take_img_output_path):
				# 	print(
				# 		f"[Warning] Extracted raw aria images found at {curr_take_img_output_path}. Skipped take {take_name}."
				# 	)
				# 	continue
				os.makedirs(curr_take_img_output_path, exist_ok=True)
				reader = PyAvReader(
					path=curr_take_video_path,
					resize=None,
					mean=None,
					frame_window_size=1,
					stride=1,
					gpu_idx=-1,
				)
				# Extract frames
				for frame_number in tqdm(take_anno.keys(), total=len(take_anno.keys())):
					f_idx = int(frame_number)
					out_path = os.path.join(
						curr_take_img_output_path, f"{f_idx:06d}.jpg"
					)
					# Avoid repetitive generation by checking file existence
					if not os.path.exists(out_path):
						frame = reader[f_idx][0].cpu().numpy()
						frame = frame if args.portrait_view else np.rot90(frame)
						frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
						cv2.imwrite(out_path, frame)
						tmp=cv2.imread(out_path)
						assert tmp is not None, out_path
				num_of_imgs = len(glob(os.path.join(curr_take_img_output_path, "*.jpg")))
				assert num_of_imgs == len(take_anno.keys()), f"Number of images {num_of_imgs} does not match number of annotations {len(take_anno)}"
# endregion


# region
def save_test_gt_anno(output_dir, gt_anno_private):
	# 1. Save private annotated test JSON file
	with open(
		os.path.join(output_dir, f"ego_pose_gt_anno_test_private.json"), "w"
	) as f:
		json.dump(gt_anno_private, f)
	# 2. Exclude GT 2D & 3D joints and valid flag information for public un-annotated test file
	gt_anno_public = copy.deepcopy(gt_anno_private)
	for _, take_anno in gt_anno_public.items():
		for _, frame_anno in take_anno.items():
			for k in [
				"left_hand_2d",
				"right_hand_2d",
				"left_hand_3d",
				"right_hand_3d",
				"left_hand_valid_3d",
				"right_hand_valid_3d",
			]:
				frame_anno.pop(k)
	# 3. Save public un-annotated test JSON file
	with open(os.path.join(output_dir, f"ego_pose_gt_anno_test_public.json"), "w") as f:
		json.dump(gt_anno_public, f)
# endregion


# region
def create_hand_gt_anno(args):
	"""
	Creates ground truth annotation file for train, val and test split. For
	test split creates two versions:
	- public: doesn't have GT 3D joints and valid flag information, used for
	public to do local inference
	- private: has GT 3D joints and valid flag information, used for server
	to evaluate model performance
	"""
	print("Generating ground truth annotation files...")
	for anno_type in args.anno_types:
		for split in args.splits:
			# For test split, only save manual annotation
			if split == "test" and anno_type == "auto":
				print("[Warning] No test gt-anno will be generated on auto data. Skipped for now.")
			# Get ground truth annotation
			gt_anno = hand_pose_anno_loader(args, split, anno_type)
			gt_anno_output_dir = os.path.join(
				args.gt_output_dir, "annotation", anno_type
			)
			os.makedirs(gt_anno_output_dir, exist_ok=True)
			# Save ground truth JSON file
			if split in ["train", "val"]:
				with open(
					os.path.join(
						gt_anno_output_dir, f"ego_pose_gt_anno_{split}_public.json"
					),
					"w",
				) as f:
					json.dump(gt_anno.db, f)
			# For test split, create two versions of GT-anno
			else:
				if len(gt_anno.db) == 0:
					print(
						"[Warning] No test gt-anno will be generated. Please download public release from shared drive."
					)
				else:
					save_test_gt_anno(gt_anno_output_dir, gt_anno.db)
# endregion


# region
def load_hand_gt_anno(args):
	"""
	Loads hand gt annotation
	"""
	print(f"Loading [{args.split}]/[{args.anno_type}] hand-pose ground truth annotation files...")
	for anno_type in args.anno_types:
		for split in args.splits:
			# For test split, only save manual annotation
			if split == "test" and anno_type == "auto":
				print("[Warning] No test gt-anno will be generated on auto data. Skipped for now.")
			# Get ground truth annotation
			gt_anno = hand_pose_anno_loader(args, split, anno_type)
			gt_anno_output_dir = os.path.join(
				args.gt_output_dir, "annotation", anno_type
			)
			os.makedirs(gt_anno_output_dir, exist_ok=True)
			# Save ground truth JSON file
			if split in ["train", "val"]:
				anno = json.load(open(os.path.join(gt_anno_output_dir, f"ego_pose_gt_anno_{split}_public.json")))
			# For test split, create two versions of GT-anno
			else:
				hand_gt_annot_path = osp.join(gt_anno_output_dir, f"ego_pose_gt_anno_{split}_public.json")
				if not osp.exists(hand_gt_annot_path):
					print(f"[Warning] No test gt-anno is detected.")
					anno = None
				else: 
					anno = json.load(open(hand_gt_annot_path))
			yield anno, anno_type, split
# endregion

# region
def create_body_gt_anno(args):
	"""
	Creates ground truth annotation file for train, val and test split. For
	test split creates two versions:
	- public: doesn't have GT 3D joints and valid flag information, used for
	public to do local inference
	- private: has GT 3D joints and valid flag information, used for server
	to evaluate model performance
	"""
	print("Generating ground truth annotation files...")
	for anno_type in args.anno_types:
		for split in args.splits:
			# For test split, only save manual annotation
			if split == "test" and anno_type == "auto":
				print("[Warning] No test gt-anno will be generated on auto data. Skipped for now.")
			# Get ground truth annotation
			gt_anno = body_pose_anno_loader(args, split, anno_type)
			gt_anno_output_dir = os.path.join(
				args.gt_output_dir, "annotation", anno_type
			)
			os.makedirs(gt_anno_output_dir, exist_ok=True)
			# Save ground truth JSON file
			if split in ["train", "val"]:
				with open(
					os.path.join(
						gt_anno_output_dir, f"ego_pose_gt_anno_{split}_public.json"
					),
					"w",
				) as f:
					json.dump(gt_anno.db, f)
			# For test split, create two versions of GT-anno
			else:
				if len(gt_anno.db) == 0:
					print(
						"[Warning] No test gt-anno will be generated. Please download public release from shared drive."
					)
				else:
					save_test_gt_anno(gt_anno_output_dir, gt_anno.db)
# endregion

# region
def load_body_gt_anno(args):
	"""
	Loads body gt annotation
	"""
	print(f"Loading [{args.split}]/[{args.anno_type}] body-pose ground truth annotation files...")
	for anno_type in args.anno_types:
		for split in args.splits:
			# For test split, only save manual annotation
			if split == "test" and anno_type == "auto":
				print("[Warning] No test gt-anno will be generated on auto data. Skipped for now.")
			# Get ground truth annotation
			gt_anno = body_pose_anno_loader(args, split, anno_type)
			gt_anno_output_dir = os.path.join(
				args.gt_output_dir, "annotation", anno_type
			)
			os.makedirs(gt_anno_output_dir, exist_ok=True)
			# Save ground truth JSON file
			if split in ["train", "val"]:
				anno = json.load(open(os.path.join(gt_anno_output_dir, f"ego_pose_gt_anno_{split}_public.json")))
			# For test split, create two versions of GT-anno
			else:
				body_gt_annot_path = osp.join(gt_anno_output_dir, f"ego_pose_gt_anno_{split}_public.json")
				if not osp.exists(body_gt_annot_path):
					print(f"[Warning] No test gt-anno is detected.")
					anno = None
				else: 
					anno = json.load(open(body_gt_annot_path))
			yield anno, anno_type, split
# endregion


# region
def create_aria_calib(args):
	# Get all annotated takes
	all_local_take_uids = find_annotated_takes(
		args.ego4d_data_dir, args.splits, args.anno_types
	)
	# Create aria calib JSON output directory
	aria_calib_json_output_dir = os.path.join(args.gt_output_dir, "aria_calib_json")
	os.makedirs(aria_calib_json_output_dir, exist_ok=True)

	# Find uid and take info
	takes = json.load(open(os.path.join(args.ego4d_data_dir, "takes.json")))
	take_to_uid = {
		each_take["take_name"]: each_take["take_uid"]
		for each_take in takes
		if each_take["take_uid"] in all_local_take_uids
	}
	assert len(all_local_take_uids) == len(
		take_to_uid
	), "Some annotation take doesn't have corresponding info in takes.json"
	# Export aria calibration to JSON file
	print("Generating aria calibration JSON file...")
	for take_name, _ in tqdm(take_to_uid.items()):
		# Get aria name
		take = [t for t in takes if t["take_name"] == take_name]
		assert len(take) == 1, f"Take: {take_name} can't be found in takes.json"
		take = take[0]
		aria_cam_name = get_ego_aria_cam_name(take)
		# 1. Generate aria calib JSON file
		vrs_path = os.path.join(
			args.ego4d_data_dir,
			"takes",
			take_name,
			f"{aria_cam_name}_noimagestreams.vrs",
		)
		if not os.path.exists(vrs_path):
			print(
				f"[Warning] No take vrs found at {vrs_path}. Skipped take {take_name}."
			)
			continue
		output_path = os.path.join(aria_calib_json_output_dir, f"{take_name}.json")
		extract_aria_calib_to_json(vrs_path, output_path)
		# 2. Overwrite f, cx, cy parameter from JSON file
		aria_calib_json = json.load(open(output_path))
		# Overwrite f, cx, cy
		all_cam_calib = aria_calib_json["CameraCalibrations"]
		aria_cam_calib = [c for c in all_cam_calib if c["Label"] == "camera-rgb"][0]
		aria_cam_calib["Projection"]["Params"][0] /= 2
		aria_cam_calib["Projection"]["Params"][1] = (
			aria_cam_calib["Projection"]["Params"][1] - 0.5 - 32
		) / 2
		aria_cam_calib["Projection"]["Params"][2] = (
			aria_cam_calib["Projection"]["Params"][2] - 0.5 - 32
		) / 2
		# Save updated JSON calib file
		with open(os.path.join(output_path), "w") as f:
			json.dump(aria_calib_json, f)
# endregion

def load_all_anno(args):
	for (body_anno, body_anno_type, body_split), (hand_anno, hand_anno_type, hand_split) in zip(load_body_gt_anno(args), load_hand_gt_anno(args)):
		assert body_split == hand_split, "Body and hand split should be the same"
		assert body_anno_type == hand_anno_type, "Body and hand annotation type should be the same"

		yield body_anno, hand_anno, body_anno_type, body_split
 
def align_all_anno_with_slam_close_loop(args):

	egoexo_utils = EgoExoUtils(args.ego4d_data_dir)
	egopose_data_preparation = EgoPoseDataPreparation(args)
	print(f"Aligning [{args.split}]/[{args.anno_types}] exported annotations with slam close loop traj paths ...")

	for body_anno, hand_anno, anno_type, split in tqdm(load_all_anno(args)):

		body_anno_keys = list(body_anno.keys())
		hand_anno_keys = list(hand_anno.keys())
		common_keys = list(set(body_anno_keys).intersection(hand_anno_keys))
		print(f"This split / anno type {len(common_keys)} hand & body annos, out of {len(body_anno_keys)} body annos and {len(hand_anno_keys)} hand annos")

		for take_uid_idx, common_take_uid in tqdm(enumerate(common_keys), total=len(common_keys)):
			# common_take_name = egoexo_utils.take_uid_to_take_names[common_take_uid]

			this_take_body_anno = body_anno[common_take_uid]
			this_take_hand_anno = hand_anno[common_take_uid]
			this_take_body_anno_frame_keys = list(this_take_body_anno.keys())
			this_take_hand_anno_frame_keys = list(this_take_hand_anno.keys())
			this_take_common_frame_keys = list(
				set(this_take_body_anno_frame_keys).intersection(this_take_hand_anno_frame_keys)
			)
			print(f"This take has {len(this_take_common_frame_keys)} common frames, out of {len(this_take_body_anno_frame_keys)} body frames and {len(this_take_hand_anno_frame_keys)} hand frames")

			this_take_hand_anno_3d_cam = []
			this_take_hand_anno_3d_world = []
			this_take_hand_anno_3d_valid_flag = []
			this_take_hand_anno_2d_hand = []
			this_take_hand_anno_hand_bbox = []
			this_take_body_anno_3d_body = []
			this_take_body_anno_3d_valid_flag = []
			this_take_ego_cam_extrs = []
			this_take_ego_cam_intrs = []
			this_take_body_anno_2d_kpts = []
			this_take_body_anno_2d_bbox = []
			
			for _, common_frame_idx in enumerate(this_take_common_frame_keys):
				this_take_frame_body_anno = this_take_body_anno[common_frame_idx]
				this_take_frame_hand_anno = this_take_hand_anno[common_frame_idx]
				this_take_ego_cam_extr = this_take_hand_anno[common_frame_idx]["camera_extrinsics"]
				this_take_ego_cam_intr = this_take_hand_anno[common_frame_idx]["camera_intrinsics"]
				this_take_ego_cam_name = this_take_hand_anno[common_frame_idx]["camera_name"]
				this_take_exo_cam_names = this_take_hand_anno[common_frame_idx]["exo_cam_names"]
				this_take_exo_cam_intrs = this_take_hand_anno[common_frame_idx]["camera_intrinsics"]
				this_take_exo_cam_extrs = this_take_hand_anno[common_frame_idx]["camera_extrinsics"]

				this_take_frame_hand_anno_3d_world = []
				this_take_frame_hand_anno_3d_valid_flag = []

				for hand_idx, hand_name in enumerate(HAND_ORDER):

					this_take_frame_hand_anno_3d_cam = this_take_frame_hand_anno[f"{hand_name}_hand_3d_cam"] # 21 x 3
					this_take_frame_hand_anno_3d_world.append(this_take_frame_hand_anno[f"{hand_name}_hand_3d_world"]) # 21 x 3
					this_take_frame_hand_anno_3d_valid_flag.append(this_take_frame_hand_anno[f"{hand_name}_hand_valid_3d"]) # 21 x 3

					this_take_frame_hand_anno_2d_hand = this_take_frame_hand_anno[f"{hand_name}_hand_2d"] # 21 x 2
					this_take_frame_hand_anno_hand_bbox = this_take_frame_hand_anno[f"{hand_name}_hand_bbox"] # 21 x 2
				 
				this_take_frame_hand_anno_3d_world = np.stack(this_take_frame_hand_anno_3d_world, axis=0)
				this_take_frame_hand_anno_3d_valid_flag = np.stack(this_take_frame_hand_anno_3d_valid_flag, axis=0)
				this_take_hand_anno_3d_world.append(this_take_frame_hand_anno_3d_world)
				this_take_hand_anno_3d_valid_flag.append(this_take_frame_hand_anno_3d_valid_flag)

				this_take_frame_body_anno_3d_body = this_take_frame_body_anno["body_3d"] # 17 x 3
				this_take_frame_body_anno_3d_valid_flag = this_take_frame_body_anno["body_valid_3d"] # 17 x 3
				this_take_body_anno_3d_body.append(this_take_frame_body_anno_3d_body)
				this_take_body_anno_3d_valid_flag.append(this_take_frame_body_anno_3d_valid_flag)

				this_take_ego_cam_extrs.append(this_take_ego_cam_extr)
				this_take_ego_cam_intrs.append(this_take_ego_cam_intr)
				
				this_take_frame_body_anno_2d_kpts = this_take_frame_body_anno["body_2d"] # 17 x 3
				this_take_frame_body_anno_2d_bbox = this_take_frame_body_anno["body_bbox"] # 17 x 3
			 
			this_take_body_anno_3d_body = np.stack(this_take_body_anno_3d_body, axis=0)  # N x 17 x 3
			this_take_body_anno_3d_valid_flag = np.stack(this_take_body_anno_3d_valid_flag, axis=0) # N x 17 x 3
			this_take_hand_anno_3d_world = np.stack(this_take_hand_anno_3d_world, axis=0) # N x 42 x 3
			this_take_hand_anno_3d_valid_flag = np.stack(this_take_hand_anno_3d_valid_flag, axis=0) # N x 42 x 3
			this_take_anno_3d = np.concatenate([this_take_body_anno_3d_body, this_take_hand_anno_3d_world], axis=1) # N x (17+42) x 3
			this_take_anno_3d_valid_flag = np.concatenate([this_take_body_anno_3d_valid_flag, this_take_hand_anno_3d_valid_flag], axis=1) # N x (17+42) x 3

			this_take_ego_cam_extrs = np.stack(this_take_ego_cam_extrs, axis=0) # N x 3 x 4
			this_take_ego_cam_intrs = np.stack(this_take_ego_cam_intrs, axis=0) # N x 3 x 3

			this_take_ego_cam_traj = T_to_pose(this_take_ego_cam_extrs, take_inv=True) # N x 7

			this_take_aligned_anno_3d, this_take_aligned_ego_cam_traj = egopose_data_preparation.align_exported_anno_to_slam_traj(take_uid=common_take_uid, 
															 egoexo_util_inst=egoexo_utils,
															 this_take_ego_cam_traj = this_take_ego_cam_traj,
															 this_take_ego_cam_intr=this_take_ego_cam_intr,
															 this_take_anno_3d=this_take_anno_3d,
															 this_take_anno_3d_valid_flag=this_take_anno_3d_valid_flag)
			this_take_body_aligned_anno_3d, this_take_hand_aligned_anno_3d = this_take_aligned_anno_3d[:, :NUM_OF_BODY_JOINTS, :], this_take_aligned_anno_3d[:, NUM_OF_BODY_JOINTS:, :]
			this_take_opengl_ego_cam_traj = aria_camera_device2opengl_pose(this_take_aligned_ego_cam_traj)

			for _, common_frame_idx in enumerate(this_take_common_frame_keys):
				for hand_idx, hand_name in enumerate(HAND_ORDER):
					this_take_hand_anno[common_frame_idx][f"{hand_name}_hand_3d_world"] = this_take_hand_aligned_anno_3d[common_frame_idx][hand_idx*NUM_OF_HAND_JOINTS:(hand_idx+1)*NUM_OF_HAND_JOINTS] # 21 x 3
				 
				this_take_body_anno[common_frame_idx]["body_3d"] = this_take_body_aligned_anno_3d[common_frame_idx] # 17 x 3
				this_take_body_anno[common_frame_idx]["ego_cam_traj"] = this_take_opengl_ego_cam_traj[common_frame_idx] # 7
			
			body_anno[common_take_uid] = this_take_body_anno
			hand_anno[common_take_uid] = this_take_hand_anno

		aligned_anno_output_dir = os.path.join(
			args.gt_output_dir, "annotation", anno_type
		)
		os.makedirs(aligned_anno_output_dir, exist_ok=True)
		body_aligned_anno_output_path = os.path.join(aligned_anno_output_dir, f"ego_body_pose_gt_anno_{split}_public.json")
		hand_aligned_anno_output_path = os.path.join(aligned_anno_output_dir, f"ego_hand_pose_gt_anno_{split}_public.json")

		json.dump(body_anno, open(body_aligned_anno_output_path, "w"))
		json.dump(hand_anno, open(hand_aligned_anno_output_path, "w"))
		logger.info(f"Successfully writing to {body_aligned_anno_output_path} and {hand_aligned_anno_output_path} ! ")

def load_all_aligned_anno(args):
	for anno_type in args.anno_types:
		for split in args.splits:
			aligned_anno_output_dir = os.path.join(
				args.gt_output_dir, "annotation", anno_type
			)
			body_aligned_anno_output_path = os.path.join(aligned_anno_output_dir, f"ego_body_pose_gt_anno_{split}_public.json")
			hand_aligned_anno_output_path = os.path.join(aligned_anno_output_dir, f"ego_hand_pose_gt_anno_{split}_public.json")
			yield json.load(open(body_aligned_anno_output_path)), json.load(open(hand_aligned_anno_output_path)), anno_type, split

def generate_smpl_converted_anno(args):
	egopose_data_preparation = EgoPoseDataPreparation(args)
	egoexo_utils = EgoExoUtils(args.ego4d_data_dir)

	for (body_aligned_anno, hand_aligned_anno, anno_type, split) in tqdm(load_all_aligned_anno(args)):
		body_anno_keys = list(body_aligned_anno.keys())
		hand_anno_keys = list(hand_aligned_anno.keys())
		common_keys = list(set(body_anno_keys).intersection(hand_anno_keys))
		print(f"This split / anno type {len(common_keys)} hand & body annos, out of {len(body_anno_keys)} body annos and {len(hand_anno_keys)} hand annos")

		smplh_aligned_anno = defaultdict(dict)
		for take_uid_idx, common_take_uid in tqdm(enumerate(common_keys), total=len(common_keys)):
			common_take_name = egoexo_utils.take_uid_to_take_names[common_take_uid]
			smplh_aligned_anno[common_take_uid] = defaultdict(dict)

			this_take_aligned_body_anno = body_aligned_anno[common_take_uid]
			this_take_aligned_hand_anno = hand_aligned_anno[common_take_uid]
			this_take_aligned_body_anno_frame_keys = list(this_take_aligned_body_anno.keys())
			this_take_aligned_hand_anno_frame_keys = list(this_take_aligned_hand_anno.keys())
			this_take_aligned_common_frame_keys = list(
				set(this_take_aligned_body_anno_frame_keys).intersection(this_take_aligned_hand_anno_frame_keys)
			)
			print(f"This take has {len(this_take_aligned_common_frame_keys)} common frames, out of {len(this_take_aligned_body_anno_frame_keys)} body frames and {len(this_take_aligned_hand_anno_frame_keys)} hand frames")

			this_take_aligned_hand_anno_3d_world = []
			this_take_aligned_hand_anno_3d_valid_flag = []
			this_take_aligned_body_anno_3d_body = []
			this_take_aligned_body_anno_3d_valid_flag = []
			
			for _, common_frame_idx in enumerate(this_take_aligned_common_frame_keys):
				this_take_aligned_frame_body_anno = this_take_aligned_body_anno[common_frame_idx]
				this_take_aligned_frame_hand_anno = this_take_aligned_hand_anno[common_frame_idx]

				this_take_aligned_frame_hand_anno_3d_world = []
				this_take_aligned_frame_hand_anno_3d_valid_flag = []

				for hand_idx, hand_name in enumerate(HAND_ORDER):

					this_take_aligned_frame_hand_anno_3d_world.append(this_take_aligned_frame_hand_anno[f"{hand_name}_hand_3d_world"]) # 21 x 3
					this_take_aligned_frame_hand_anno_3d_valid_flag.append(this_take_aligned_frame_hand_anno[f"{hand_name}_hand_valid_3d"]) # 21 x 3
				 
				this_take_aligned_frame_hand_anno_3d_world = np.stack(this_take_aligned_frame_hand_anno_3d_world, axis=0)
				this_take_aligned_frame_hand_anno_3d_valid_flag = np.stack(this_take_aligned_frame_hand_anno_3d_valid_flag, axis=0)
				this_take_aligned_hand_anno_3d_world.append(this_take_aligned_frame_hand_anno_3d_world)
				this_take_aligned_hand_anno_3d_valid_flag.append(this_take_aligned_frame_hand_anno_3d_valid_flag)

				this_take_aligned_frame_body_anno_3d_body = this_take_aligned_frame_body_anno["body_3d"] # 17 x 3
				this_take_aligned_frame_body_anno_3d_valid_flag = this_take_aligned_frame_body_anno["body_valid_3d"] # 17 x 3
				this_take_aligned_body_anno_3d_body.append(this_take_aligned_frame_body_anno_3d_body)
				this_take_aligned_body_anno_3d_valid_flag.append(this_take_aligned_frame_body_anno_3d_valid_flag)

				smplh_aligned_anno[common_take_uid][common_frame_idx]["ego_cam_traj"] = this_take_aligned_frame_body_anno["ego_cam_traj"]
			 
			this_take_aligned_body_anno_3d_body = np.stack(this_take_aligned_body_anno_3d_body, axis=0)  # N x 17 x 3
			this_take_aligned_body_anno_3d_valid_flag = np.stack(this_take_aligned_body_anno_3d_valid_flag, axis=0) # N x 17 x 3
			this_take_aligned_hand_anno_3d_world = np.stack(this_take_aligned_hand_anno_3d_world, axis=0) # N x 42 x 3
			this_take_aligned_hand_anno_3d_valid_flag = np.stack(this_take_aligned_hand_anno_3d_valid_flag, axis=0) # N x 42 x 3

			this_take_aligned_anno_3d = np.concatenate([this_take_aligned_body_anno_3d_body, this_take_aligned_hand_anno_3d_world], axis=1) # N x (17+42) x 3
			this_take_aligned_anno_3d_valid_flag = np.concatenate([this_take_aligned_body_anno_3d_valid_flag, this_take_aligned_hand_anno_3d_valid_flag], axis=1) # N x (17+42) x 3

			this_take_smplh_anno_3d = egopose_data_preparation.generate_smpl_converted_anno(this_take_anno_3d=this_take_aligned_anno_3d, 
																  this_take_anno_3d_valid_flag=this_take_aligned_anno_3d_valid_flag,
																seq_name=common_take_name)

			smplh_aligned_anno[common_take_uid]["smplh_aligned_pose"] = this_take_smplh_anno_3d # T x 52 x 3
			smplh_aligned_anno[common_take_uid]["metadata"] = this_take_aligned_hand_anno["metadata"]

		smplh_anno_output_dir = os.path.join(
			args.gt_output_dir, "annotation", anno_type
		)
		os.makedirs(smplh_anno_output_dir, exist_ok=True)
		smplh_aligned_anno_output_path = os.path.join(smplh_anno_output_dir, f"ego_body_pose_gt_anno_{split}_public.json")

		json.dump(smplh_anno_output_dir, open(smplh_aligned_anno_output_path, "w"))
		logger.info(f"Successfully writing to {smplh_aligned_anno_output_path} ! ")

def convert_all_vrs_to_mp4(args):

	# Load all takes metadata
	takes = json.load(open(os.path.join(args.ego4d_data_dir, "takes.json")))

	for anno_type in args.anno_types:
		for split in args.splits:
			# Load GT annotation
			gt_anno_path = os.path.join(
				args.gt_output_dir,
				"annotation",
				anno_type,
				f"ego_pose_gt_anno_{split}_public.json",
			)
			# Check gt-anno file existence
			if not os.path.exists(gt_anno_path):
				print(
					f"[Warning] Conversion of vrs to mp4 files fails for split={split}({anno_type}). Invalid path: {gt_anno_path}. Skipped for now."
				)
				continue
			gt_anno = json.load(open(gt_anno_path))
			# Input and output root path
			take_video_dir = os.path.join(args.ego4d_data_dir, "takes")
			mp4_output_root = os.path.join(
				args.gt_output_dir, "exported_mp4", split
			)
			os.makedirs(mp4_output_root, exist_ok=True)
			# Extract frames with annotations for all takes
			print("Exporting Aria vrs to mp4...")
			for i, (take_uid, take_anno) in enumerate(gt_anno.items()):
				# Get current take's metadata
				take = [t for t in takes if t["take_uid"] == take_uid]
				assert len(take) == 1, f"Take: {take_uid} does not exist"
				take = take[0]
				# Get current take's name and aria camera name
				take_name = take["take_name"]
				print(f"[{i+1}/{len(gt_anno)}] processing {take_name}")
				ego_aria_cam_name = get_ego_aria_cam_name(take)
				# Load current take's aria video
				curr_take_vrs_path = os.path.join(
					take_video_dir,
					take_name,
					f"{ego_aria_cam_name}.vrs",
				)
				if not os.path.exists(curr_take_vrs_path):
					print(
						f"[Warning] No frame aligned videos found at {curr_take_vrs_path}. Skipped take {take_name}."
					)
					continue
				curr_mp4_output_path = osp.join(mp4_output_root, f"{take_name}.mp4")
				convert_vrs_to_mp4(vrs_file=curr_take_vrs_path, output_video=mp4_output_root, log_folder=None, down_sample_factor=1)
				# Testing for extracting timestamps from exported mp4 files.
				exported_mp4_timestamps = get_timestamp_from_mp4(curr_mp4_output_path)
				print(f"Exported mp4 timestamps: {exported_mp4_timestamps.shape}")
			


def main(args):
	for step in args.steps:
		if step == "aria_calib":
			create_aria_calib(args)
		elif step == "hand_gt_anno":
			create_hand_gt_anno(args)
		elif step == "body_gt_anno":
			create_body_gt_anno(args)
		elif step == "raw_image":
			extract_aria_img(args)
		elif step == "undistorted_image":
			undistort_aria_img(args)
		elif step == "load_hand_gt_anno":
			load_hand_gt_anno(args)
		elif step == "load_body_gt_anno":
			load_body_gt_anno(args)
		elif step == "load_all_anno":
			load_all_anno(args)
		elif step == "align_all_anno_with_slam_close_loop":
			align_all_anno_with_slam_close_loop(args)
		elif step == "generate_smpl_converted_anno":
			generate_smpl_converted_anno(args)
		elif step == "convert_vrs_to_mp4":
			convert_all_vrs_to_mp4(args)
	  


if __name__ == "__main__":
	args = create_arg_parse()
	main(args)
