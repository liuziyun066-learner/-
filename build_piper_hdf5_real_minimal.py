import argparse
import json
import os
import sys
import time

import cv2
import h5py
import numpy as np
import torch

from piper_ik_from_me import PiperIK


# 读取Phantom动作，计算PiPER关节目标并保存HDF5。
# 默认只读；--send允许发送，--live-camera记录执行前的实时观测。


# 基本设置。默认仍兼容原目录，也允许在设备主机上通过参数覆盖。
parser = argparse.ArgumentParser(description="PiPER 真机回放与反馈记录")
parser.add_argument("episode_number", nargs="?", help="与离线版相同的 episode 编号")
parser.add_argument("--send", action="store_true", help="允许发送真机命令")
parser.add_argument("--live-camera", action="store_true", help="通过ROS 2实时采集RGB和深度")
parser.add_argument(
    "--rgb-topic",
    default="/cam_high/cam_high/color/image_raw",
    help="ROS 2彩色Image话题",
)
parser.add_argument(
    "--depth-topic",
    default="/cam_high/cam_high/aligned_depth_to_color/image_raw",
    help="ROS 2中已对齐到RGB的深度Image话题",
)
parser.add_argument(
    "--camera-info-topic",
    default="/cam_high/cam_high/color/camera_info",
    help="与RGB对应的ROS 2 CameraInfo话题",
)
parser.add_argument("--camera-fps", type=float, default=30.0, help="实时相机帧率")
parser.add_argument(
    "--live-depth-scale",
    type=float,
    default=0.001,
    help="实时深度换算为米的系数；RealSense 16UC1通常为0.001",
)
parser.add_argument("--sync-tolerance", type=float, default=0.03, help="三路采样时间最大跨度，单位秒")
parser.add_argument("--camera-max-age", type=float, default=0.5, help="允许的图像最大接收年龄，单位秒")
parser.add_argument("--feedback-max-age", type=float, default=0.5, help="允许的机械臂反馈最大年龄，单位秒")
parser.add_argument("--command-topic", default="/joint_position_ctrl", help="绝对关节位置命令话题")
parser.add_argument("--feedback-topic", default="/joint_states_single", help="右臂关节反馈话题")
parser.add_argument("--status-topic", default="/arm_status", help="右臂状态话题")
parser.add_argument("--ros-node-name", default="piper_hdf5_ros_io", help="本采集进程的ROS 2节点名前缀")
parser.add_argument("--episode-name", default="ego_test", help="输入数据中的episode组名")
parser.add_argument(
    "--data-root",
    default=os.environ.get("PHANTOM_DATA_ROOT", "/root/autodl-tmp/phantom"),
    help="包含data和calibration目录的Phantom根目录",
)
parser.add_argument(
    "--urdf",
    default=os.environ.get(
        "PIPER_URDF",
        "/root/autodl-tmp/piper_ros/src/piper_description/urdf/"
        "piper_no_gripper_description.urdf",
    ),
    help="PiPER无夹爪URDF路径",
)
parser.add_argument("--calibration", help="base_from_phantom.json路径")
parser.add_argument("--gripper-calibration", help="gripper_calibration.json路径")
parser.add_argument("--output-file", help="输出HDF5路径；默认写入data-root")
args = parser.parse_args()
if args.live_camera:
    live_values = [args.camera_fps, args.live_depth_scale, args.sync_tolerance,
                   args.camera_max_age, args.feedback_max_age]
    if not np.isfinite(live_values).all() or min(live_values) <= 0:
        parser.error("相机帧率、深度系数、同步容差和消息时限必须为正数")
episode_number = args.episode_number
send_to_robot = args.send
if episode_number is None:
    episode_number = input("请输入 episode 编号:").strip()

if not episode_number.isdigit():
    raise ValueError("episode 编号必须是数字")
target_hand = "right"
episode_name = args.episode_name
episode_group = "episode_" + episode_number.zfill(4)

data_root = os.path.abspath(os.path.expanduser(args.data_root))
raw_data_dir = os.path.join(data_root, "data", "raw_data", episode_name, episode_number)
processed_data_dir = os.path.join(
    data_root, "data", "processed_data", episode_name, episode_number,
)

action_file = os.path.join(
    processed_data_dir,
    "action_processor/actions_" + target_hand + "_single_arm.npz",
)
video_file = os.path.join(raw_data_dir, "video_L.mp4")
depth_file = os.path.join(raw_data_dir, "depth.npy")
camera_intrinsics_file = os.path.join(raw_data_dir, "cam_intrinsics.json")
hdf5_file = os.path.abspath(os.path.expanduser(
    args.output_file
    or os.path.join(data_root, episode_name + "_" + episode_number + "_piper_dataset.hdf5")
))
urdf_file = os.path.abspath(os.path.expanduser(args.urdf))

task_name = "test_task"
control_hz = 10.0

# 【真机修改】False只计算IK，True读取真机反馈。
use_real_robot = True
if args.live_camera and not use_real_robot:
    raise ValueError("--live-camera需要use_real_robot=True")
if send_to_robot and not use_real_robot:
    raise ValueError("use_real_robot=False时不能使用--send")

# 【真机修改】发送速度、步长和到位容差
speed_percent = 5.0
max_joint_step = 0.05
joint_tolerance = 0.03
gripper_tolerance = 0.005
command_timeout = 8.0

# depth.npy当前按米读取。如果原始单位是毫米,应改成0.001
depth_scale = 1.0

# 固定夹爪标定不使用gripper_scale；保留离线版默认值。
gripper_scale = 1.0
gripper_min = 0.0
gripper_max = 0.07

# 绝对坐标标定不使用move_scale；保留离线版默认值。
move_scale = 0.25

# 坐标和夹爪标定仍由现有文件提供；命令行参数便于在设备主机上使用。
calibration_file = os.path.abspath(os.path.expanduser(
    args.calibration or os.path.join(data_root, "calibration", "base_from_phantom.json")
))
gripper_calibration_file = os.path.abspath(os.path.expanduser(
    args.gripper_calibration
    or os.path.join(data_root, "calibration", "gripper_calibration.json")
))

# False固定初始朝向，True跟随标定后的手部朝向。
use_rotation = False


required_files = [action_file, video_file, urdf_file, calibration_file, gripper_calibration_file]
if not args.live_camera:
    required_files += [depth_file, camera_intrinsics_file]
for path in required_files:
    if not os.path.isfile(path):
        raise FileNotFoundError("找不到文件：" + path)
os.makedirs(os.path.dirname(hdf5_file), exist_ok=True)
# 输出命名与离线版相同，避免覆盖已有结果。
if os.path.exists(hdf5_file):
    raise FileExistsError("输出文件已存在，请先备份或移动：" + hdf5_file)


def valid_rotation(rotation):
    return (np.isfinite(rotation).all()
            and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
            and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3))


# 按原版读取动作与原视频帧号。
with np.load(action_file) as action_data:
    ee_pts = action_data["ee_pts"].astype(np.float32)
    ee_oris = action_data["ee_oris"].astype(np.float32)
    ee_widths = action_data["ee_widths"].astype(np.float32).reshape(-1, 1)
    frame_ids = action_data["union_indices"].astype(np.int64)
num_samples = len(ee_pts)

# 【真机修改】读取固定夹爪标定
with open(gripper_calibration_file, "r", encoding="utf-8") as file:
    gripper_calibration = json.load(file)

# valid字段可省略；存在时须为true。
if "valid" in gripper_calibration and gripper_calibration["valid"] is not True:
    raise ValueError("夹爪标定文件标记为无效")

hand_width_min = float(gripper_calibration["phantom_closed_width"])
hand_width_max = float(gripper_calibration["phantom_open_width"])
gripper_min = float(gripper_calibration["piper_closed_width"])
gripper_max = float(gripper_calibration["piper_open_width"])

if not np.isfinite([hand_width_min, hand_width_max, gripper_min, gripper_max]).all():
    raise ValueError("夹爪标定值必须是有限数值")
if hand_width_max <= hand_width_min:
    raise ValueError("Phantom夹爪标定范围无效")
if not 0.0 <= gripper_min < gripper_max <= 0.07:
    raise ValueError("PiPER夹爪标定范围必须在0～0.07 m内")

if num_samples == 0:
    raise ValueError("动作数据为空")
for name, array, shape in (
    ("ee_pts", ee_pts, (num_samples, 3)),
    ("ee_oris", ee_oris, (num_samples, 3, 3)),
    ("ee_widths", ee_widths, (num_samples, 1)),
    ("union_indices", frame_ids, (num_samples,)),
):
    if array.shape != shape:
        raise ValueError(f"{name}形状不正确：{array.shape}")

print("动作数：", num_samples)
print("帧号范围：", int(frame_ids[0]), "到", int(frame_ids[-1]))


# 原视频只提供动作来源及原始时间；实时模式不读取其中的图像。
video = cv2.VideoCapture(video_file)
if not video.isOpened():
    raise ValueError("无法打开视频：" + video_file)

video_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
image_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
image_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
camera_fps = float(video.get(cv2.CAP_PROP_FPS))

if camera_fps <= 0:
    camera_fps = 30.0
if frame_ids.min() < 0 or frame_ids.max() >= video_frame_count:
    video.release()
    raise ValueError("动作帧号超出了视频范围")

source_timestamps = frame_ids.astype(np.float64) / camera_fps
if args.live_camera:
    video.release()
else:
    rgb = np.zeros((num_samples, image_height, image_width, 3), dtype=np.uint8)

    for i in range(num_samples):
        video.set(cv2.CAP_PROP_POS_FRAMES, int(frame_ids[i]))
        read_success, bgr_image = video.read()
        if not read_success:
            video.release()
            raise ValueError("读取视频第" + str(int(frame_ids[i])) + "帧失败")
        rgb[i] = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    video.release()


    # 深度与原视频按同一组frame_ids对齐。
    source_depth = np.load(depth_file, mmap_mode="r")

    if source_depth.shape != (video_frame_count, image_height, image_width):
        raise ValueError("深度帧数或图像尺寸与视频不同")

    depth = np.array(source_depth[frame_ids], dtype=np.float32)
    depth = depth * depth_scale
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0.0] = 0.0


    # 读取相机内参
    with open(camera_intrinsics_file, "r", encoding="utf-8") as file:
        camera_data = json.load(file)

    left_camera = camera_data["left"]
    camera_intrinsics = np.array(
        [
            [left_camera["fx"], 0.0, left_camera["cx"]],
            [0.0, left_camera["fy"], left_camera["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )



# 【真机修改】读取实测标定：p_base = R @ p_phantom + t，位置和平移均为米。
with open(calibration_file, "r", encoding="utf-8") as file:
    calibration_data = json.load(file)

if "valid" in calibration_data and calibration_data["valid"] is not True:
    raise ValueError("坐标标定文件标记为无效，禁止用于真机")

base_from_phantom = np.array(calibration_data["base_from_phantom"], dtype=np.float64)
if base_from_phantom.shape != (4, 4) or not np.isfinite(base_from_phantom).all():
    raise ValueError("base_from_phantom必须是包含有限数值的4×4矩阵")
if (not np.allclose(base_from_phantom[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
        or not valid_rotation(base_from_phantom[:3, :3])):
    raise ValueError("标定矩阵的末行应为[0, 0, 0, 1]，旋转部分须正交且行列式为1")

T_base_from_phantom = torch.tensor(base_from_phantom, dtype=torch.float64)

# 【真机修改】初始关节角取自真机；离线时使用原来的q0。
if use_real_robot:
    from piper_ros_io import PiperRosIO

    robot_io = PiperRosIO(
        allow_send=False,
        max_joint_step=max_joint_step,
        gripper_min=gripper_min,
        gripper_max=gripper_max,
        command_topic=args.command_topic,
        feedback_topic=args.feedback_topic,
        status_topic=args.status_topic,
        node_name=args.ros_node_name,
        feedback_max_age=args.feedback_max_age,
    )
    first_feedback = robot_io.wait_feedback(timeout=5.0)
    initial_joint_pos = first_feedback["joint_pos"]
    robot_start_time = float(first_feedback["robot_timestamp"])
    if not np.isfinite(robot_start_time):
        raise ValueError("初始反馈时间无效")
else:
    robot_io = None
    initial_joint_pos = np.array([0.0, 1.5, -1.0, 0.0, 0.8, 0.0], dtype=np.float64)
    robot_start_time = 0.0

initial_joint_pos = np.asarray(initial_joint_pos, dtype=np.float64)
if initial_joint_pos.shape != (6,) or not np.isfinite(initial_joint_pos).all():
    raise ValueError("初始关节角必须是6个有限数值")
ik = PiperIK(urdf_file, q0=initial_joint_pos.tolist(), jump=0.3)
initial_joint_tensor = torch.tensor(initial_joint_pos, dtype=torch.float64)
initial_ee_pose = ik.fk(initial_joint_tensor)[0]

camera_io = None
if args.live_camera:
    from piper_camera_io import PiperCameraIO

    camera_io = PiperCameraIO(robot_io, args.rgb_topic, args.depth_topic,
                              args.camera_info_topic, args.live_depth_scale,
                              args.sync_tolerance, args.camera_max_age)
    episode_start_time = robot_io.ros_time()
    try:
        first_image = camera_io.read()
    except BaseException:
        camera_io.close()
        raise
    image_height, image_width = first_image["depth"].shape
    camera_intrinsics = first_image["intrinsics"]
    camera_fps = args.camera_fps
    rgb = np.zeros((num_samples, image_height, image_width, 3), dtype=np.uint8)
    depth = np.zeros((num_samples, image_height, image_width), dtype=np.float32)

if send_to_robot:
    if input("确认真机配置及急停可用，输入SEND开始发送：").strip() != "SEND":
        if camera_io is not None:
            camera_io.close()
        raise RuntimeError("已取消发送")
    robot_io.allow_send = True


# actions每帧保存[q1,q2,q3,q4,q5,q6,gripper]
# IK失败的帧保留为NaN，不能把它发送给真实机械臂
actions = np.full((num_samples, 7), np.nan, dtype=np.float32)

# 保存真正传给IK的目标位置和目标旋转
target_ee_pos = np.full((num_samples, 3), np.nan, dtype=np.float32)
target_ee_rot = np.full((num_samples, 3, 3), np.nan, dtype=np.float32)

# 保存每帧是否有效、IK是否成功以及失败原因
frame_valid = np.zeros(num_samples, dtype=np.uint8)
ik_success = np.zeros(num_samples, dtype=np.uint8)
# 0～6表示输入或IK错误；通信、执行错误另记termination_reason。
failure_code = np.zeros(num_samples, dtype=np.int8)
# 【真机修改】新增处理标志，区分中止后尚未处理的帧。
frame_processed = np.zeros(num_samples, dtype=np.uint8)

# 保存每一帧的FK检查结果
fk_position_error = np.full(num_samples, np.nan, dtype=np.float32)
fk_rotation_error = np.full(num_samples, np.nan, dtype=np.float32)

# 【真机修改】这些数组现在保存真实命令状态和编码器反馈
command_sent = np.zeros(num_samples, dtype=np.uint8)
execution_success = np.zeros(num_samples, dtype=np.uint8)
robot_error_code = np.full(num_samples, -1, dtype=np.int32)
robot_timestamps = np.full(num_samples, np.nan, dtype=np.float64)
joint_pos = np.full((num_samples, 6), np.nan, dtype=np.float32)
joint_vel = np.full((num_samples, 6), np.nan, dtype=np.float32)
gripper_width = np.full((num_samples, 1), np.nan, dtype=np.float32)
ee_pos = np.full((num_samples, 3), np.nan, dtype=np.float32)
ee_rot = np.full((num_samples, 3, 3), np.nan, dtype=np.float32)

# Action Processor已经过滤了没有检测到手的帧
hand_detected = np.ones(num_samples, dtype=np.uint8)
hand_bbox = np.full((num_samples, 4), np.nan, dtype=np.float32)

# 原视频时间
timestamps = (np.full(num_samples, np.nan, dtype=np.float64)
              if args.live_camera else source_timestamps.copy())
camera_timestamps = timestamps.copy()
# 机器人时间相对于第一条反馈，与原视频使用不同的时间起点。


# 【真机修改】读取、校验并保存一条真实反馈，供IK和到位检查共用。
def read_robot_feedback(index, feedback=None, record=True):
    if feedback is None:
        feedback = robot_io.get_feedback()
    if feedback is None:
        raise RuntimeError("没有收到真机反馈")
    # 先记录驱动错误码，再由接口检查数值、新鲜度和控制状态。
    robot_error_code[index] = feedback["robot_error_code"]
    robot_io.check_feedback(feedback, require_control=send_to_robot)
    received_time = feedback["feedback_receive_time"]
    q = np.asarray(feedback["joint_pos"], dtype=np.float64)
    if np.any(q < ik.q_min.numpy()) or np.any(q > ik.q_max.numpy()):
        raise ValueError("真机反馈超出URDF关节限位")
    if not record:
        return feedback, received_time
    joint_pos[index] = q
    joint_vel[index] = feedback["joint_vel"]
    gripper_width[index, 0] = feedback["gripper_width"]
    robot_timestamps[index] = (feedback["sample_timestamp"] - episode_start_time
                               if args.live_camera else float(feedback["robot_timestamp"]) - robot_start_time)
    if not np.isfinite(robot_timestamps[index]) or robot_timestamps[index] < 0:
        raise ValueError("反馈时间无效或发生回退")
    pose = ik.fk(torch.tensor(q, dtype=torch.float64))[0]
    ee_pos[index] = pose[:3, 3].numpy()
    ee_rot[index] = pose[:3, :3].numpy()

    return feedback, received_time


def wait_execution(index, target_q, target_width, sent_time):
    # 到位检查保留错误记录；实时模式不覆盖执行前的观测。
    while time.monotonic() - sent_time < command_timeout:
        final_feedback, received_time = read_robot_feedback(index, record=not args.live_camera)
        joint_reached = np.max(np.abs(final_feedback["joint_pos"] - target_q)) <= joint_tolerance
        gripper_reached = abs(final_feedback["gripper_width"] - target_width) <= gripper_tolerance
        if received_time > sent_time and joint_reached and gripper_reached:
            return True
        time.sleep(0.02)
    return False


stop_reason = ""
stage = "runtime_error"
try:
    for i in range(num_samples):
        loop_start_time = time.monotonic()
        frame_processed[i] = 1

        # 每次IK前，读取真实编码器反馈并覆盖上一帧的理论解。
        if use_real_robot:
            stage = "robot_feedback_or_status_invalid"
            if args.live_camera:
                stage = "camera_sync_failed"
                sample = camera_io.read()
                if (sample["rgb"].shape != rgb.shape[1:]
                        or not np.allclose(sample["intrinsics"], camera_intrinsics)):
                    raise ValueError("采集期间相机分辨率或内参发生变化")
                feedback, _ = read_robot_feedback(i, feedback=sample["feedback"])
                rgb[i], depth[i] = sample["rgb"], sample["depth"]
                timestamps[i] = camera_timestamps[i] = sample["camera_timestamp"] - episode_start_time
            else:
                feedback, _ = read_robot_feedback(i)
            ik.q_old = torch.tensor(feedback["joint_pos"], dtype=torch.float64)
        stage = "ik_processing"

        current_ee_pts = torch.tensor(ee_pts[i], dtype=torch.float64)
        current_ee_oris = torch.tensor(ee_oris[i], dtype=torch.float64)
        current_ee_width = float(ee_widths[i, 0])
        input_error = ""
        if not torch.isfinite(current_ee_pts).all():
            input_error = "位置含NaN或Inf"
        elif not valid_rotation(current_ee_oris.numpy()):
            input_error = "姿态不是有效的旋转矩阵"
        elif not np.isfinite(current_ee_width):
            input_error = "夹爪宽度含NaN或Inf"
        if input_error:
            failure_code[i] = 6
            print("第", i, "帧输入无效:", input_error)
            if send_to_robot:
                stop_reason = "invalid_input"
                break
            continue

        # 【真机修改】左乘实测标定，将Phantom末端位姿转换到base_link。
        # 手部末端的轴定义须与IK末端link6约定一致；这里不额外缩放或平移轨迹。
        phantom_ee_pose = torch.eye(4, dtype=torch.float64)
        phantom_ee_pose[:3, :3] = current_ee_oris
        phantom_ee_pose[:3, 3] = current_ee_pts
        target_ee_pose = T_base_from_phantom @ phantom_ee_pose
        if not use_rotation:
            # 保留原设置：位置使用标定结果，朝向固定为初始真机朝向。
            target_ee_pose[:3, :3] = initial_ee_pose[:3, :3]
        target_ee_pos[i] = target_ee_pose[:3, 3].numpy()
        target_ee_rot[i] = target_ee_pose[:3, :3].numpy()

        try:
            solved_joint_pos, success, reason = ik.solve(target_ee_pose)
        except (RuntimeError, ValueError) as error:
            success, reason = False, str(error)
        if success and (solved_joint_pos.shape != (6,) or not torch.isfinite(solved_joint_pos).all()):
            success, reason = False, "IK返回的关节角形状或数值无效"
        if not success:
            reason = str(reason)
            if "跳变" in reason:
                failure_code[i] = 3
            elif "限位" in reason:
                failure_code[i] = 2
            elif "不可达" in reason:
                failure_code[i] = 1
            else:
                failure_code[i] = 4
            print("第", i, "帧IK失败:", reason)
            if send_to_robot:
                stop_reason = "ik_failed"
                break
            continue

        # 保留原版FK复核，超差解禁止发送。
        fk_ee_pose = ik.fk(solved_joint_pos)[0]
        position_error = torch.linalg.norm(target_ee_pose[:3, 3] - fk_ee_pose[:3, 3])
        rotation_difference = target_ee_pose[:3, :3] @ fk_ee_pose[:3, :3].T
        rotation_value = (torch.trace(rotation_difference) - 1.0) / 2.0
        rotation_error = torch.acos(torch.clamp(rotation_value, -1.0, 1.0))
        fk_position_error[i] = float(position_error)
        fk_rotation_error[i] = float(rotation_error)
        if (
            not np.isfinite([float(position_error), float(rotation_error)]).all()
            or float(position_error) >= 0.005
            or float(rotation_error) >= 0.05
        ):
            failure_code[i] = 4
            if send_to_robot:
                stop_reason = "fk_check_failed"
                break
            continue

        # 按实测开合范围映射夹爪宽度；超出Phantom标定范围时截断到端点。
        gripper_ratio = (current_ee_width - hand_width_min) / (hand_width_max - hand_width_min)
        gripper_ratio = np.clip(gripper_ratio, 0.0, 1.0)
        target_gripper_width = gripper_min + gripper_ratio * (gripper_max - gripper_min)
        solved_joint_array = solved_joint_pos.numpy().astype(np.float32)
        actions[i, :6] = solved_joint_array
        actions[i, 6] = np.float32(target_gripper_width)
        ik_success[i] = 1
        failure_code[i] = 0

        if not send_to_robot:
            frame_valid[i] = np.uint8(not use_real_robot)
            print("第", i, "帧IK成功，未发送命令")
            if use_real_robot:
                steps = np.abs(solved_joint_array - feedback["joint_pos"])
                joint_index = int(np.argmax(steps))
                if steps[joint_index] > max_joint_step:
                    print(f"joint{joint_index + 1}距目标{steps[joint_index]:.4f} rad，"
                          f"超过单次发送限制{max_joint_step:.4f} rad；请核对标定及起始姿态")
            continue

        stage = "command_rejected"
        if args.live_camera:
            # IK计算过久时不再执行基于旧观测的动作。
            age = time.monotonic() - sample["camera_receive_time"]
            if not 0 <= age <= robot_io.feedback_max_age:
                raise RuntimeError("执行前观测已过期，请检查IK耗时")
        sent_time = time.monotonic()
        robot_io.send_joint_command(
            solved_joint_array, target_gripper_width, speed_percent=speed_percent,
        )
        command_sent[i] = 1
        stage = "execution_feedback_or_status_invalid"
        reached = wait_execution(i, solved_joint_array, target_gripper_width, sent_time)
        if not reached:
            stop_reason = "execution_timeout"
            print("第", i, "帧执行超时")
            break

        # IK成功只设置ik_success；真机还需确认关节和夹爪到位。
        frame_valid[i] = 1
        execution_success[i] = 1
        print("第", i, "帧执行成功")
        # control_hz=10.0 是最大发送频率；到位等待可能使实际频率更低。
        wait_time = 1.0 / control_hz - (time.monotonic() - loop_start_time)
        if wait_time > 0:
            time.sleep(wait_time)
except KeyboardInterrupt:
    stop_reason = "user_interrupted"
    print("用户中止，停止发送后续帧并保存已有反馈")
except Exception as error:
    stop_reason = stage
    print("运行停止，保存已有反馈:", repr(error))
finally:
    # 仅停止本采集进程继续发布；不会撤回最后的目标或代替硬件急停。
    if camera_io is not None:
        camera_io.close()
    if robot_io is not None:
        robot_io.close()


# 保留FK误差和接近限位统计。
successful_frames = ik_success == 1
if np.any(successful_frames):
    for name, errors, unit in (("位置", fk_position_error, "m"), ("旋转", fk_rotation_error, "rad")):
        errors = errors[successful_frames]
        print(f"FK{name}误差：平均{errors.mean():.6f}，最大{errors.max():.6f} {unit}")

joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
joint_lower_limits = ik.q_min.numpy()
joint_upper_limits = ik.q_max.numpy()
limit_margin = 0.05
q = actions[successful_frames, :6]
near_limit_count = np.any(
    (q - joint_lower_limits < limit_margin) | (joint_upper_limits - q < limit_margin), axis=1,
).sum()
print("接近关节限位的帧数：", int(near_limit_count))

# 【真机修改】试算不计为执行成功，中止原因与IK错误码分别记录。
episode_success = bool(send_to_robot and not stop_reason and np.all(execution_success == 1))
if stop_reason:
    termination_reason = stop_reason
elif not use_real_robot:
    termination_reason = "offline_ik_only"
elif not send_to_robot:
    termination_reason = "dry_run_no_command"
else:
    termination_reason = "completed" if episode_success else "incomplete"

# 按piper_hdf5_schema.md中的目录和字段保存HDF5
with h5py.File(hdf5_file, "x") as hdf5:
    meta = hdf5.create_group("meta")
    meta.attrs.update({
        "format_version": "1.1",
        "robot_name": "PiPER",
        "num_episodes": np.int64(1),
        "total_samples": np.int64(num_samples),
        "action_type": "joint_position",
        "action_source": "ik_retargeted",
        "joint_unit": "rad",
        "position_unit": "m",
        "gripper_unit": "m",
        "time_unit": "s",
        "time_base": "episode_start_ros2" if args.live_camera else "separate_video_and_robot_start",
        "capture_mode": "live_rgbd_robot_replay" if args.live_camera else "recorded_video_robot_replay",
        "ros_distribution": "humble",
        "ros_version": np.int64(2),
        "command_topic": args.command_topic,
        "feedback_topic": args.feedback_topic,
        "status_topic": args.status_topic,
    })

    robot = hdf5.create_group("robot")
    text = h5py.string_dtype(encoding="utf-8")
    robot.create_dataset("joint_names", data=joint_names, dtype=text)
    robot.create_dataset("joint_lower_limits", data=joint_lower_limits)
    robot.create_dataset("joint_upper_limits", data=joint_upper_limits)
    robot.attrs.update({
        "urdf_path": urdf_file,
        "end_link": "link6",
        "gripper_min": np.float64(gripper_min),
        "gripper_max": np.float64(gripper_max),
    })

    calibration = hdf5.create_group("calibration")
    calibration.create_dataset("camera_intrinsics", data=camera_intrinsics)
    calibration.create_dataset("T_base_from_phantom", data=base_from_phantom)
    calibration.attrs["T_base_from_phantom_valid"] = np.uint8(1)
    calibration.attrs["use_rotation"] = np.uint8(use_rotation)
    for name in ("phantom_closed_width", "phantom_open_width", "piper_closed_width", "piper_open_width"):
        calibration.attrs[name] = np.float64(gripper_calibration[name])

    raw_episode = hdf5.create_group("raw/" + episode_group)
    for name, values in {"frame_ids": frame_ids, "timestamps": source_timestamps,
                         "hand_detected": hand_detected, "hand_bbox": hand_bbox}.items():
        raw_episode.create_dataset(name, data=values)

    raw_ee_pts = raw_episode.create_dataset("ee_pts", data=ee_pts)
    raw_ee_pts.attrs["coordinate_frame"] = "phantom"
    raw_ee_pts.attrs["unit"] = "m"

    raw_ee_oris = raw_episode.create_dataset("ee_oris", data=ee_oris)
    raw_ee_oris.attrs["coordinate_frame"] = "phantom"
    raw_ee_oris.attrs["unit"] = "dimensionless"

    raw_ee_widths = raw_episode.create_dataset("ee_widths", data=ee_widths)
    raw_ee_widths.attrs["unit"] = "m"

    episode = hdf5.create_group("data/" + episode_group)
    for name, values in {
        "actions": actions, "timestamps": timestamps, "camera_timestamps": camera_timestamps,
        "robot_timestamps": robot_timestamps, "frame_ids": frame_ids,
        "frame_valid": frame_valid, "frame_processed": frame_processed,
        "ik_success": ik_success, "command_sent": command_sent,
        "execution_success": execution_success, "failure_code": failure_code,
        "robot_error_code": robot_error_code,
    }.items():
        episode.create_dataset(name, data=values)

    target_ee_pos_data = episode.create_dataset("target_ee_pos", data=target_ee_pos)
    target_ee_pos_data.attrs["unit"] = "m"
    target_ee_pos_data.attrs["coordinate_frame"] = "base_link"

    target_ee_rot_data = episode.create_dataset("target_ee_rot", data=target_ee_rot)
    target_ee_rot_data.attrs["unit"] = "dimensionless"
    target_ee_rot_data.attrs["coordinate_frame"] = "base_link"

    # 【真机修改】保存编码器反馈
    obs = episode.create_group("obs")
    for name, values in {"joint_pos": joint_pos, "joint_vel": joint_vel,
                         "gripper_width": gripper_width, "ee_pos": ee_pos, "ee_rot": ee_rot}.items():
        obs.create_dataset(name, data=values)
    obs.create_dataset("rgb", data=rgb, compression="gzip")

    depth_set = obs.create_dataset(
        "depth",
        data=depth,
        compression="gzip",
        chunks=(1, image_height, image_width),
    )
    depth_set.attrs.update({
        "unit": "m",
        "coordinate_frame": "camera",
        "invalid_value": np.float32(0.0),
        "alignment": "ros2_header_timestamps" if args.live_camera else "frame_ids",
        "source_frame_count": np.int64(camera_io.depth_count if args.live_camera else source_depth.shape[0]),
    })

    obs.attrs.update({
        "robot_feedback_available": np.uint8(np.isfinite(joint_pos).any()),
        "image_source": "live_camera" if args.live_camera else "recorded_demonstration",
        "images_synchronized_with_robot": np.uint8(args.live_camera and np.isfinite(camera_timestamps).any()),
        "rgb_format": "RGB",
    })

    episode.attrs.update({
        "num_samples": np.int64(num_samples),
        "source_video": "video_L.mp4",
        "target_hand": target_hand,
        "valid_trajectory": np.uint8(np.all(frame_valid == 1)),
        "start_frame": np.int64(frame_ids[0]),
        "end_frame": np.int64(frame_ids[-1]),
        "success": np.uint8(episode_success),
        "termination_reason": termination_reason,
        "task_name": task_name,
        "control_hz": np.float64(control_hz),
        "camera_fps": np.float64(camera_fps),
    })


print("HDF5保存完成:", hdf5_file)
print("处理帧数：", num_samples)
print("IK成功:", int(ik_success.sum()))
print("IK失败:", int(num_samples - ik_success.sum()))

print("实际执行成功:", int(execution_success.sum()))
print("结束原因:", termination_reason)
if stop_reason:
    sys.exit(1)
