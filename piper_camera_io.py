"""ROS 2 Humble：读取 RGB、对齐到 RGB 的深度并匹配 PiPER 反馈。"""

from collections import deque
import threading
import time

import numpy as np
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class PiperCameraIO:
    def __init__(
        self,
        robot_io,
        rgb_topic,
        depth_topic,
        info_topic,
        depth_scale,
        tolerance=0.03,
        max_age=0.5,
    ):
        if (
            not np.isfinite([depth_scale, tolerance, max_age]).all()
            or min(depth_scale, tolerance, max_age) <= 0
        ):
            raise ValueError("深度换算系数、同步容差和消息时限必须大于0")
        self.robot_io = robot_io
        self.node = robot_io.node
        self.rgb_topic = str(rgb_topic)
        self.depth_topic = str(depth_topic)
        self.info_topic = str(info_topic)
        self.depth_scale = float(depth_scale)
        self.tolerance = float(tolerance)
        self.max_age = float(max_age)
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.rgb = deque(maxlen=60)
        self.depth = deque(maxlen=60)
        self.info = None
        self.last_stamp = 0.0
        self.depth_count = 0
        self.subscribers = []
        self._closed = False

        robot_io.enable_sync()
        try:
            self.subscribers.append(
                self.node.create_subscription(
                    Image,
                    self.rgb_topic,
                    self._rgb,
                    qos_profile_sensor_data,
                )
            )
            self.subscribers.append(
                self.node.create_subscription(
                    Image,
                    self.depth_topic,
                    self._depth,
                    qos_profile_sensor_data,
                )
            )
            self.subscribers.append(
                self.node.create_subscription(
                    CameraInfo,
                    self.info_topic,
                    self._info,
                    qos_profile_sensor_data,
                )
            )
        except Exception:
            self.close()
            raise

    @staticmethod
    def _stamp_to_sec(msg):
        stamp = msg.header.stamp
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _rgb(self, msg):
        with self.lock:
            self.rgb.append((msg, time.monotonic()))

    def _depth(self, msg):
        with self.lock:
            self.depth.append((msg, time.monotonic()))
            self.depth_count += 1

    def _info(self, msg):
        with self.lock:
            self.info = msg

    def read(self, timeout=5.0):
        """等待调用后到达的新 RGB-D，并匹配最接近的机械臂反馈。"""
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout必须大于0")
        after_receive = time.monotonic()
        deadline = after_receive + timeout
        while time.monotonic() < deadline:
            with self.lock:
                colors = list(self.rgb)
                depths = list(self.depth)
                info = self.info
            for color, color_receive in reversed(colors):
                if color_receive <= after_receive:
                    continue
                stamp = self._stamp_to_sec(color)
                if not np.isfinite(stamp) or stamp <= self.last_stamp:
                    continue
                candidates = [
                    (depth_msg, receive_time)
                    for depth_msg, receive_time in depths
                    if receive_time > after_receive
                    and np.isfinite(self._stamp_to_sec(depth_msg))
                ]
                if not candidates or info is None:
                    continue
                depth, depth_receive = min(
                    candidates,
                    key=lambda item: abs(self._stamp_to_sec(item[0]) - stamp),
                )
                depth_stamp = self._stamp_to_sec(depth)
                feedback = self.robot_io.feedback_near(stamp, self.tolerance)
                if feedback is None:
                    continue
                sample_times = [stamp, depth_stamp, feedback["sample_timestamp"]]
                if max(sample_times) - min(sample_times) > self.tolerance:
                    continue
                if time.monotonic() - max(color_receive, depth_receive) > self.max_age:
                    continue

                self.robot_io.check_feedback(feedback)
                result = self._convert(color, depth, info)
                result.update(
                    camera_timestamp=stamp,
                    depth_timestamp=depth_stamp,
                    camera_receive_time=max(color_receive, depth_receive),
                    feedback=feedback,
                )
                self.last_stamp = max(stamp, depth_stamp)
                return result
            time.sleep(0.005)
        raise RuntimeError(
            "ROS 2相机同步超时：请检查RGB、aligned_depth_to_color、CameraInfo话题，"
            "并确认相机与PiPER消息使用同一ROS时钟"
        )

    def _convert(self, color, depth, info):
        if (
            not color.header.frame_id
            or color.header.frame_id != depth.header.frame_id
            or color.header.frame_id != info.header.frame_id
        ):
            raise ValueError("RGB、对齐深度和CameraInfo必须属于同一相机光学坐标系")
        if depth.encoding not in ("16UC1", "32FC1"):
            raise ValueError("深度编码必须是16UC1或32FC1")
        if color.encoding not in ("rgb8", "bgr8", "rgba8", "bgra8"):
            raise ValueError("RGB话题必须提供8位彩色图像")

        rgb = np.array(
            self.bridge.imgmsg_to_cv2(color, "rgb8"),
            dtype=np.uint8,
            copy=True,
        )
        depth_array = np.array(
            self.bridge.imgmsg_to_cv2(depth, "passthrough"),
            dtype=np.float32,
            copy=True,
        )
        if (
            rgb.shape != (info.height, info.width, 3)
            or depth_array.shape != rgb.shape[:2]
        ):
            raise ValueError("RGB、对齐深度和CameraInfo的尺寸不同")
        if (
            info.binning_x > 1
            or info.binning_y > 1
            or info.roi.x_offset
            or info.roi.y_offset
        ):
            raise ValueError("请使用无裁剪、无像素合并的相机输出")

        # 原始 color/image_raw 使用 K；image_rect* 使用 P 的左侧 3x3。
        if "image_rect" in self.rgb_topic:
            intrinsics = (
                np.asarray(info.p, dtype=np.float64)
                .reshape(3, 4)[:, :3]
                .copy()
            )
        else:
            intrinsics = np.asarray(info.k, dtype=np.float64).reshape(3, 3).copy()
        if (
            not np.isfinite(intrinsics).all()
            or min(intrinsics[0, 0], intrinsics[1, 1]) <= 0
            or not np.allclose(intrinsics[2], [0, 0, 1])
        ):
            raise ValueError("CameraInfo内参矩阵无效")

        depth_array *= self.depth_scale
        depth_array[~np.isfinite(depth_array) | (depth_array < 0)] = 0
        return dict(rgb=rgb, depth=depth_array, intrinsics=intrinsics)

    def close(self):
        if self._closed:
            return
        self._closed = True
        for subscriber in self.subscribers:
            self.node.destroy_subscription(subscriber)
        self.subscribers.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
