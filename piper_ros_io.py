"""ROS 2 Humble 下的 PiPER 机械臂命令与反馈接口。"""

from collections import deque
import os
import threading
import time

import numpy as np
import rclpy
from piper_msgs.msg import PiperStatusMsg
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState


class PiperRosIO:
    """读取 PiPER 反馈并通过 ROS 2 发送绝对关节位置命令。

    远端 ``piper_ctrl_single_node`` 中 ``joint_ctrl_single`` 是 MIT 控制
    输入，而 ``joint_position_ctrl`` 才是绝对位置输入。本类默认使用后者，
    避免把轨迹回放误发到 MIT 控制通道。
    """

    JOINT_NAMES = [
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper",
    ]

    def __init__(
        self,
        allow_send=False,
        max_joint_step=0.05,
        gripper_min=0.0,
        gripper_max=0.07,
        command_topic="/joint_position_ctrl",
        feedback_topic="/joint_states_single",
        status_topic="/arm_status",
        node_name="piper_hdf5_ros_io",
        feedback_max_age=0.5,
        sync_history_size=400,
    ):
        self.allow_send = bool(allow_send)
        self.max_joint_step = float(max_joint_step)
        self.gripper_min = float(gripper_min)
        self.gripper_max = float(gripper_max)
        self.feedback_max_age = float(feedback_max_age)
        self.command_topic = str(command_topic)
        self.feedback_topic = str(feedback_topic)
        self.status_topic = str(status_topic)
        if (
            not np.isfinite(
                [max_joint_step, gripper_min, gripper_max, feedback_max_age]
            ).all()
            or max_joint_step <= 0
            or feedback_max_age <= 0
            or not 0 <= gripper_min < gripper_max
        ):
            raise ValueError("步长、反馈时限或夹爪范围无效")
        if int(sync_history_size) < 2:
            raise ValueError("sync_history_size至少为2")

        # 单调时钟仅用于新鲜度检查，ROS 时间用于机械臂和相机采样时间匹配。
        self._lock = threading.Lock()
        self._joint_sample = None
        self._joint_history = deque(maxlen=int(sync_history_size))
        self._status = None
        self._closed = False
        self.last_command = None

        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init(args=None)

        # 使用 PID 避免同一主机上并行试算时产生完全相同的节点名。
        safe_node_name = f"{node_name}_{os.getpid()}"
        self.node = rclpy.create_node(safe_node_name)
        command_qos = QoSProfile(depth=1)
        command_qos.reliability = ReliabilityPolicy.RELIABLE
        feedback_qos = QoSProfile(depth=10)
        feedback_qos.reliability = ReliabilityPolicy.RELIABLE

        self.command_pub = self.node.create_publisher(
            JointState, self.command_topic, command_qos,
        )
        self.joint_sub = self.node.create_subscription(
            JointState, self.feedback_topic, self._joint_callback, feedback_qos,
        )
        self.status_sub = self.node.create_subscription(
            PiperStatusMsg, self.status_topic, self._status_callback, feedback_qos,
        )

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            name="piper_ros2_executor",
            daemon=True,
        )
        self._spin_thread.start()

    @staticmethod
    def _stamp_to_sec(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def ros_time(self):
        """返回当前 ROS 2 时钟，单位秒。"""
        self._ensure_open()
        return self.node.get_clock().now().nanoseconds * 1e-9

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("PiperRosIO已经关闭")
        if not rclpy.ok():
            raise RuntimeError("ROS 2已经停止")

    def _joint_callback(self, msg):
        receive_monotonic = time.monotonic()
        receive_ros = self.node.get_clock().now().nanoseconds * 1e-9
        if len(msg.position) < 7:
            return
        message_stamp = self._stamp_to_sec(msg.header.stamp)
        # PiPER ROS 2 驱动会写入设备反馈时间。若驱动仍给出零时间戳，
        # 使用本节点收到消息的 ROS 时间兜底并显式标记。
        stamp_from_header = bool(np.isfinite(message_stamp) and message_stamp > 0)
        sample_timestamp = message_stamp if stamp_from_header else receive_ros
        sample = {
            "joint_pos": np.asarray(msg.position[:6], dtype=np.float64).copy(),
            "joint_vel": (
                np.asarray(msg.velocity[:6], dtype=np.float64).copy()
                if len(msg.velocity) >= 6
                else np.full(6, np.nan, dtype=np.float64)
            ),
            "gripper_width": float(msg.position[6]),
            "robot_timestamp": float(sample_timestamp),
            "sample_timestamp": float(sample_timestamp),
            "ros_receive_timestamp": float(receive_ros),
            "feedback_receive_time": float(receive_monotonic),
            "stamp_from_header": stamp_from_header,
        }
        with self._lock:
            self._joint_sample = sample
            self._joint_history.append(sample)

    def _status_callback(self, msg):
        status = {
            "robot_error_code": int(msg.err_code),
            "arm_status": int(msg.arm_status),
            "ctrl_mode": int(msg.ctrl_mode),
            "status_receive_time": time.monotonic(),
        }
        with self._lock:
            self._status = status

    @staticmethod
    def _copy_sample(sample):
        result = dict(sample)
        result["joint_pos"] = sample["joint_pos"].copy()
        result["joint_vel"] = sample["joint_vel"].copy()
        return result

    def _with_latest_status(self, sample):
        if sample is None:
            return None
        result = self._copy_sample(sample)
        with self._lock:
            status = None if self._status is None else dict(self._status)
        if status is None:
            result.update(
                robot_error_code=-1,
                arm_status=-1,
                ctrl_mode=-1,
                status_receive_time=0.0,
            )
        else:
            result.update(status)
        return result

    def get_feedback(self):
        """返回最新反馈快照；不会在读取时伪造新的采样时间。"""
        with self._lock:
            sample = self._joint_sample
        return self._with_latest_status(sample)

    def enable_sync(self):
        """开始为相机同步保留新的反馈历史。"""
        self._ensure_open()
        with self._lock:
            self._joint_history.clear()

    def feedback_near(self, sample_timestamp, tolerance):
        """返回 ROS 采样时间最接近 ``sample_timestamp`` 的反馈。"""
        if not np.isfinite([sample_timestamp, tolerance]).all() or tolerance <= 0:
            raise ValueError("采样时间和同步容差无效")
        with self._lock:
            history = list(self._joint_history)
        if not history:
            return None
        sample = min(
            history,
            key=lambda item: abs(item["sample_timestamp"] - sample_timestamp),
        )
        if abs(sample["sample_timestamp"] - sample_timestamp) > tolerance:
            return None
        return self._with_latest_status(sample)

    def check_feedback(self, feedback, require_control=False):
        self._ensure_open()
        if feedback is None:
            raise RuntimeError("没有收到完整关节反馈")
        q = np.asarray(feedback["joint_pos"], dtype=np.float64)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("关节反馈必须包含6个有限数值")
        if not np.isfinite(feedback["gripper_width"]):
            raise ValueError("夹爪反馈无效")

        now = time.monotonic()
        for key, label in (
            ("feedback_receive_time", "关节"),
            ("status_receive_time", "状态"),
        ):
            receive_time = float(feedback.get(key, 0.0))
            age = now - receive_time
            if not np.isfinite(age) or not 0 <= age <= self.feedback_max_age:
                raise RuntimeError(f"{label}反馈已超时")
        if feedback["robot_error_code"] != 0 or feedback["arm_status"] != 0:
            raise RuntimeError(
                "机械臂状态异常，错误码: " + str(feedback["robot_error_code"])
            )
        if require_control and feedback["ctrl_mode"] != 1:
            raise RuntimeError("机械臂不在CAN指令控制模式")

    def wait_feedback(self, timeout=5.0):
        """等待第一组有效的关节和状态反馈。"""
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout必须大于0")
        deadline = time.monotonic() + timeout
        last_error = "尚未收到消息"
        while rclpy.ok() and time.monotonic() < deadline:
            feedback = self.get_feedback()
            try:
                self.check_feedback(feedback)
                return feedback
            except (RuntimeError, ValueError) as error:
                last_error = str(error)
            time.sleep(0.02)
        raise RuntimeError(
            "等待PiPER ROS 2反馈超时；"
            f"feedback={self.feedback_topic}, status={self.status_topic}，最后状态：{last_error}"
        )

    def send_joint_command(
        self,
        joint_pos,
        gripper_width,
        speed_percent=10.0,
        gripper_effort=0.5,
    ):
        if not self.allow_send:
            raise RuntimeError("当前禁止发送命令")
        joint_pos = np.asarray(joint_pos, dtype=np.float64).reshape(-1)
        if joint_pos.size != 6 or not np.isfinite(joint_pos).all():
            raise ValueError("joint_pos必须包含6个有限数值")
        if (
            not np.isfinite(gripper_width)
            or not self.gripper_min <= gripper_width <= self.gripper_max
        ):
            raise ValueError("夹爪目标超出范围或数值无效")
        if not np.isfinite(speed_percent) or not 1 <= speed_percent <= 100:
            raise ValueError("speed_percent应在1到100之间")
        if not np.isfinite(gripper_effort) or gripper_effort < 0:
            raise ValueError("gripper_effort无效")

        feedback = self.get_feedback()
        self.check_feedback(feedback, require_control=True)
        if np.any(np.abs(joint_pos - feedback["joint_pos"]) > self.max_joint_step):
            raise RuntimeError("本次关节角命令跳变过大")
        if self.command_pub.get_subscription_count() == 0:
            raise RuntimeError(
                "没有ROS 2订阅者连接到PiPER绝对位置控制话题: " + self.command_topic
            )

        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = list(self.JOINT_NAMES)
        msg.position = joint_pos.tolist() + [float(gripper_width)]
        # 远端 joint_position_ctrl_callback 使用 velocity[6] 作为整臂速度百分比。
        msg.velocity = [0.0] * 6 + [float(speed_percent)]
        msg.effort = [0.0] * 6 + [float(gripper_effort)]
        self.command_pub.publish(msg)
        with self._lock:
            self.last_command = (joint_pos.copy(), float(gripper_width))
        return True

    def wait_until_reached(
        self,
        target_joint_pos,
        timeout=5.0,
        tolerance=0.03,
        target_gripper_width=None,
        gripper_tolerance=0.005,
    ):
        """用发送后的新反馈检查关节与夹爪是否到位。"""
        target_joint_pos = np.asarray(target_joint_pos, dtype=np.float64).reshape(-1)
        if target_joint_pos.size != 6 or not np.isfinite(target_joint_pos).all():
            raise ValueError("目标关节角必须包含6个有限数值")
        if (
            not np.isfinite([timeout, tolerance, gripper_tolerance]).all()
            or timeout <= 0
            or tolerance <= 0
            or gripper_tolerance <= 0
        ):
            raise ValueError("超时和到位容差必须大于0")
        if target_gripper_width is None:
            with self._lock:
                command = self.last_command
            if command is None or not np.array_equal(command[0], target_joint_pos):
                raise ValueError("请先发送该目标，或提供target_gripper_width")
            target_gripper_width = command[1]
        if (
            not np.isfinite(target_gripper_width)
            or not self.gripper_min <= target_gripper_width <= self.gripper_max
        ):
            raise ValueError("目标夹爪宽度无效")

        start_time = time.monotonic()
        last_feedback = None
        while rclpy.ok() and time.monotonic() - start_time < timeout:
            last_feedback = self.get_feedback()
            try:
                self.check_feedback(last_feedback, require_control=True)
            except (RuntimeError, ValueError):
                return False, last_feedback
            joint_reached = (
                np.max(np.abs(last_feedback["joint_pos"] - target_joint_pos))
                <= tolerance
            )
            gripper_reached = (
                abs(last_feedback["gripper_width"] - target_gripper_width)
                <= gripper_tolerance
            )
            if (
                last_feedback["feedback_receive_time"] > start_time
                and joint_reached
                and gripper_reached
            ):
                return True, last_feedback
            time.sleep(0.02)
        return False, last_feedback

    def close(self):
        """停止本进程的 ROS 2 I/O；不会发送新的运动或失能命令。"""
        if self._closed:
            return
        self.allow_send = False
        self._closed = True
        self._executor.shutdown(timeout_sec=2.0)
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
        self.node.destroy_node()
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
