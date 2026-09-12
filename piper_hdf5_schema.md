# PiPER 数据集 HDF5 字段与格式规范

## 1. 统一约定

| 关节角单位 | 弧度 `rad` |
| 位置单位 | 米 `m` |
| 夹爪宽度单位 | 米 `m` |
| 时间单位 | 秒 `s` |
| 机械臂基坐标系 | `base_link` |
| 机械臂末端坐标系 | `link6` |
| 图像格式 | RGB，`uint8`，范围 `0～255` |
| 深度格式 | `float32`，单位 `m`，`0` 表示无效深度 |
| 动作顺序 | `[q1,q2,q3,q4,q5,q6,gripper]` |

## 2. HDF5 完整目录结构

```text
piper_dataset.hdf5
├── meta
├── robot
│   ├── joint_names
│   ├── joint_lower_limits
│   └── joint_upper_limits
├── calibration
│   ├── camera_intrinsics
│   └── T_base_from_phantom
├── raw
│   └── episode_0000
│       ├── frame_ids
│       ├── timestamps
│       ├── hand_detected
│       ├── hand_bbox
│       ├── ee_pts
│       ├── ee_oris
│       └── ee_widths
└── data
    └── episode_0000
        ├── actions
        ├── timestamps
        ├── camera_timestamps
        ├── robot_timestamps
        ├── frame_ids
        ├── frame_valid
        ├── ik_success
        ├── command_sent
        ├── execution_success
        ├── failure_code
        ├── robot_error_code
        ├── target_ee_pos
        ├── target_ee_rot
        ├── obs
           ├── joint_pos
           ├── joint_vel
           ├── gripper_width
           ├── ee_pos
           ├── ee_rot
           ├── rgb
           └── depth
```

## 3. `meta`
| 属性 | 类型 | 示例 |
|---|---|---|
| `format_version` | UTF-8 | `"1.1"`（包含深度） |
| `robot_name` | UTF-8 | `"PiPER"` |
| `num_episodes` | `int64` | episode 数量 |
| `total_samples` | `int64` | 总帧数 |
| `action_type` | UTF-8 | `"joint_position"` |
| `action_source` | UTF-8 | `"ik_retargeted"` |
| `joint_unit` | UTF-8 | `"rad"` |
| `position_unit` | UTF-8 | `"m"` |
| `gripper_unit` | UTF-8 | `"m"` |
| `time_unit` | UTF-8 | `"s"` |
| `time_base` | UTF-8 | `"episode_start"` |

## 4. `robot`：PiPER 机械臂信息
以下假设机械臂有6个joint

| 字段 | 形状 | 类型 | 单位 |
|---|---:|---|---|
| `joint_names` | `(6,)` | UTF-8 字符串 | 无 |
| `joint_lower_limits` | `(6,)` | `float64` | rad |
| `joint_upper_limits` | `(6,)` | `float64` | rad |

`joint_names` 内容：
```text
["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
```

`robot` Group 建议增加以下属性：

| 属性 | 类型 | 含义 |
|---|---:|---|
| `urdf_path` | UTF-8 字符串 | 生成数据时使用的 URDF |
| `end_link` | UTF-8 字符串 | `"link6"` |
| `gripper_min` | `float64` | 夹爪最小开口，单位 m |
| `gripper_max` | `float64` | 夹爪最大开口，单位 m |

## 5. `calibration`：标定数据

| 字段 | 形状 | 类型 | 单位 |
|---|---:|---|
| `camera_intrinsics` | `(3,3)` | `float64` | 像素 | 
| `T_base_from_phantom` | `(4,4)` | `float64` | 平移单位为 m |
- 旋转矩阵，无单位。
- 平移矩阵的单位为m。

## 6. `raw/episode_XXXX`：Action Processor 原始输出

假设原视频或处理结果共有 `N` 帧。

| 字段 | 形状 | 类型 | 单位/格式 |
|---|---:|---|---|
| `frame_ids` | `(N,)` | `int64` | 原视频帧号 |
| `timestamps` | `(N,)` | `float64` | s |
| `hand_detected` | `(N,)` | `uint8` | `0` 或 `1` |
| `hand_bbox` | `(N,4)` | `float32` | 像素，`[x1,y1,x2,y2]` |
| `ee_pts` | `(N,3)` | `float32` | Phantom 原始单位 | 
| `ee_oris` | `(N,3,3)` | `float32` | 旋转矩阵，无单位 | 
| `ee_widths` | `(N,1)` | `float32` | Phantom 原始单位 | 

原始数据不应直接当作 PiPER `base_link` 坐标。为 `ee_pts` 添加属性：

```text
coordinate_frame = "phantom"
unit = "unknown"
```
coordinate_frame表示一组位置或姿态数据是相对于哪个坐标系描述的。

确认 Phantom 的位置单位后，才能把 `unit` 改成真实单位。

## 7. `data/episode_XXXX`：最终轨迹

### 7.1 `actions`

```text
shape = (N, 7)
dtype = float32
```
每帧数据：
```text
[q1, q2, q3, q4, q5, q6, gripper]
```
| 部分 | 形状 | 单位 |
|---|---:|---|---|
| `q1～q6` | `(N,6)` | rad |
| `gripper` | `(N,1)` | m |

### 7.2 `target_ee_pos`
它表示坐标转换后、真正传给 IK 的目标末端位置。
```text
shape = (N, 3)
dtype = float32
unit = m
coordinate_frame = base_link
```

每帧数据：

```text
[x, y, z]
```

### 7.3 `target_ee_rot`
它表示坐标转换后、真正传给 IK 的目标末端姿态。
```text
shape = (N, 3, 3)
dtype = float32
unit = dimensionless
coordinate_frame = base_link
```

### 7.4 时间和帧号

| 字段 | 形状 | 类型 | 单位 |
|---|---:|---|---|
| `timestamps` | `(N,)` | `float64` | s |
| `camera_timestamps` | `(N,)` | `float64` | s，图像采集时间 |
| `robot_timestamps` | `(N,)` | `float64` | s，机械臂状态接收时间 |
| `frame_ids` | `(N,)` | `int64` | 帧号 |

### 7.5 IK 状态

| 字段 | 形状 | 类型 | 含义 |
|---|---:|---|---|
| `ik_success` | `(N,)` | `uint8` | `1` 成功，`0` 失败 |
| `failure_code` | `(N,)` | `int8` | 失败原因编号 |

错误码固定为：

| 数值 | 含义 |
|---:|---|
| `0` | 成功 |
| `1` | 目标不可达 |
| `2` | 关节超过限位 |
| `3` | 相邻帧关节跳变过大 |
| `4` | IK 数值计算失败 |
| `5` | 手部没有检测到 |
| `6` | 坐标、姿态或输入数据无效 |

### 7.6 真实执行状态

| 字段 | 形状 | 类型 | 含义 |
|---|---:|---|---|
| `frame_valid` | `(N,)` | `uint8` | 当前帧整体是否有效 |
| `command_sent` | `(N,)` | `uint8` | 该动作是否发送给机械臂 |
| `execution_success` | `(N,)` | `uint8` | 机械臂是否成功执行 |
| `robot_error_code` | `(N,)` | `int32` | PiPER 驱动返回的错误码 |

## 8. `obs`：当前帧观测

| 字段 | 形状 | 类型 | 单位/格式 | 含义 |
|---|---:|---|---|---|
| `joint_pos` | `(N,6)` | `float32` | rad | 六个关节当前角度 |
| `joint_vel` | `(N,6)` | `float32` | rad/s | 关节速度 |
| `gripper_width` | `(N,1)` | `float32` | m | 夹爪当前开口宽度 |
| `ee_pos` | `(N,3)` | `float32` | m，`base_link` | 末端当前三维位置 |
| `ee_rot` | `(N,3,3)` | `float32` | 旋转矩阵 | 末端当前朝向 |
| `rgb` | `(N,H,W,3)` | `uint8` | RGB，`0～255` |
| `depth` | `(N,H,W)` | `float32` | m，`0` 表示无效深度 | 相机坐标系中的逐像素深度 |

`depth[t,y,x]` 表示第 `t` 帧图像中像素 `(x,y)` 的深度。它必须和
`rgb[t]`、`frame_ids[t]`、`actions[t]` 对应同一个时刻。若原始深度帧数
与动作数量不同，必须通过 `frame_ids` 对齐，不能直接截取前 `N` 帧。

给 `obs/depth` 保存以下属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `unit` | UTF-8 | `"m"` |
| `coordinate_frame` | UTF-8 | `"camera"` |
| `invalid_value` | `float32` | `0.0` |
| `alignment` | UTF-8 | `"frame_ids"` |
| `source_frame_count` | `int64` | 原始 `depth.npy` 的帧数 |

actions[t]
= 第t帧下发给PiPER的目标关节角和夹爪宽度

obs/joint_pos[t]
= 第t帧机械臂编码器返回的真实关节角

obs/gripper_width[t]
= 第t帧真实夹爪反馈

真实采集时，不能直接令：
obs/joint_pos = actions[:, :6]
否则无法判断机械臂是否真正执行到位。

## 9. 每个 episode 的属性

`data/episode_XXXX` 应包含以下属性：

| 属性 | 类型 | 含义 |
|---|---|---|
| `num_samples` | `int64` | 当前轨迹帧数 `N` |
| `source_video` | UTF-8 字符串 | 原视频名称 |
| `target_hand` | UTF-8 字符串 | `"right"` 或 `"left"` |
| `valid_trajectory` | `uint8` | `1` 有效，`0` 无效 |
| `start_frame` | `int64` | 原视频起始帧 |
| `end_frame` | `int64` | 原视频结束帧 |
| `success` | `uint8` | 整个任务是否成功 |
| `termination_reason` | UTF-8 字符串 | 正常结束、人工停止、碰撞等 |
| `task_name` | UTF-8 字符串 | 例如 `"pick_cup"` |
| `control_hz` | `float64` | 机械臂控制频率 |
| `camera_fps` | `float64` | 相机帧率 |
