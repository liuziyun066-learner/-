import math
import xml.etree.ElementTree as ET
import torch

def rpy_to_R(rpy):
    r, p, y = rpy
    sr, cr = math.sin(r), math.cos(r)
    sp, cp = math.sin(p), math.cos(p)
    sy, cy = math.sin(y), math.cos(y)
    rx = torch.tensor([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=torch.float64)
    ry = torch.tensor([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=torch.float64)
    rz = torch.tensor([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=torch.float64)
    return rz @ ry @ rx

def rotate(axis, angle):
    axis = axis / torch.linalg.norm(axis)
    x, y, z = axis
    K = torch.stack([
        torch.stack([x * 0, -z, y]),
        torch.stack([z, x * 0, -x]),
        torch.stack([-y, x, x * 0]),
    ])
    R = (
        torch.eye(3, dtype=torch.float64)
        + torch.sin(angle) * K
        + (1 - torch.cos(angle)) * (K @ K)
    )
    T = torch.eye(4, dtype=torch.float64)
    T[:3, :3] = R
    return T


def rotation_vector(R):
    """旋转矩阵的轴角向量，接近180度时单独求旋转轴。"""
    skew = torch.stack((R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1])) / 2
    sine = torch.linalg.norm(skew)
    cosine = torch.clamp((torch.trace(R) - 1) / 2, -1.0, 1.0)
    angle = torch.atan2(sine, cosine)
    if angle < 1e-6:
        return skew
    if math.pi - angle < 1e-4:
        _, vectors = torch.linalg.eigh((R + R.T) / 2)
        axis = vectors[:, -1]
        if torch.dot(axis, skew) < 0:
            axis = -axis
        return angle * axis
    return angle * skew / sine


class PiperIK:
    def __init__(self, urdf_path, q0=None, jump=0.3):
        self.fixed_T, self.axis, self.q_min, self.q_max = self.read_urdf(urdf_path)
        self.jump = jump
        if q0 is None:
            q0 = [0.0] * 6
        self.q_old = torch.tensor(q0, dtype=torch.float64)
        if self.q_old.shape != (6,) or not torch.isfinite(self.q_old).all():
            raise ValueError("初始关节角必须包含6个有限数值")
        if torch.any(self.q_old < self.q_min) or torch.any(self.q_old > self.q_max):
            raise ValueError("初始关节角超出限位")

    def read_urdf(self, urdf_path):
        """读取 6 个关节。"""
        root = ET.parse(urdf_path).getroot()
        joint_map = {}
        for joint in root.findall("joint"):
            child = joint.find("child").get("link")
            joint_map[child] = joint
        arm = []
        link = "link6"
        while link != "base_link":
            joint = joint_map[link]
            arm.append(joint)
            link = joint.find("parent").get("link")
        arm.reverse()
        if len(arm) != 6:
            raise ValueError("base_link 到 link6 必须有 6 个关节")
        fixed_T, axis, q_min, q_max = [], [], [], []
        for joint in arm:
            origin = joint.find("origin")
            xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()]
            rpy = [float(x) for x in origin.get("rpy", "0 0 0").split()]
            T = torch.eye(4, dtype=torch.float64)
            T[:3, :3] = rpy_to_R(rpy)
            T[:3, 3] = torch.tensor(xyz, dtype=torch.float64)
            fixed_T.append(T)
            a = [float(x) for x in joint.find("axis").get("xyz").split()]
            axis.append(torch.tensor(a, dtype=torch.float64))
            limit = joint.find("limit")
            q_min.append(float(limit.get("lower")))
            q_max.append(float(limit.get("upper")))
        return (
            fixed_T,
            axis,
            torch.tensor(q_min, dtype=torch.float64),
            torch.tensor(q_max, dtype=torch.float64),
        )

    def fk(self, q):
        T = torch.eye(4, dtype=torch.float64)
        pos = []
        axis_base = []
        for i in range(6):
            T = T @ self.fixed_T[i]
            pos.append(T[:3, 3].clone())
            axis_base.append(T[:3, :3] @ self.axis[i])
            T = T @ rotate(self.axis[i], q[i])
        J = torch.zeros((6, 6), dtype=torch.float64)
        end_pos = T[:3, 3]
        for i in range(6):
            J[:3, i] = torch.linalg.cross(axis_base[i], end_pos - pos[i])
            J[3:, i] = axis_base[i]
        return T, J

    def solve(self, T_goal):
        T_goal = torch.as_tensor(T_goal, dtype=torch.float64).clone()
        if T_goal.shape != (4, 4) or not torch.isfinite(T_goal).all():
            return self.q_old.clone(), False, "目标位姿必须有效"
        R = T_goal[:3, :3]
        if (not torch.allclose(T_goal[3], torch.tensor([0., 0., 0., 1.], dtype=torch.float64), atol=1e-6)
                or not torch.allclose(R.T @ R, torch.eye(3, dtype=torch.float64), atol=1e-3)
                or not torch.isclose(torch.det(R), torch.tensor(1., dtype=torch.float64), atol=1e-3)):
            return self.q_old.clone(), False, "目标位姿必须是刚体变换"
        if self.q_old.shape != (6,) or not torch.isfinite(self.q_old).all():
            return self.q_old.clone(), False, "当前关节角无效"
        if torch.any(self.q_old < self.q_min) or torch.any(self.q_old > self.q_max):
            return self.q_old.clone(), False, "当前关节角超出限位"
        q = self.q_old.clone()
        limit_hit = False
        for _ in range(100):
            T_now, J = self.fk(q)
            e_pos = T_goal[:3, 3] - T_now[:3, 3]
            e_rot = rotation_vector(T_goal[:3, :3] @ T_now[:3, :3].T)
            e_angle = torch.linalg.norm(e_rot)
            if torch.linalg.norm(e_pos) < 0.005 and e_angle < 0.05:
                if torch.any(torch.abs(q - self.q_old) > self.jump):
                    return self.q_old.clone(), False, "关节角跳变过大"
                self.q_old = q.clone()
                return q, True, "成功"
            e = torch.cat([e_pos, 0.5 * e_rot])
            
            u, s, vh = torch.linalg.svd(J, full_matrices=False)
            s_inv = torch.zeros_like(s)
            use = s > s.max() * 1e-4
            s_inv[use] = 1.0 / s[use]
            dq = vh.T @ torch.diag(s_inv) @ u.T @ e
            dq = torch.clamp(dq, -0.1, 0.1)
            q_new = q + 0.5 * dq
            q_safe = torch.clamp(q_new, self.q_min, self.q_max)
            if not torch.allclose(q_new, q_safe):
                limit_hit = True
            q = q_safe
        if limit_hit:
            return self.q_old.clone(), False, "目标不可达或受到关节限位"
        return self.q_old.clone(), False, "目标不可达"
