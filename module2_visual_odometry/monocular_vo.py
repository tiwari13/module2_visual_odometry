#!/usr/bin/env python3
"""
PHASE 2 — Step 6: Monocular Visual Odometry

Improvements over Step 5 (epipolar geometry):
  1. KLT optical flow  — tracks features frame-to-frame without descriptor matching
  2. Keyframe selection — only estimate pose when enough parallax has accumulated
  3. Triangulation      — reconstruct 3D points to get relative scale between keyframes
  4. Scale normalization— divide t by median triangulated depth for scale consistency
  5. Auto re-detection  — detect new features when tracked count drops too low

Pipeline per frame:
  ┌─ Track existing features with KLT from prev frame
  ├─ Enough motion since last keyframe?
  │     YES → Essential Matrix + recoverPose (vs last keyframe)
  │         → Triangulate 3D points
  │         → Normalize t by median depth
  │         → Chain pose → update trajectory
  │         → Make current frame the new keyframe
  │     NO  → just update prev_gray / prev_pts
  └─ Feature count < MIN_FEATURES? → re-detect with Shi-Tomasi

Run:
  ros2 run module2_visual_odometry monocular_vo

Fly the drone with circle_flight while this runs.
Ctrl+C generates a full report + trajectory plot.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from px4_msgs.msg import VehicleLocalPosition
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation
import cv2
import numpy as np
import time

# ── Camera intrinsics (Module 1 calibration) ─────────────────────────────────
K = np.array([
    [1397.22,    0.0,  960.0],
    [   0.0, 1397.22,  540.0],
    [   0.0,    0.0,    1.0]
], dtype=np.float64)

# ── Tuning parameters ─────────────────────────────────────────────────────────
MAX_FEATURES   = 500    # Shi-Tomasi corners to detect
MIN_FEATURES   = 150    # re-detect when tracked count falls below this
MIN_FLOW       = 8.0    # px — mean optical flow to trigger a new keyframe
MIN_INLIER_R   = 0.4    # RANSAC inlier ratio floor to accept a pose estimate
MIN_MATCHES    = 12     # minimum correspondences to attempt E matrix

# LK optical flow parameters
LK_PARAMS = dict(
    winSize   = (21, 21),
    maxLevel  = 3,
    criteria  = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

# Shi-Tomasi corner detection parameters
ST_PARAMS = dict(
    maxCorners   = MAX_FEATURES,
    qualityLevel = 0.01,
    minDistance  = 15,
    blockSize    = 7,
)


class MonocularVO(Node):

    def __init__(self):
        super().__init__('monocular_vo')
        self.bridge = CvBridge()

        # ── Per-frame tracking state ──────────────────────────────────────────
        self.prev_gray = None        # last frame (grayscale)
        self.prev_pts  = None        # tracked points in prev_gray

        # ── Keyframe state ────────────────────────────────────────────────────
        self.kf_gray   = None        # last keyframe image
        self.kf_pts    = None        # feature positions IN the keyframe

        # ── Global pose (world ← camera) ──────────────────────────────────────
        self.R_world   = np.eye(3)
        self.t_world   = np.zeros((3, 1))

        # ── Trajectory storage ────────────────────────────────────────────────
        self.trajectory    = [np.zeros(3)]   # estimated positions
        self.gt_trajectory = []              # PX4 ground truth
        self.gt_origin     = None

        # ── Statistics ────────────────────────────────────────────────────────
        self.keyframe_log = []   # {time, n_pts, inlier_ratio, scale, n_map_pts}
        self.frame_count  = 0
        self.kf_count     = 0
        self.start_time   = time.time()

        # ── Publisher for Step 8 EKF VIO ─────────────────────────────────────
        self.pose_pub = self.create_publisher(
            PoseStamped, '/monocular_vo/pose', 10
        )

        # ── Subscriptions ─────────────────────────────────────────────────────
        cam_topic = (
            '/world/default/model/x500_skydio_0/model'
            '/camera_front/link/camera_link/sensor/IMX214/image'
        )
        self.create_subscription(Image, cam_topic, self._image_cb, 10)

        sensor_qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position_v1',
            self._gt_cb, sensor_qos,
        )

        self.create_timer(5.0, self._status_cb)

        self.get_logger().info("=" * 60)
        self.get_logger().info("PHASE 2  Step 6: Monocular Visual Odometry")
        self.get_logger().info("=" * 60)
        self.get_logger().info("Using KLT tracking + keyframe selection")
        self.get_logger().info("Fly a circle with circle_flight, then Ctrl+C")

    # ── Ground truth ──────────────────────────────────────────────────────────
    def _gt_cb(self, msg):
        pos = np.array([msg.x, msg.y, msg.z])
        if self.gt_origin is None:
            self.gt_origin = pos.copy()
        self.gt_trajectory.append(pos - self.gt_origin)

    # ── Main image callback ───────────────────────────────────────────────────
    def _image_cb(self, msg):
        gray = cv2.cvtColor(
            self.bridge.imgmsg_to_cv2(msg, 'bgr8'),
            cv2.COLOR_BGR2GRAY
        )
        self.frame_count += 1

        # ── First frame: initialise ───────────────────────────────────────────
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts  = self._detect(gray)
            self.kf_gray   = gray
            self.kf_pts    = self.prev_pts.copy()
            return

        # ── Step 1: KLT tracking ──────────────────────────────────────────────
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **LK_PARAMS
        )

        # Keep only successfully tracked points
        if curr_pts is None or status is None:
            self.prev_gray = gray
            self.prev_pts  = self._detect(gray)
            self.kf_gray   = gray
            self.kf_pts    = self.prev_pts.copy()
            return

        ok           = status.ravel() == 1
        curr_pts_ok  = curr_pts[ok]
        prev_pts_ok  = self.prev_pts[ok]

        if len(curr_pts_ok) < MIN_MATCHES:
            # Lost tracking — re-initialise
            self.prev_gray = gray
            self.prev_pts  = self._detect(gray)
            self.kf_gray   = gray
            self.kf_pts    = self.prev_pts.copy()
            return

        # ── Step 2: measure motion since last keyframe ────────────────────────
        # Re-track keyframe features into the current frame
        kf_curr, kf_status, _ = cv2.calcOpticalFlowPyrLK(
            self.kf_gray, gray, self.kf_pts, None, **LK_PARAMS
        )

        if kf_curr is not None and kf_status is not None:
            kf_ok      = kf_status.ravel() == 1
            kf_curr_ok = kf_curr[kf_ok]
            kf_prev_ok = self.kf_pts[kf_ok]
            mean_flow  = float(np.mean(
                np.linalg.norm(kf_curr_ok - kf_prev_ok, axis=1)
            )) if len(kf_curr_ok) > 0 else 0.0
        else:
            mean_flow, kf_curr_ok, kf_prev_ok = 0.0, np.zeros((0,2)), np.zeros((0,2))

        # ── Step 3: new keyframe? ─────────────────────────────────────────────
        if mean_flow >= MIN_FLOW and len(kf_curr_ok) >= MIN_MATCHES:
            self._process_keyframe(gray, kf_curr_ok, kf_prev_ok)

        # ── Step 4: re-detect if features running low ─────────────────────────
        if len(curr_pts_ok) < MIN_FEATURES:
            self.prev_pts = self._detect(gray)
            self.kf_gray  = gray
            self.kf_pts   = self.prev_pts.copy()
        else:
            self.prev_pts = curr_pts_ok.reshape(-1, 1, 2)

        self.prev_gray = gray

    # ── Process a keyframe pair ───────────────────────────────────────────────
    def _process_keyframe(self, gray, curr_pts, prev_pts):
        """
        curr_pts — tracked positions of keyframe features in the current frame
        prev_pts — those same features' positions in the keyframe
        """
        # ── Essential Matrix (RANSAC) ─────────────────────────────────────────
        E, mask = cv2.findEssentialMat(
            curr_pts, prev_pts, K,
            method=cv2.RANSAC, prob=0.999, threshold=1.0,
        )
        if E is None or mask is None:
            return

        n_inliers    = int(mask.sum())
        inlier_ratio = n_inliers / len(curr_pts)

        if inlier_ratio < MIN_INLIER_R or n_inliers < MIN_MATCHES:
            return

        # ── Recover R, t ──────────────────────────────────────────────────────
        _, R, t, pose_mask = cv2.recoverPose(
            E, curr_pts, prev_pts, K, mask=mask
        )

        # ── Triangulate to get scale ──────────────────────────────────────────
        # Projection matrices: P1 = K[I|0],  P2 = K[R|t]
        P1 = K @ np.hstack([np.eye(3),    np.zeros((3, 1))])
        P2 = K @ np.hstack([R,             t              ])

        inlier_curr = curr_pts[mask.ravel() == 255].T   # (2, N)
        inlier_prev = prev_pts[mask.ravel() == 255].T

        if inlier_curr.shape[1] >= 4:
            pts4d  = cv2.triangulatePoints(P1, P2, inlier_prev, inlier_curr)
            pts3d  = pts4d[:3] / (pts4d[3] + 1e-8)        # (3, N) homogeneous → 3D
            depths = pts3d[2, pts3d[2] > 0.1]              # positive depths only
            scale  = float(np.median(depths)) if len(depths) > 3 else 1.0
        else:
            scale = 1.0

        # ── Update global pose ────────────────────────────────────────────────
        # Scale normalizes translation to metric-like units (relative scale)
        if scale > 0.01:
            t_scaled = t / scale
        else:
            t_scaled = t

        # Compose:  t_world += R_world @ t_scaled
        #           R_world  = R @ R_world
        self.t_world = self.t_world + self.R_world @ t_scaled
        self.R_world = R @ self.R_world

        self.trajectory.append(self.t_world.flatten().copy())

        # ── Log ───────────────────────────────────────────────────────────────
        self.kf_count += 1
        self.keyframe_log.append({
            'time'        : time.time() - self.start_time,
            'frame'       : self.frame_count,
            'n_pts'       : len(curr_pts),
            'n_inliers'   : n_inliers,
            'inlier_ratio': inlier_ratio,
            'scale'       : scale,
        })

        rvec = cv2.Rodrigues(R)[0].flatten()
        self.get_logger().info(
            f"KF {self.kf_count:3d} | "
            f"Pts: {len(curr_pts):3d} | "
            f"Inliers: {n_inliers:3d} ({inlier_ratio*100:.0f}%) | "
            f"Scale: {scale:6.2f} | "
            f"R(deg): [{np.degrees(rvec[0]):5.1f}, "
            f"{np.degrees(rvec[1]):5.1f}, "
            f"{np.degrees(rvec[2]):5.1f}] | "
            f"t: [{self.t_world[0,0]:6.2f}, "
            f"{self.t_world[1,0]:6.2f}, "
            f"{self.t_world[2,0]:6.2f}]"
        )

        # ── Publish pose for Step 8 EKF VIO ──────────────────────────────────
        pose_msg = PoseStamped()
        pose_msg.header.stamp    = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'camera'
        pose_msg.pose.position.x = float(self.t_world[0, 0])
        pose_msg.pose.position.y = float(self.t_world[1, 0])
        pose_msg.pose.position.z = float(self.t_world[2, 0])
        q = Rotation.from_matrix(self.R_world).as_quat()   # [x,y,z,w]
        pose_msg.pose.orientation.x = float(q[0])
        pose_msg.pose.orientation.y = float(q[1])
        pose_msg.pose.orientation.z = float(q[2])
        pose_msg.pose.orientation.w = float(q[3])
        self.pose_pub.publish(pose_msg)

        # ── New keyframe ──────────────────────────────────────────────────────
        self.kf_gray = gray
        self.kf_pts  = curr_pts.reshape(-1, 1, 2)

    # ── Feature detection ─────────────────────────────────────────────────────
    def _detect(self, gray):
        """Shi-Tomasi corner detection — better than ORB for KLT tracking."""
        pts = cv2.goodFeaturesToTrack(gray, **ST_PARAMS)
        if pts is None:
            return np.zeros((0, 1, 2), dtype=np.float32)
        return pts

    # ── Status timer ──────────────────────────────────────────────────────────
    def _status_cb(self):
        elapsed = time.time() - self.start_time
        pos     = self.t_world.flatten()
        if self.keyframe_log:
            recent_ratio = np.mean(
                [k['inlier_ratio'] for k in self.keyframe_log[-10:]]
            )
            self.get_logger().info(
                f"[{elapsed:5.1f}s] Frames: {self.frame_count} | "
                f"Keyframes: {self.kf_count} | "
                f"Pos: [{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}] | "
                f"Inlier ratio (recent): {recent_ratio*100:.0f}% | "
                f"GT pts: {len(self.gt_trajectory)}"
            )
        else:
            self.get_logger().info(
                f"[{elapsed:5.1f}s] Waiting for keyframes "
                f"(move the drone, need >{MIN_FLOW}px mean flow)..."
            )

    # ── Report ────────────────────────────────────────────────────────────────
    def generate_report(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        print("\n" + "=" * 70)
        print("PHASE 2 — Step 6: MONOCULAR VO REPORT")
        print("=" * 70)

        if not self.keyframe_log:
            print("No keyframes processed.")
            return

        elapsed  = time.time() - self.start_time
        ratios   = [k['inlier_ratio']  for k in self.keyframe_log]
        scales   = [k['scale']         for k in self.keyframe_log]
        times    = [k['time']          for k in self.keyframe_log]
        n_pts    = [k['n_pts']         for k in self.keyframe_log]

        print(f"\n  Runtime:             {elapsed:.1f}s")
        print(f"  Frames processed:    {self.frame_count}")
        print(f"  Keyframes selected:  {self.kf_count}")
        print(f"  KF rate:             {self.kf_count/elapsed:.2f} KF/s")
        print(f"\n  Inlier ratio:        {np.mean(ratios)*100:.1f}% avg")
        print(f"  Scale consistency:   std={np.std(scales):.3f}  "
              f"(lower = more consistent)")
        print(f"  Features tracked:    {np.mean(n_pts):.0f} avg per KF")

        traj = np.array(self.trajectory)
        if len(traj) > 1:
            total_dist = np.sum(
                np.linalg.norm(np.diff(traj, axis=0), axis=1)
            )
            print(f"  Path length (VO):    {total_dist:.2f} units")

        # Scale consistency assessment
        scale_std = np.std(scales)
        if scale_std < 1.0:
            verdict = "EXCELLENT — scale very consistent between keyframes"
        elif scale_std < 3.0:
            verdict = "GOOD — minor scale drift, acceptable for short flights"
        else:
            verdict = "MODERATE — scale drift present, IMU will fix this in Step 9"
        print(f"\n  Assessment: {verdict}")

        print(f"\n  Key insight:")
        print(f"    Scale is RELATIVE — each keyframe normalizes by median depth.")
        print(f"    Drift still accumulates over time. IMU integration (Step 9)")
        print(f"    provides absolute metric scale from accelerometer integration.")

        # ── Plots ─────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Trajectory top-down
        ax = axes[0, 0]
        if len(traj) > 1:
            ax.plot(traj[:, 0], traj[:, 1], 'b-', lw=1.5, label='Monocular VO')
            ax.scatter(*traj[0, :2],  c='g', s=100, zorder=5, label='Start')
            ax.scatter(*traj[-1, :2], c='r', s=100, marker='x', zorder=5, label='End')
        if len(self.gt_trajectory) > 1:
            gt = np.array(self.gt_trajectory)
            ax.plot(gt[:, 0], gt[:, 1], 'r--', lw=1.5, alpha=0.7, label='GT (NED)')
        ax.set_xlabel('X'); ax.set_ylabel('Y')
        ax.set_title('Trajectory (top-down)\nVO scale is relative to GT')
        ax.legend(fontsize=8); ax.grid(True); ax.set_aspect('equal')

        # 2. Inlier ratio over time
        ax2 = axes[0, 1]
        ax2.plot(times, [r*100 for r in ratios], 'g-', lw=1.5)
        ax2.axhline(40, color='r', ls='--', alpha=0.5, label='40% floor')
        ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Inlier Ratio (%)')
        ax2.set_title('RANSAC Quality Over Time')
        ax2.legend(); ax2.grid(True); ax2.set_ylim(0, 105)

        # 3. Scale over time (consistency check)
        ax3 = axes[1, 0]
        ax3.plot(times, scales, 'm-', lw=1.5)
        ax3.axhline(np.mean(scales), color='k', ls='--',
                    label=f'Mean={np.mean(scales):.2f}')
        ax3.fill_between(times,
                         np.mean(scales) - np.std(scales),
                         np.mean(scales) + np.std(scales),
                         alpha=0.2, color='m', label=f'±1σ={np.std(scales):.2f}')
        ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Triangulated Scale')
        ax3.set_title('Scale Consistency\n(flat = good; IMU fixes drift)')
        ax3.legend(); ax3.grid(True)

        # 4. Feature count over time
        ax4 = axes[1, 1]
        ax4.plot(times, n_pts, 'steelblue', lw=1.5)
        ax4.axhline(MIN_FEATURES, color='r', ls='--',
                    label=f'Re-detect threshold ({MIN_FEATURES})')
        ax4.set_xlabel('Time (s)'); ax4.set_ylabel('Tracked Features')
        ax4.set_title('Feature Count per Keyframe')
        ax4.legend(); ax4.grid(True)

        plt.suptitle('Phase 2 — Step 6: Monocular Visual Odometry',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        out = 'monocular_vo_step6.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"\n  Saved: {out}")
        print("=" * 70 + "\n")


def main(args=None):
    rclpy.init(args=args)
    node = MonocularVO()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nGenerating report...")
        node.generate_report()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
