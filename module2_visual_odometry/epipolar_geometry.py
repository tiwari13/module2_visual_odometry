#!/usr/bin/env python3
"""
PHASE 2 — Step 5: Epipolar Geometry & Essential Matrix

Theory recap:
  - Two views of the same scene are related by the Essential Matrix E
  - For calibrated cameras: x2.T @ E @ x1 = 0  (epipolar constraint)
  - E encodes rotation R and translation t between the two views
  - RANSAC robustly estimates E despite outlier feature matches
  - cv2.recoverPose() gives R and t (translation up to scale — monocular limit)

What this node does:
  1. Detects ORB features in consecutive camera frames
  2. Matches features with ratio test (Lowe's test)
  3. Computes Essential Matrix via RANSAC
  4. Recovers R, t from E using cheirality check
  5. Accumulates a trajectory and compares against PX4 ground truth
  6. Generates a report plot on Ctrl+C

Run:
  ros2 run module2_visual_odometry epipolar_geometry

Fly the drone in a slow circle or figure-8 to collect a trajectory.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from px4_msgs.msg import VehicleLocalPosition
from cv_bridge import CvBridge
import cv2
import numpy as np
import time


# ── Camera intrinsics from Module 1 calibration ──────────────────────────────
K = np.array([
    [1397.22,    0.0,  960.0],
    [   0.0, 1397.22,  540.0],
    [   0.0,    0.0,    1.0]
], dtype=np.float64)


class EpipolarGeometry(Node):

    def __init__(self):
        super().__init__('epipolar_geometry')

        self.bridge = CvBridge()

        # ORB: fast, binary descriptor — good for real-time VO
        self.detector = cv2.ORB_create(nfeatures=2000)
        # knnMatch needs k=2 for Lowe's ratio test
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        # Previous frame state
        self.prev_gray = None
        self.prev_kp   = None
        self.prev_desc = None

        # Accumulated pose (relative to first frame)
        self.R_accum   = np.eye(3)
        self.t_accum   = np.zeros((3, 1))

        # Trajectory storage
        self.estimated_traj = [np.zeros(3)]   # list of (3,) arrays
        self.gt_traj        = []               # list of (3,) NED arrays
        self.gt_origin      = None

        # Per-frame statistics
        self.pose_log = []   # dicts with inliers, ratio, R, t, time
        self.frame_count = 0

        # ── Subscriptions ────────────────────────────────────────────────────
        cam_topic = (
            '/world/default/model/x500_skydio_0/model'
            '/camera_front/link/camera_link/sensor/IMX214/image'
        )
        self.create_subscription(Image, cam_topic, self._image_cb, 10)

        self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position_v1',
            self._gt_cb,
            rclpy.qos.qos_profile_sensor_data,
        )

        self.create_timer(5.0, self._status_cb)

        self.get_logger().info("=" * 60)
        self.get_logger().info("PHASE 2  Step 5: Epipolar Geometry")
        self.get_logger().info("=" * 60)
        self.get_logger().info("Fly a slow circle or figure-8")
        self.get_logger().info("Ctrl+C → full report + trajectory plot")

    # ── Ground truth ─────────────────────────────────────────────────────────
    def _gt_cb(self, msg):
        pos = np.array([msg.x, msg.y, msg.z])
        if self.gt_origin is None:
            self.gt_origin = pos.copy()
        self.gt_traj.append(pos - self.gt_origin)

    # ── Main image callback ──────────────────────────────────────────────────
    def _image_cb(self, msg):
        bgr  = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        kp, desc = self.detector.detectAndCompute(gray, None)

        if desc is not None and len(kp) >= 10 and self.prev_desc is not None:
            self._process_pair(gray, kp, desc, msg.header.stamp)

        self.prev_gray = gray
        self.prev_kp   = kp
        self.prev_desc = desc
        self.frame_count += 1

    # ── Core epipolar pipeline ───────────────────────────────────────────────
    def _process_pair(self, gray, kp, desc, stamp):
        # ── Step 1: Match with Lowe's ratio test ─────────────────────────────
        raw = self.matcher.knnMatch(desc, self.prev_desc, k=2)
        good = []
        for pair in raw:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:   # Lowe's threshold
                    good.append(m)

        if len(good) < 8:
            return   # 8-point algorithm needs at least 8 correspondences

        pts_curr = np.float32([kp[m.queryIdx].pt          for m in good])
        pts_prev = np.float32([self.prev_kp[m.trainIdx].pt for m in good])

        # ── Step 2: Essential Matrix via RANSAC ───────────────────────────────
        # RANSAC iteratively fits E and counts inliers (points satisfying
        # the epipolar constraint x2.T @ E @ x1 = 0 within `threshold` px)
        E, mask = cv2.findEssentialMat(
            pts_curr, pts_prev, K,
            method=cv2.RANSAC,
            prob=0.999,       # confidence that result is outlier-free
            threshold=1.0,    # max epipolar distance in pixels to be inlier
        )

        if E is None or mask is None:
            return

        n_inliers     = int(mask.sum())
        inlier_ratio  = n_inliers / len(good)

        if n_inliers < 8 or inlier_ratio < 0.3:
            return   # too noisy to trust

        # ── Step 3: Recover R, t from E ──────────────────────────────────────
        # recoverPose picks the correct one of 4 (R,t) solutions using the
        # cheirality check: reconstructed points must be in front of both cameras
        _, R, t, _ = cv2.recoverPose(E, pts_curr, pts_prev, K, mask=mask)

        # ── Step 4: Accumulate trajectory ─────────────────────────────────────
        # Compose: world_R_cam  = prev_world_R_cam  @ frame_delta_R
        #          world_t_cam += world_R_cam @ frame_delta_t
        # Note: |t| = 1 always (monocular scale ambiguity — fixed in VIO)
        self.R_accum  = R @ self.R_accum
        self.t_accum += self.R_accum @ t
        self.estimated_traj.append(self.t_accum.flatten().copy())

        # ── Log ───────────────────────────────────────────────────────────────
        rvec    = cv2.Rodrigues(R)[0].flatten()
        ts      = stamp.sec + stamp.nanosec * 1e-9

        self.pose_log.append({
            'time':         ts,
            'R':            R.copy(),
            't':            t.copy(),
            'inliers':      n_inliers,
            'total':        len(good),
            'inlier_ratio': inlier_ratio,
        })

        self.get_logger().info(
            f"Frame {self.frame_count:4d} | "
            f"Matches: {len(good):3d} | "
            f"Inliers: {n_inliers:3d} ({inlier_ratio*100:4.0f}%) | "
            f"R(deg): [{np.degrees(rvec[0]):5.1f}, "
            f"{np.degrees(rvec[1]):5.1f}, "
            f"{np.degrees(rvec[2]):5.1f}] | "
            f"t(unit): [{t[0,0]:5.2f}, {t[1,0]:5.2f}, {t[2,0]:5.2f}]"
        )

    # ── Periodic status ──────────────────────────────────────────────────────
    def _status_cb(self):
        if not self.pose_log:
            self.get_logger().info("Waiting for motion (move the drone)...")
            return
        recent        = self.pose_log[-20:]
        avg_inliers   = np.mean([p['inliers']      for p in recent])
        avg_ratio     = np.mean([p['inlier_ratio']  for p in recent])
        self.get_logger().info(
            f"[STATUS] Poses: {len(self.pose_log)} | "
            f"Avg inliers: {avg_inliers:.0f} | "
            f"Avg ratio: {avg_ratio*100:.0f}% | "
            f"GT pts: {len(self.gt_traj)}"
        )

    # ── Final report ─────────────────────────────────────────────────────────
    def generate_report(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        print("\n" + "=" * 70)
        print("PHASE 2 — Step 5: EPIPOLAR GEOMETRY REPORT")
        print("=" * 70)

        if not self.pose_log:
            print("No pose data collected — did the drone move?")
            return

        inliers = [p['inliers']      for p in self.pose_log]
        ratios  = [p['inlier_ratio']  for p in self.pose_log]
        times   = [p['time'] - self.pose_log[0]['time'] for p in self.pose_log]

        print(f"\n  Frames processed:    {self.frame_count}")
        print(f"  Pose estimates:      {len(self.pose_log)}")
        print(f"\n  Inlier statistics:")
        print(f"    Mean inliers:      {np.mean(inliers):.1f}")
        print(f"    Mean inlier ratio: {np.mean(ratios)*100:.1f}%")
        print(f"    Min inliers:       {np.min(inliers)}")

        quality = np.mean(ratios)
        if quality > 0.70:
            verdict = "EXCELLENT — ready to build VO on top of this"
        elif quality > 0.50:
            verdict = "GOOD — sufficient for Visual Odometry"
        else:
            verdict = "MODERATE — fly slower or use a more textured environment"
        print(f"\n  Quality: {verdict}")

        print(f"\n  Key insight:")
        print(f"    t magnitude is always 1 (monocular scale ambiguity).")
        print(f"    The DIRECTION of t is correct; scale comes from IMU in Step 9.")

        # ── Plot ──────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1. Trajectory (top-down XZ view)
        ax = axes[0]
        if len(self.estimated_traj) > 1:
            traj = np.array(self.estimated_traj)
            ax.plot(traj[:, 0], traj[:, 2], 'b-', lw=1.5, label='Epipolar VO (unit scale)')
            ax.scatter(*traj[0, [0, 2]],  c='g', s=100, zorder=5, label='Start')
            ax.scatter(*traj[-1, [0, 2]], c='r', s=100, marker='x', zorder=5, label='End')
        if len(self.gt_traj) > 1:
            gt = np.array(self.gt_traj)
            ax.plot(gt[:, 0], gt[:, 1], 'r--', lw=1.5, alpha=0.7, label='GT (NED X/Y)')
        ax.set_xlabel('X'); ax.set_ylabel('Z')
        ax.set_title('Trajectory (top-down)\nNote: VO scale is arbitrary')
        ax.legend(fontsize=8); ax.grid(True); ax.set_aspect('equal')

        # 2. Inlier ratio over time
        ax2 = axes[1]
        ax2.plot(times, [r * 100 for r in ratios], 'g-', lw=1.5)
        ax2.axhline(50, color='r', ls='--', alpha=0.5, label='50% floor')
        ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Inlier Ratio (%)')
        ax2.set_title('RANSAC Quality Over Time')
        ax2.legend(); ax2.grid(True); ax2.set_ylim(0, 105)

        # 3. Inlier count distribution
        ax3 = axes[2]
        ax3.hist(inliers, bins=30, color='steelblue', edgecolor='white')
        ax3.axvline(np.mean(inliers), color='r', ls='--', label=f'Mean={np.mean(inliers):.0f}')
        ax3.set_xlabel('Inliers per frame')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Inlier Distribution\n(higher = cleaner geometry)')
        ax3.legend(); ax3.grid(True)

        plt.suptitle('Phase 2 — Step 5: Epipolar Geometry', fontsize=14, fontweight='bold')
        plt.tight_layout()
        out = 'epipolar_geometry_step5.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"\n  Saved: {out}")
        print("=" * 70 + "\n")


def main(args=None):
    rclpy.init(args=args)
    node = EpipolarGeometry()
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
