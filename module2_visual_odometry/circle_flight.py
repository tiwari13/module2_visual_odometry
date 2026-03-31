#!/usr/bin/env python3
"""
Circle / Figure-8 flight for collecting Visual Odometry data.
Uses the same arming/offboard pattern as SmartNavigator.

Phases:
  1. Arm + takeoff to HEIGHT
  2. Fly to circle start
  3. Fly LAPS circles (or figure-8)
  4. Return home and land

Usage:
  ros2 run module2_visual_odometry circle_flight

Edit the constants below to tune the flight.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
)
import numpy as np
import math
import time

# ── Flight parameters ─────────────────────────────────────────────────────────
RADIUS  = 5.0    # circle radius in metres
HEIGHT  = -3.0   # NED altitude (negative = up)
OMEGA   = 0.15   # angular speed rad/s → one lap ≈ 42 s
LAPS    = 2
MODE    = 'circle'   # 'circle' or 'figure8'
POS_TOL = 0.4    # metres — close enough to waypoint


class CircleFlight(Node):

    INIT       = 'INIT'
    TAKEOFF    = 'TAKEOFF'
    TO_START   = 'TO_START'
    FLYING     = 'FLYING'
    RETURNING  = 'RETURNING'
    DONE       = 'DONE'

    def __init__(self):
        super().__init__('circle_flight')

        # Same QoS as SmartNavigator
        self.qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.offboard_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', self.qos)
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', self.qos)
        self.cmd_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', self.qos)

        # PX4 sensor topics use BEST_EFFORT + VOLATILE — must match exactly
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position_v1',
            self._pos_cb, sensor_qos,
        )

        self.pos     = None
        self.state   = self.INIT
        self.counter = 0
        self.t_fly   = 0.0
        self.total_angle = 0.0
        self.armed   = False

        # Takeoff target
        self.takeoff_pt = [0.0, 0.0, HEIGHT]

        self.create_timer(0.05, self._tick)   # 20 Hz

        self.get_logger().info("=" * 55)
        self.get_logger().info(f"Circle Flight  r={RADIUS}m  h={abs(HEIGHT)}m  ω={OMEGA} rad/s")
        self.get_logger().info(f"Mode: {MODE}  Laps: {LAPS}")
        self.get_logger().info("=" * 55)

    def _pos_cb(self, msg):
        self.pos = [msg.x, msg.y, msg.z]

    # ── Main tick ─────────────────────────────────────────────────────────────
    def _tick(self):
        self._pub_offboard()

        if self.state == self.INIT:
            self._pub_setpoint(self.takeoff_pt)
            # Send setpoints for 1s (20 ticks) before attempting mode switch
            # Re-send every 20 ticks until confirmed armed (PX4 may ignore first attempt)
            if self.counter >= 20 and self.counter % 20 == 0 and not self.armed:
                self.get_logger().info(f"Sending offboard + arm (attempt {self.counter // 20})...")
                self._send_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self._send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            if self.counter >= 20 and self.pos is not None and not self.armed:
                # Detect arming by checking if drone is responding (z starts changing)
                self.armed = True
                self.get_logger().info("Armed — taking off...")
                self.state = self.TAKEOFF
            self.counter += 1

        elif self.state == self.TAKEOFF:
            self._pub_setpoint(self.takeoff_pt)
            if self.pos and abs(self.pos[2] - HEIGHT) < POS_TOL:
                self.get_logger().info("At altitude — moving to circle start")
                self.state = self.TO_START

        elif self.state == self.TO_START:
            start = [RADIUS, 0.0, HEIGHT]
            self._pub_setpoint(start)
            if self.pos and self._dist(self.pos, start) < POS_TOL:
                self.get_logger().info(f"At start — flying {LAPS} lap(s)")
                self.t_fly = 0.0
                self.total_angle = 0.0
                self.state = self.FLYING

        elif self.state == self.FLYING:
            x, y = self._traj(self.t_fly)
            yaw  = self._heading(self.t_fly)
            self._pub_setpoint([x, y, HEIGHT], yaw=yaw)
            self.t_fly       += 0.05
            self.total_angle  = OMEGA * self.t_fly
            if self.total_angle >= LAPS * 2 * math.pi:
                self.get_logger().info("Laps done — returning home")
                self.state = self.RETURNING

        elif self.state == self.RETURNING:
            self._pub_setpoint(self.takeoff_pt)
            if self.pos and self._dist(self.pos, self.takeoff_pt) < POS_TOL:
                self.get_logger().info("Home — landing")
                self._send_cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.state = self.DONE

        elif self.state == self.DONE:
            self._pub_setpoint([0.0, 0.0, 0.0])

    # ── Trajectory generators ─────────────────────────────────────────────────
    def _traj(self, t):
        theta = OMEGA * t
        if MODE == 'figure8':
            return RADIUS * math.cos(theta), RADIUS * math.sin(theta) * math.cos(theta)
        return RADIUS * math.cos(theta), RADIUS * math.sin(theta)

    def _heading(self, t):
        if MODE == 'figure8':
            return 0.0
        return float(OMEGA * t + math.pi / 2)

    # ── Publishers ────────────────────────────────────────────────────────────
    def _pub_offboard(self):
        msg = OffboardControlMode()
        msg.position     = True
        msg.velocity     = True
        msg.acceleration = False
        msg.timestamp    = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(msg)

    def _pub_setpoint(self, pos, yaw=0.0):
        msg = TrajectorySetpoint()
        msg.position  = [float(pos[0]), float(pos[1]), float(pos[2])]
        msg.yaw       = float(yaw)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        # Add velocity towards target (same as SmartNavigator)
        if self.pos:
            dx = pos[0] - self.pos[0]
            dy = pos[1] - self.pos[1]
            dz = pos[2] - self.pos[2]
            msg.velocity = [dx, dy, dz]

        self.setpoint_pub.publish(msg)

    def _send_cmd(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.command          = command
        msg.param1           = float(param1)
        msg.param2           = float(param2)
        msg.target_system    = 1
        msg.target_component = 1
        msg.source_system    = 1
        msg.source_component = 1
        msg.from_external    = True
        msg.timestamp        = int(self.get_clock().now().nanoseconds / 1000)
        self.cmd_pub.publish(msg)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _dist(self, a, b):
        return math.sqrt(sum((a[i] - b[i])**2 for i in range(3)))


def main(args=None):
    rclpy.init(args=args)
    node = CircleFlight()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
