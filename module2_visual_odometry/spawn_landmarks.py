#!/usr/bin/env python3
"""
Spawn visual landmark boxes in Gazebo for feature detection testing.

ORB/SIFT need texture and edges to detect features.
A flat empty Gazebo world gives zero features → no epipolar geometry.
This spawns a ring of coloured boxes at flight altitude so the
front camera always has something to track during circle flight.

Run ONCE before starting circle_flight + epipolar_geometry:
  ros2 run module2_visual_odometry spawn_landmarks
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess
import math
import time


# Landmark field: multiple rings (inside + outside the 5 m flight circle) at
# several heights, so the tangent-facing front camera always has >=15 features
# in view (OpenVINS init needs >=15; a single sparse ring gave ~11 -> never init).
FLIGHT_HEIGHT = 3.0                  # metres (positive, Gazebo Z-up)
RING_RADII    = [4.0, 7.0, 10.0]     # inside, on, and outside the flight circle
RING_HEIGHTS  = [1.5, 3.0, 4.5]      # fill vertical FOV
BOXES_PER_RING = 12                  # per (radius, height) ring
RING_RADIUS   = 7.0                  # kept for log message compatibility
N_BOXES       = len(RING_RADII) * len(RING_HEIGHTS) * BOXES_PER_RING  # 108


BOX_SDF_TEMPLATE = """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <box><size>1.0 1.0 1.5</size></box>
        </geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
        </material>
      </visual>
      <collision name="collision">
        <geometry>
          <box><size>1.0 1.0 1.5</size></box>
        </geometry>
      </collision>
    </link>
    <pose>{x} {y} {z} 0 0 0</pose>
  </model>
</sdf>"""


def spawn_box(name, x, y, z, r, g, b):
    sdf = BOX_SDF_TEMPLATE.format(
        name=name, x=x, y=y, z=z, r=r, g=g, b=b
    )
    # Use gz service to spawn model
    result = subprocess.run(
        ['gz', 'service', '-s', '/world/default/create',
         '--reqtype', 'gz.msgs.EntityFactory',
         '--reptype', 'gz.msgs.Boolean',
         '--timeout', '2000',
         '--req', f'sdf: "{sdf.strip()}"'],
        capture_output=True, text=True
    )
    return result.returncode == 0


class SpawnLandmarks(Node):
    def __init__(self):
        super().__init__('spawn_landmarks')

    def spawn_all(self):
        self.get_logger().info(f"Spawning {N_BOXES} landmark boxes...")
        self.get_logger().info(f"Ring radius: {RING_RADIUS}m at height: {FLIGHT_HEIGHT}m")

        # Colour palette — high contrast for feature detection
        colours = [
            (1.0, 0.0, 0.0),   # red
            (0.0, 1.0, 0.0),   # green
            (0.0, 0.0, 1.0),   # blue
            (1.0, 1.0, 0.0),   # yellow
            (1.0, 0.0, 1.0),   # magenta
            (0.0, 1.0, 1.0),   # cyan
        ]

        ok = 0
        i = 0
        for ri, radius in enumerate(RING_RADII):
            for hi, z in enumerate(RING_HEIGHTS):
                # stagger angle per ring so boxes don't line up radially
                offset = 2 * math.pi * (ri * 0.33 + hi * 0.17) / BOXES_PER_RING
                for k in range(BOXES_PER_RING):
                    angle = 2 * math.pi * k / BOXES_PER_RING + offset
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    r, g, b = colours[i % len(colours)]
                    name = f'lm_r{radius:.0f}_h{z:.0f}_{k:02d}'
                    if spawn_box(name, x, y, z, r, g, b):
                        ok += 1
                    else:
                        self.get_logger().warn(f"  ✗ Failed to spawn {name}")
                    i += 1
        self.get_logger().info(f"  spawned {ok} ring boxes across "
                               f"{len(RING_RADII)} radii × {len(RING_HEIGHTS)} heights")

        # Also add a textured ground pattern (4 large flat boxes)
        for i, (gx, gy) in enumerate([(5,5),(-5,5),(-5,-5),(5,-5)]):
            spawn_box(f'ground_marker_{i}', gx, gy, 0.1,
                      0.8 if i % 2 == 0 else 0.2,
                      0.8 if i % 2 == 1 else 0.2,
                      0.2)

        self.get_logger().info(f"\nSpawned {ok}/{N_BOXES} boxes successfully.")
        self.get_logger().info("Now run circle_flight + epipolar_geometry.")


def main(args=None):
    rclpy.init(args=args)
    node = SpawnLandmarks()
    node.spawn_all()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
