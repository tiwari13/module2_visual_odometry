#!/usr/bin/env python3
"""
Spawn a dense visual-feature "arena" in Gazebo for VIO.

OpenVINS (and any feature tracker) needs texture/edges; the default PX4 Gazebo
world is a flat, textureless plane -> too few features -> VIO never initializes
and EV fusion is intermittent. This spawns:
  - rings of coloured pillars ALL AROUND the flight circle, at several heights,
    so the tangent-facing front camera always sees >=15 features as it sweeps;
  - a ground grid of flat tiles, so downward-tilted views also have features.

All objects are OUTSIDE the 5 m flight circle (nearest ring at 6 m) so they
never collide with the drone.

Run ONCE after the sim is up, before circle_flight:
  ros2 run module2_visual_odometry spawn_landmarks

Re-running is safe: it removes any previously spawned arena objects first.

NOTE on a circle flight: continuous yaw (camera sweeps 360deg/lap) is a stress
case for VIO feature tracking. The dense field mitigates it; for the steadiest
VIO, a translation-heavy / less rotation-heavy path is even better.
"""

import math
import subprocess

import rclpy
from rclpy.node import Node

# ── Arena geometry ───────────────────────────────────────────────────────────
RING_RADII   = [6.0, 8.0, 10.0, 12.0]   # all OUTSIDE the 5 m flight circle (safe)
RING_HEIGHTS = [1.5, 3.0, 4.5]          # fill the vertical FOV
BOXES_PER_RING = 16                     # per (radius, height) ring
GROUND_EXTENT = 14.0                    # ground grid spans +/- this (m)
GROUND_STEP   = 3.5                     # tile spacing (m)

COLOURS = [
    (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0),
]

_MODEL_SDF = """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
        </material>
      </visual>
    </link>
    <pose>{x} {y} {z} 0 0 0</pose>
  </model>
</sdf>"""


def _gz(args, req):
    """Call a gz service; return (ok, stdout). req is the text-format request.

    The SDF body contains double-quotes, so the request string MUST be delimited
    with SINGLE quotes (the original bug used double quotes -> silent parse fail).
    """
    out = subprocess.run(
        ["gz", "service", "-s", args["s"],
         "--reqtype", args["reqtype"], "--reptype", "gz.msgs.Boolean",
         "--timeout", "3000", "--req", req],
        capture_output=True, text=True,
    )
    ok = out.returncode == 0 and "data: true" in out.stdout
    return ok, (out.stdout + out.stderr).strip()


def spawn_box(name, x, y, z, r, g, b, sx=1.0, sy=1.0, sz=1.5):
    sdf = _MODEL_SDF.format(name=name, x=x, y=y, z=z, r=r, g=g, b=b,
                            sx=sx, sy=sy, sz=sz).strip()
    return _gz({"s": "/world/default/create", "reqtype": "gz.msgs.EntityFactory"},
               f"sdf: '{sdf}'")[0]


def remove_model(name):
    return _gz({"s": "/world/default/remove", "reqtype": "gz.msgs.Entity"},
               f"name: '{name}' type: MODEL")[0]


class SpawnLandmarks(Node):
    def __init__(self):
        super().__init__("spawn_landmarks")

    def spawn_all(self):
        log = self.get_logger()

        # 1. idempotent: clear any arena objects from a previous run
        removed = 0
        for ri, radius in enumerate(RING_RADII):
            for hi, _ in enumerate(RING_HEIGHTS):
                for k in range(BOXES_PER_RING):
                    if remove_model(f"pillar_{ri}_{hi}_{k:02d}"):
                        removed += 1
        ng = int(2 * GROUND_EXTENT / GROUND_STEP) + 1
        for gi in range(ng):
            for gj in range(ng):
                if remove_model(f"gtile_{gi}_{gj}"):
                    removed += 1
        if removed:
            log.info(f"removed {removed} arena objects from a previous run")

        # 2. rings of pillars all around, multiple heights
        pillars = 0
        idx = 0
        for ri, radius in enumerate(RING_RADII):
            for hi, z in enumerate(RING_HEIGHTS):
                offset = 2 * math.pi * (ri * 0.31 + hi * 0.17) / BOXES_PER_RING
                for k in range(BOXES_PER_RING):
                    ang = 2 * math.pi * k / BOXES_PER_RING + offset
                    x, y = radius * math.cos(ang), radius * math.sin(ang)
                    r, g, b = COLOURS[idx % len(COLOURS)]
                    if spawn_box(f"pillar_{ri}_{hi}_{k:02d}", x, y, z, r, g, b):
                        pillars += 1
                    else:
                        log.warn(f"  ✗ pillar_{ri}_{hi}_{k:02d} failed")
                    idx += 1
        log.info(f"spawned {pillars} pillars "
                 f"({len(RING_RADII)} radii × {len(RING_HEIGHTS)} heights)")

        # 3. ground grid of flat checkerboard tiles (features for downward views)
        tiles = 0
        n = int(2 * GROUND_EXTENT / GROUND_STEP) + 1
        for gi in range(n):
            for gj in range(n):
                x = -GROUND_EXTENT + gi * GROUND_STEP
                y = -GROUND_EXTENT + gj * GROUND_STEP
                dark = (gi + gj) % 2 == 0
                c = (0.05, 0.05, 0.05) if dark else (0.9, 0.9, 0.9)
                if spawn_box(f"gtile_{gi}_{gj}", x, y, 0.05, *c,
                             sx=GROUND_STEP, sy=GROUND_STEP, sz=0.02):
                    tiles += 1
        log.info(f"spawned {tiles} ground tiles ({n}×{n} grid)")

        total = pillars + tiles
        log.info(f"ARENA READY: {total} feature objects. Now run circle_flight.")
        if pillars < len(RING_RADII) * len(RING_HEIGHTS) * BOXES_PER_RING:
            log.warn("Some pillars failed to spawn — check gz is up and the "
                     "world is /world/default.")


def main(args=None):
    rclpy.init(args=args)
    node = SpawnLandmarks()
    node.spawn_all()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
