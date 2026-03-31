# World Setup for Visual Odometry

## What's here
`default_with_landmarks.sdf` — the Gazebo world file with 12 coloured landmark
pillars added for VO feature tracking. Pillars are placed in a ring at r=8m, z=3m
around the drone start position.

## On a new machine — copy this file into PX4:

```bash
cp worlds/default_with_landmarks.sdf \
   ~/PX4-Autopilot/Tools/simulation/gz/worlds/default.sdf
```

Then launch as normal:
```bash
cd ~/PX4-Autopilot
make px4_sitl gz_x500_skydio
```

## Why
ORB feature detection needs edges and texture to track.
The default Gazebo world is a flat grey plane — zero features.
The landmark pillars give the camera something to lock onto during circle flight.
