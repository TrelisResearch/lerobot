<h1 align="center">
    <p>LeRobot Hackathon • October 2024</p>
    <sub>Real World RL by Alexander Soare</sub>
</h1>

_This branch of the [LeRobot](https://github.com/huggingface/lerobot) project will be used for the [LeRobot Hackathon of October 2024](https://github.com/huggingface/lerobot_hackathon_oct2024)._


## The Goal

We'll experiment with [LeRobot's TD-MPC](./lerobot/common/policies/tdmpc/modeling_tdmpc.py), starting from a [baseline setup that I've already validated](https://x.com/asoare159/status/1834246102297510301). From there, we'll try to see if we can do something more impressive and interesting.

## Required materials and workspace setup

You'll need to get materials and set up your workspace like so:

![alt text](image.jpg)

Here's what we see in the image:
- The base of a **Koch v1.0 robot** is at the top of the image.
  - Note: You can also use the Koch v1.1, Moss v1.0, or S0-100.
- The **blue cube (3x3x3 cm)** will be pushed around by the robot arm. It's important that nothing else is blue in the scene, as we will be using simple color segmentation code to segment out the cube. You can use a different color if you want, you'll just need to tweak the segmentation params.
- The two smaller rectangular regions inside the larger one are the goal regions for the cube (they alternate). Use any means you want to outline the goal regions in your setup. **I used white sports tape**.
- The outside rectangular region forms the boundaries of the cube's world. Use any means you want to set this up. **I used white cardboard strips with a longitudinal fold in the middle**.
  - Note: It's important that this has some height to it to prevent the cube going out of bounds. But it should be so high that the arm can't push against the cube from outside, when the cube is up against the edge.
  - Note: Notice that my rails are long enough for me to make the cube's workspace large if I want. I just need to translate the bottom rail down, and the top rail up a bit.

## Code

I'll highlight the main changes and additions relative to the main LeRobot fork. The code is far from perfect and is quite hacky in places:

- `train.py`: Updated to use the real robot environment.
