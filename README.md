# HVI-Diff: Physically Decoupled Interactive Diffusion for Low-Light Image Enhancement

This repository contains the official PyTorch implementation of the paper: **"HVI-Diff: Physically Decoupled Interactive Diffusion for Low-Light Image Enhancement"**.

Low-light image enhancement (LLIE) is challenging due to the complex entanglement of illumination degradation and chromatic distortion. While diffusion models excel at generation, they often struggle with color shifts and structural hallucinations in the RGB space.

**HVI-Diff** is a novel framework that integrates physical priors with diffusion models. By introducing the **HVI color space**, we explicitly decouple the input into **Intensity** and **Chromaticity** components. Our method features:

* **Parallel Convolutional Coarse Module (PCCM)**: Provides deterministic physical anchors for stable restoration.
* **Physically Decoupled Interactive Diffusion (PDID)**: A dual-stream diffusion module for stochastic refinement of details.
* **Interaction Factor (IF)**: A mechanism to dynamically bridge the two streams, ensuring structural and chromatic consistency.



The code will be coming soon!
