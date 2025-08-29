---
license: other
license_name: pi-lab-license-1.0
license_link: LICENSE
pipeline_tag: image-to-image
library_name: diffusers
---

# DiT4SR: Taming Diffusion Transformer for Real-World Image Super-Resolution

This repository contains the official model checkpoint for the paper [DiT4SR: Taming Diffusion Transformer for Real-World Image Super-Resolution](https://arxiv.org/abs/2503.23580).

**Project Page:** [https://adam-duan.github.io/projects/dit4sr/](https://adam-duan.github.io/projects/dit4sr/)

## Abstract

Large-scale pre-trained diffusion models are becoming increasingly popular in solving the Real-World Image Super-Resolution (Real-ISR) problem because of their rich generative priors. The recent development of diffusion transformer (DiT) has witnessed overwhelming performance over the traditional UNet-based architecture in image generation, which also raises the question: Can we adopt the advanced DiT-based diffusion model for Real-ISR? To this end, we propose our DiT4SR, one of the pioneering works to tame the large-scale DiT model for Real-ISR. Instead of directly injecting embeddings extracted from low-resolution (LR) images like ControlNet, we integrate the LR embeddings into the original attention mechanism of DiT, allowing for the bidirectional flow of information between the LR latent and the generated latent. The sufficient interaction of these two streams allows the LR stream to evolve with the diffusion process, producing progressively refined guidance that better aligns with the generated latent at each diffusion step. Additionally, the LR guidance is injected into the generated latent via a cross-stream convolution layer, compensating for DiT's limited ability to capture local information. These simple but effective designs endow the DiT model with superior performance in Real-ISR, which is demonstrated by extensive experiments.

## Usage

DiT4SR is built using the Diffusers library. For detailed usage instructions, including how to load the model and run inference for super-resolution, please refer to the [official project page](https://adam-duan.github.io/projects/dit4sr/).