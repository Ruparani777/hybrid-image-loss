# hybrid-image-loss
Hybrid Image Loss Function

This repository implements a hybrid image loss function that combines L1 loss and perceptual loss (based on VGG19 features).

## Loss Formula

    L_total = λ1 * L1(G(x), y) + λ2 * ||ϕ(G(x)) - ϕ(y)||^2

- L1: pixel-level difference
- ϕ: features from intermediate layers of pretrained VGG19
- λ1 and λ2 are balancing weights

Useful for high-resolution texture generation and image reconstruction tasks.
