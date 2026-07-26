# 🎨 DDPM from Scratch using PyTorch

A PyTorch implementation of a **Denoising Diffusion Probabilistic Model (DDPM)** trained from scratch on the Fashion-MNIST dataset using a custom U-Net architecture.

The project implements the complete diffusion process including:

- Forward diffusion (adding Gaussian noise)
- Reverse diffusion (image generation)
- Sinusoidal Time Embedding
- Custom U-Net backbone
- DDPM sampling algorithm

---

## 📌 Features

- ✅ DDPM implemented from scratch
- ✅ Custom U-Net architecture
- ✅ Sinusoidal Time Embeddings
- ✅ Fashion-MNIST training
- ✅ TensorBoard logging
- ✅ Image sampling during training
- ✅ AdamW optimizer
- ✅ Gradient clipping

---

## Architecture

```
Input Image
      │
      ▼
Forward Diffusion
(Add Gaussian Noise)
      │
      ▼
   U-Net
(Time Conditioned)
      │
      ▼
Predicted Noise
      │
      ▼
Reverse Diffusion
(Image Generation)
```

---

## Network

```
Input
 ↓
Down Block (64)
 ↓
Down Block (128)
 ↓
Down Block (256)
 ↓
Bottleneck (512)
 ↑
Up Block (256)
 ↑
Up Block (128)
 ↑
Up Block (64)
 ↓
Output Noise Prediction
```

---

## Dataset

Fashion-MNIST

- Training Images: 60,000
- Image Size: 28×28
- Channels: 1

---

## Training

Loss Function

- Mean Squared Error (MSE)

Optimizer

- AdamW

Learning Rate

```
1e-4
```

Epochs

```
100
```

Diffusion Steps

```
1000
```

---

## Results

Example:
![](images/generated_vs_original.png)

---

## TensorBoard
# Loss
![](images/tensorboard/loss.png)

--
# Gradiant
![](images/tensorboard/image.png)

```
Logs/
```

Run

```bash
tensorboard --logdir Logs
```

---

## Project Structure

```
.
├── Data/
├── images/
├── Logs/
├── ddpm.ipynb
├── ddpm_fashion_mnist.pt
├── ddpm_mnist_digits.pt
├── assets/
└── README.md
```

---

## Run

Clone the repository

```bash
git clone https://github.com/ziadTeama-dev/pytorch-ddpm-from-scratch.git
```

Install requirements

```bash
pip install -r requirements.txt
```

Run

```bash
jupyter notebook ddpm.ipynb
```

---

## Future Improvements

- DDIM Sampling
- Class Conditional DDPM
- CIFAR-10 Training
- EMA Model
- Mixed Precision Training
- Cosine Noise Schedule
- Attention U-Net

---

## Technologies

- Python
- PyTorch
- Torchvision
- TensorBoard
- Matplotlib

---

## Author

**Ziad Abdelhaliem Teama**