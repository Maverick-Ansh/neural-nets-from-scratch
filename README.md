# Neural Nets from Scratch

Built live on a Colab **T4 GPU**, from the ground up — **no black boxes**. Starts at a single neuron whose
gradient is computed by hand, and ends at a convolutional network, with a from-scratch autograd engine
underneath it all. Every gradient is derived, and every "trick" is shown *breaking* first, then *fixed*.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maverick-Ansh/neural-nets-from-scratch/blob/main/neural_nets_from_scratch.ipynb)

## Run it
Click the badge → **Runtime → Change runtime type → T4 GPU** → run the cells top to bottom.

## The journey

| # | Topic | What happens | Core idea |
|---|---|---|---|
| 1 | One neuron | Hand-computed gradient + descent, visualized as a ball rolling down the loss surface | Learning = roll the loss downhill using only the local slope |
| 2 | Hidden layer | 2-layer MLP with **hand-written backprop**, cracks the non-linear "moons" (98%) | A hidden layer bends the straight line into a curve |
| 3 | Autograd unmasked | Prove `loss.backward()` == the hand chain rule (diff ~1e-7) | Autograd is the chain rule, recorded and replayed |
| 4 | Why depth breaks | Measure signals vanishing over 20 layers; a 16-layer net frozen at loss `ln(2)` until proper init | Init keeps the signal alive layer-to-layer |
| 5 | Scaling | 1.5M-param MLP on FashionMNIST; fp32 vs **fp16** tensor cores on the T4 | Same machinery, bigger; fp16 speeds the matmuls |
| 6 | Going deep | 50-layer net: plain 80% → **+ residual 88.5%** | Normalization steadies; **residuals** make depth pay off |
| 7 | Optimizers | Hand-coded **SGD / Momentum / Adam** traced on a ravine | Adam rescales each axis so it drives straight down |
| 8 | Overfitting | Net memorizes 1000 images — even 30% random labels; **early stopping** +9.8 pts | Capacity is huge; the real limit is data |
| F | micrograd | A scalar `Value` autograd class — **0.0 grad diff vs PyTorch** — trains a real net | You wrote the engine PyTorch hides |
| G | Conv nets | A CNN: **90.5% with 3.5× fewer params** than the MLP; learned filters visualized | Architecture matching the data beats brute force |
| H | Augmentation | Flip/shift on the fly: smaller train/test gap, no memorization | Manufacture data to fight overfitting |

## Selected results
- One neuron recovers a hidden rule (`y = 2x − 1`) from noisy data alone.
- Hand-coded backprop hits **98%** on `make_moons`.
- Custom autograd matches PyTorch to **0.0** (float64).
- A 16-layer net sits frozen at `ln(2)` until Xavier/Kaiming init revives it.
- 50-layer net: plain **80%** → + BatchNorm + residual **88.5%**.
- MLP **88.4%** (1.46M params) vs CNN **90.5%** (0.42M params) on FashionMNIST.
- Mixed precision (fp16) ~**1.3×** faster on the T4 at equal accuracy.

## The meta-lesson
Regularizers (dropout, weight-decay, augmentation) reliably shrink the **train/test gap**, but their boost to
test *accuracy* is small when the real bottleneck is **data**. The big, clean wins came from **architecture**
(conv: +2 pts *and* fewer params) and **early stopping** (+9.8 pts). The skill is **measuring which lever
matters** — not trusting the slogan.

## Built with
PyTorch · Google Colab (T4) · NumPy · Matplotlib · scikit-learn (toy datasets)
