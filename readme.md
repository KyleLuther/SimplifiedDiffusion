# A simple diffusion model in PyTorch
Here is an unconditional diffusion model from scratch using PyTorch. For building my own intuition, I wanted to see a simple working example without too many nested variables or too much abstraction.

This model isn't state of the art and isn't an exact implementation of any existing diffusion model, but it loosely follows the the general framework set out by this insightful but technical NVIDIA paper [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364). Specifically there is no coupling between training and generation noise distributions. To demonstrate the simplicity, here is the generation code:
```python
@torch.no_grad()
def generate_samples(model, sigma=100.0, sigma_min=0.03, alpha=0.1, beta=.40, device='cuda'):
    x = sigma * torch.randn((64,1,32,32), device=device)
    xs = [x.cpu()]
    while sigma > sigma_min:
        x = x - alpha * sigma * model(x,sigma) + beta * sigma * torch.randn_like(x)
        sigma = sigma * np.sqrt((1-alpha)**2 + beta**2) # noise decays exponentially
        xs.append(x.cpu()) # save all intermediate generations

    return torch.stack(xs) # (nsteps, batch, channels, height, width)
```
We start with an image of pure noise, then iteratively subtract some fraction of the predicted noise at each step, and add back in some new noise. In our code, the amount of subtraction and reinjection is controlled by the hyperparameters $\alpha$ and $\beta$ respectively. This subtraction/reinjection leads to a noise level which decays exponentially (see below for explanation). Here are some generations after 50 epochs (45 minutes on Colab) of training on padded MNIST digits:

<img src="generated.png"  height="200" />

Most of the specifics are in the code itself, but I'll give an overview below.

## Overview

### Training
We train a UNet to predict the normalized noise content of images corrupted with various amounts of noise. Specifically we sample noise levels from a LogUniform distribution (which preferentially weights low noise levels more). A Log Uniform distribution was used in the popular [score-based generative modeling through stochastic differential equations](https://arxiv.org/abs/2011.13456).

$$ \sigma \sim \text{LogUniform}(\sigma_{\text{min}}=0.03, \sigma_{\text{max}}=100.0) $$ 

$\sigma_{\text{min}}$ and $\sigma_{\text{max}}$ are hyperparameters. We use mean squared error loss between the unit variance noise $\epsilon$ and network outputs, as done in the classic [DDPM paper](https://arxiv.org/abs/2006.11239).

$$ L(\theta) = E_{x, \sigma} \left\Vert \epsilon - f_\theta (x+\sigma \epsilon, \sigma) \right\Vert^2 $$ 

### Generation
We choose a starting noise level $\sigma_0$ and initialize $x_0 \sim \mathcal{N}(0,\sigma_0^2)$ as noise and iterate the following:

$$x_{i+1} = x_i - \alpha \sigma_i \hat{\epsilon}_{\theta}(x_i, \sigma_i) + \beta \sigma_i z_i$$

$$\sigma_{i+1} = \sigma_i \sqrt{(1-\alpha)^2 + \beta^2}$$

where $z_i$ is a random unit Gaussian vector. We're subtracting some fraction of the predicted noise $\sigma_i \hat{\epsilon}_{\theta}(x_i, \sigma_i)$, then adding back in some more noise.

$\alpha$ and $\beta$ are hyperparameters. They should be chosen so that $\sqrt{(1-\alpha)^2 + \beta^2}<1$ (ie so that $\sigma$ decays with each step). Commonly, you'll see diffusion models define some set of noise levels and derive dynamics from the noise levels, here we define dynamics and then infer the noise levels. 

#### Why is the noise decaying exponentially?
Our denoiser requires the noise level as input so we need to be able to at least estimate the noise level $\sigma$ during generation. In full generality, this is actually pretty tricky without the probabilistic framework of DDPM, Langevin dynamics, etc. But there is a simple non-rigorous way to estimate $\sigma$ here using the argument proposed in [Kadkhodaie and Simoncelli](https://arxiv.org/pdf/2007.13640). First suppose at iteration $i$ we have a pattern which is the sum of a single image and Gaussian noise with known noise level $\sigma_i$.

$$ x_i = x + \sigma_i \epsilon_i$$

Now suppose our denoiser is good, so good that it exactly outputs the noise content of the image. In other words $\hat\epsilon_\theta(x_i, \sigma_i) = \epsilon_i$. Let's run one step of our algorithm:

$$ x_{i+1} = (x + \sigma_i \epsilon_i) - \alpha \sigma_i \epsilon + \beta \sigma_i z_i  = x + \sigma_i [(1-\alpha) \epsilon_i + \beta z_i] $$

$\epsilon_i$ and $z_i$ are two iid gaussian vectors, so $[(1-\alpha) \epsilon_i + \beta z_i]$ is also a gaussian vector with mean 0 and variance $(1-\alpha)^2 + \beta^2$. We therefore write:

$$ x_{i+1} = x + \sigma_i \sqrt{(1-\alpha)^2 + \beta^2} \epsilon_{i+1} = x + \sigma_{i+1} \epsilon_{i+1}$$

This means that the noise level at the next iteration is $\sigma_{i+1} = \sigma_i \sqrt{(1-\alpha)^2 + \beta^2}$.  

### Model
We use a UNet that is conditioned on the noise level. Noise conditioning is implemented in two ways. First, we rescale the inputs $x \leftarrow x / \sqrt{1+\sigma}^2$ as the first operation in our network. Second, we use a noise-level-dependent affine transformation applied to some of the feature maps. The exact method used in the notebook is a little unconventional, but at a high level this works by:

1. mapping the noise level $\sigma$ to a scalar between 0 and 1 via $\gamma = \sigma / \sqrt{1+\sigma^2}$
2. apply a random sinusoidal embedding of this recaled noise level: $\mathbf{q} = sin(\mathbf{w} \gamma)$ where $\mathbf{w}$ is a random weight vector that is fixed during training.
3. compute a per-layer affine transform: $\mathbf{s}^l = \mathbf{W}_s^l \mathbf{q}$ and $\mathbf{b}^l = \mathbf{W}_b^l \mathbf{q}$
4. apply this affine transform to feature maps: $x^l_{b,i,u} = s^l_i x^l_{b,i,u} + b^l_i$ where $b,i,u$ are batch, channel, space indices

I'm also using GroupNorm to normalize the net, which is concerning as the network is unable to *see* and therefore denoise the DC component of input images. For these simple MNIST digits, its ok to ignore this point but its worth considering.

## Evaluation
Ultimately I'm just eyeballing the generations and picking parameters ($\sigma_{\text{min}}, $\sigma_{\text{max}}, $\alpha$, $\beta$) which give nice-looking generations. To push this forward, we'd really want to use a quantitative evaluation metric. A common metric is FID (Fréchet inception distance) which is a method to compare the distribution of generated digits with some held-out set of real digits. It works by computing features with a pre-trained classifier and then comparing covariance matrices of these features for the real and generated digits. 
