# Approximate distributions
It could be sometimes extremely difficult to get sample data from distributions such as mixtured models.
When you want to do that, you should approximate the distribution.
This program is about the method to approximate the distribution.
You can see the following methods

- Gibbs sampling
- Variational inference

# Problem formulation
We applied these method to estimate 2-dimentional Gaussian model.
If you want to know the more detail, you can go the reference or the reference book which URL is in top README

# Usage

- Gibbs sampling

```
$ python gibbs.py
```

- Variational inference

```
$ python variation.py
```

# Expected Results

- Gibbs sampling

<img src = https://github.com/Shunichi09/Bayesian_Inference_training/blob/demo_pic/4th/sampling/animation_1.gif width = 70%>

KL divergence

<img src = https://github.com/Shunichi09/Bayesian_Inference_training/blob/demo_pic/4th/sampling/Figure_1.png width = 70%>

- Variational inference

<img src = https://github.com/Shunichi09/Bayesian_Inference_training/blob/demo_pic/4th/sampling/animation_2.gif width = 70%>

KL divergence

<img src = https://github.com/Shunichi09/Bayesian_Inference_training/blob/demo_pic/4th/sampling/Figure_2.png width = 70%>

# Requirement

- python3.5 or more
- numpy
- matplotlib

# Reference

- http://machine-learning.hatenablog.com/entry/2016/02/04/201945