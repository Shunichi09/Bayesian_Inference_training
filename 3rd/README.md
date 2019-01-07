# Linear regression by using bayesian inference
This program is about linear regression by using bayesian inference

# Problem formulation
## Linear regression problem

<a href="https://www.codecogs.com/eqnedit.php?latex=y_n&space;=&space;\boldsymbol{w}^T\boldsymbol{x_n}&plus;\epsilon&space;\\" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_n&space;=&space;\boldsymbol{w}^T\boldsymbol{x_n}&plus;\epsilon&space;\\" title="y_n = \boldsymbol{w}^T\boldsymbol{x_n}+\epsilon \\" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\epsilon&space;=&space;N(\epsilon|0,&space;\lambda^{-1})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\epsilon&space;=&space;N(\epsilon|0,&space;\lambda^{-1})" title="\epsilon = N(\epsilon|0, \lambda^{-1})" /></a>

We can get the X_data and Y_data, and We want to estimate the parameter, "w".
We assume that the W has Gaussian distribution.
By using the observation X_data and the input X_data, we can estimate optimal W.

<a href="https://www.codecogs.com/eqnedit.php?latex=p(\boldsymbol{w}|\boldsymbol{X,&space;Y})&space;\propto&space;p(\boldsymbol{w})\prod_{n=1}^{N}p(y_n|\boldsymbol{x_n,&space;w})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(\boldsymbol{w}|\boldsymbol{X,&space;Y})&space;\propto&space;p(\boldsymbol{w})\prod_{n=1}^{N}p(y_n|\boldsymbol{x_n,&space;w})" title="p(\boldsymbol{w}|\boldsymbol{X, Y}) \propto p(\boldsymbol{w})\prod_{n=1}^{N}p(y_n|\boldsymbol{x_n, w})" /></a>

By using above equation, We can calculate covarince matrix and mean of W's and predicted y's distribution.

# Usage

- with the animation

```
$ python linear_regression_with_animation.py
```

- comparing the dimention of the models

```
$ python linear_regression_comparing_dim.py
```

if you want to change the dimention of the model, please look at the linear_regression_with_animation

# Expected Results

- **animation**

<img src = https://github.com/Shunichi09/Bayesian_Inference_training/blob/demo_pic/3rd/animation_1.gif width = 70%>

- **comparing dimentions**

upper dimention is 2
lower dimention is 7

<img src = https://github.com/Shunichi09/Bayesian_Inference_training/blob/demo_pic/3rd/Figure_1.png width = 70%>

# Requirements
- python3.5 or more
- numpy
- matplotlib
