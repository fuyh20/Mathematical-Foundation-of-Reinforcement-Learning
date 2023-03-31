# Stochastic Approximation and Stochastic Gradient Descent

## Motivating examples

Revisit the mean estimation problem:

* Conisder a random variable $X$
* Our aim is to estimate $\mathbb{E}[X]$
* Suppose that we collected a sequence of iid samples $\{x_i\}_{i=1}^N$
* The expectation of $X$ can be approximated by

$$
\mathbb{E}[X]\approx\bar{x}\coloneqq\frac1N\sum_{i=1}^N x_i
$$

Why do we care about mean estimation so much?

* Many values in RL such as state/action values are defined as means

How to calculate the means?

In particular, suppose

$$
w_{k+1}=\frac1k \sum_{i=1}^k x_i, \quad k=1,2,\dots
$$

and hence

$$
w_k=\frac{1}{k-1}\sum_{i=1}^{k-1}x_i, \quad k=1,2,\dots
$$

Then, $w_{k+1}$ can be expressed in terms of $w_k$ as

$$
w_{k+1}=\frac1k\sum_{i=1}^k x_i = \frac1k\left(\sum_{i=1}^{k-1}x_i + x_k \right)=w_k-\frac1k(w_k-x_k)
$$

Therefore, we obtain the following iterative algorithm:

$$
w_{k+1}=w_k-\frac1k(w_k-x_k)
$$

Furthermore, consider an algorithm with a more general expression:

$$
w_{k+1}=w_{k}-\alpha_k(w_k-x_k)
$$

where $1/k$ is replaced by $\alpha_k > 0$

* If $\{\alpha_k\}$ satisfy some mild conditions, this algorithm still converge to the mean $\mathbb{E}[X]$
* This algorithm is a special SA algorithm and also a special stochastic gradient descent algorithm.

## Robbins-Monro algrithm

Stochastic approximation (SA):

* SA refers to a broad class of stochastic iterative algorithms solving root finding or optimization problems.
* SA is powerful in the sense that it does not require to know the expression of the objective function nor its derivative.

Robbins-Monro (RM) algorithm:

* It is pioneering work in the field of stochastic approximation.
* The famous stochastic gradient descent algorithm is a special form of the RM algorithm.

---

**Problem statement**: Suppose we would like to find the root of the equation

$$
g(w)=0
$$

where $w\in\mathbb{R}$ is the variable to be solved and $g:\mathbb{R}\to\mathbb{R}$ is a function.

* Many problems can be eventually converted to this root finding problem. For example, suppose $J(w)$ is an objective function to be minimized. Then, the optimization problem can be converted to

$$
g(w) = \nabla_w J(w) = 0
$$

* Note that an equation like $g(w)=c$ with $c$ as a constant can be converted to the above equation by writing $g(w)-c$ as a new function.

If the expression of the function $g$ is unknown, for example, the function is represented by an artifical neural network.

The Robbins-Monro (RM) algorithm can solve this problem:

$$
w_{k+1}=w_k-a_k \tilde{g}(w_k,\eta_k), \quad k = 1, 2, 3, \dots
$$

where

* $w_k$ is the kth estimate of the root
* $\tilde{g}(w_k, \eta_k)=g(w_k)+\eta_k$ is the kth noisy observation
* $a_k$ is a positive coefficient

The function $g(w)$ is a black box! This algorithm relies on data:

* Input sequence: $\{w_k\}$
* Noisy output sequence: $\{\tilde{g}(w_k,\eta_k)\}$

Philosophy: without model, we need data

## Robbins-Monro algorithm - Convergence properties

Robbins-Monro Theorem

> In the Robbins-Monro algorithm, if
>
> 1. $0<c_1\le\nabla_w g(w)\le c_2 \quad \mathrm{for\ all}\quad w$
> 2. $\sum_{k=1}^\infty a_k = \infty$ and $\sum_{k=1}^\infty a_k^2 < \infty$
> 3. $\mathbb{E}[\eta_k|\mathcal{H}_k]$ and $\mathbb{E}[\eta_k^2|\mathcal{H}_k]<\infty$
>
> where $\mathcal{H_k}=\{w_k,w_{k-1},\dots\}$ , then $w_k$ converges with probability 1 (w.p.1) to the root $w^*$ satisfying $g(w^*) = 0$

## Robbins-Monro algorithm - Apply to mean estimation

Recall that

$$
w_{k+1}=w_k+\alpha_k(x_k-w_k)
$$

is the mean estimation algorithm

* if $\alpha_k$ is not $1/k$ , the convergence was not analyzed

This algorithm is a special case of the RM algorithm.

## SGD - Introduction

SGD is a special RM algorithm

Suppose we aim to solve the following optimization problem:

$$
\min_w J(w)=\mathbb{E}[f(w,X)]
$$

* $w$ is the parameter to be optimized
* $X$ is a random variable. The expectation is with respect to $X$
* $w$ and $X$ can be either scalars or vectors. The function $f(\cdot)$ is a scalar

Method 1: gradient descent (GD)

$$
w_{k+1}=w_k - \alpha_k \nabla_w \mathbb{E}[f(w,X)]=w_k-\alpha_k\mathbb{E}[\nabla_w f(w_k, X)]
$$

Drawback: the expected value is difficult to obtain.

Method 2: batch gradient descent (BGD)

$$
\mathbb{E}[\nabla_w f(w_k, X)]\approx \frac1n\sum_{i=1}^n\nabla_w f(w_k, x_i)
$$

$$
w_{k+1}=w_k-\alpha_k \frac1n\sum_{i=1}^n\nabla_w f(w_k, x_i)
$$

Drawback: it requires many samples in each iteration for each $w_k$

Method 3: stochastic gradient descent (SGD)

$$
w_{k+1}=w_k-\alpha_k\nabla_w f(w_k, x_k)
$$

## SGD - Convergence

SCD is a special RM algorithm. Then, the convergence naturally follows

The aim of SGD is to minimize

$$
J(w) = \mathbb{E}[f(w,X)]
$$

This problem can be converted to a root-finding problem:

$$
\nabla_w J(w)=\mathbb{E}[\nabla_w f(w, X)]=0
$$

Let

$$
g(w)=\nabla_w J(w)=\mathbb{E}[\nabla_w f(w,X)]
$$

Then, the aim of SGD is to find the root of $g(w) = 0$

What we can measure is

$$
\begin{align}
\tilde{g}(w,\eta) & =\nabla_w f(w,x) \\
& =\underbrace{\mathbb{E}[\nabla_w f(w,X)]}_{g(w)} + \underbrace{\nabla_w f(w,x)-\mathbb{E}[\nabla_w f(w,X)]}_\eta
\end{align}
$$

Then, the RM algorithm for solving $g(w)=0$ is

$$
w_{k+1}=w_k-a_k\tilde{g}(w_k, \eta_k)=w_k-a_k\nabla_wf(w_k,x_k)
$$

Therefore, SGD is a special RM algorithm.

## SGD - interesting properties

$$
\delta_k\le\frac{| \overbrace{\nabla_wf(w_k,x_k)}^{\mathrm{stochastic \  gradient}} - \overbrace{\mathbb{E}[\nabla_wf(w_k,X)] }^{\mathrm{true\ gradient}}|}{\underbrace{c|w_k-w^*|}_{\mathrm{distance\ the\ optimal\ solution}}}
$$

The above equation suggests an interesting convergence pattern of SGD,

* The relative error $\delta_k$ is inversely proportional to $|w_k-w^*|$
* When $|w_k-w^*|$ is large, $\delta_k$ is small and SGD behaves like GD
* When $w_k$ is close to $w^*$ , the relative error may be large and the convergence exhibits more randomness in the neighborhood of $w^*$

## SGD - A deterministic formulation

The formulation of SGD we introduced above involves random variables and expectation.

One may often encounter a deterministic formulation of SGD without involving any random variables.

Consider the optimization problem:

$$
\min_w J(w)=\frac1n \sum_{i=1}^n f(w, x_i)
$$

* $f(w,x_i)$ is parameterized function.
* $w$ is the parameter to be optimized.
* a set of real number $\{x_i\}_{i=1}^n$ , where $x_i$ does not have to be a sample of any random variable.

We can use the following iterative algorithm:

$$
w_{k+1}=w_k - \alpha_k\nabla_wf(w_k,x_k)
$$

Suppose $X$ is a random variable defined on the set $\{x_i\}_{i=1}^n$ . Suppose its probability distribution is uniform such that

$$
p(X=x_i)=\frac1n
$$

Then, the deterministic optimization problem becomes a stochastic one:

$$
\min_w J(w)=\frac1n \sum_{i=1}^n f(w, x_i)=\mathbb{E}[f(w,X)]
$$

## BGD, MBGD, and SGD

Suppose we would like to minimize $J(w)=\mathbb{E}[f(w,X)]$ given a set of random samples $\{x_i\}_{i=1}^n$ of $X$ . The BGD, SGD, MBGD algorithm solving this problem are, respectively,

$$
\begin{align}
\nonumber & w_{k+1} = w_k - \alpha_k \frac1n \sum_{i=1}^n \nabla_w f(w_k, x_i) \quad (BGD) \\
\nonumber & w_{k+1} = w_k - \alpha_k \frac1m \sum_{j\in\mathcal{I}_k}\nabla_w f(w_k, x_j) \quad (MBGD) \\
\nonumber & w_{k+1} = w_k - \alpha_w \nabla_w f(w_k, x_k) \quad (SGD)
\end{align}
$$
