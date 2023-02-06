# Policy Gradient Method

* from value-based to policy-based
* from value function approximation to policy function approximation

## Basic idea

polices can be represented by parameterized function:

$$
\pi(a|s,\theta)
$$

where $\theta\in\mathbb{R}^m$ is a parameter vector.

* The function can be, for example, a neural network, whose input is $s$ , output is the probability to take each action, and parameter is $\theta$
* Advantage: when the state space is large, the tabular representation will be of low efficiency in terms of storage and generalization
* The function representation is also sometimes written as $\pi(s,a,\theta),\pi_\theta(a|s)$ or $\pi_\theta(a,s)$

The basic idea of the policy gradient is simple:

* First, metrics (or objective functions) to define optimal policies: $J(\theta)$ which can define optimal policies.
* Second, gradient-based optimization algorithms to search for optimal policies:

  $$
  \theta_{t+1}=\theta_t + \alpha\nabla_\theta J(\theta_t)
  $$

  Although the idea is simple, the complication emerges when we try to answer the following questions.

* What appropriate metrics should be used?
* How to calculate the gradients of the metrics?

## Metrics to define optimal policies

### averge value

$$
\bar{v}_\pi = \sum_{s\in\mathcal{S}}d(s)v_\pi(s)
$$

* $\bar{v}_\pi$ is weighted average of the state values.
* $d(s)\ge 0$ is the weight for state $s$
* Since $\sum_{s\in\mathcal{S}}d(s)=1$ , we can interpret $d(s)$ as a probability distribution. Then, the metric can be written as

  $$
  \bar{v}_\pi = \mathbb{E}[v_\pi(S)]
  $$

  where $S\sim d$

$$
J(\theta) = \mathbb{E}\left[\sum_{t=0}^\infty \gamma^tR_{t+1} \right] = \bar{v}_\pi
$$

How to select the distributino $d$ ?

The first case is the $d$ is independent of the policy $\pi$

* One trivial way is to treat all the states equally important and hence select $d_0(s) = 1/|\mathcal{S}|$
* Another important case is that we are only interested in a specific state $s_0$ . For example, the episodes in some tasks always start from the same state $s_0$ . Then, we only care about the long-term return starting from $s_0$ . In this case,

  $$
  d_0(s_0)=1,\quad d_0(s\ne s_0)=0
  $$

The second case is that $d$ depends on the policy $\pi$

* A common way to select $d$ as $d_\pi(s)$ , which is the stationary distribution under $\pi$ . Details of stationary distribution can be found in the last lecture and the book.
  * One basic property of $d_\pi$ is that it satisfies

  $$
  d_\pi^T P_\pi = d_\pi^T
  $$

  where $P_\pi$ is the state transition probability matrix.

### average reward

$$
\bar{r}_\pi \doteq \sum_{s\in\mathcal{S}}d_\pi(s)r_\pi(s)=\mathbb{E}[r_\pi(S)]
$$

where $S\sim d_\pi$ . Here

$$
r_\pi(s) \doteq \sum_{a\in\mathcal{A}}\pi(a|s)r(s,a)
$$

is average of the one-step immediate reward that can be obtained starting from state $s$ , and

$$
r(s,a)=\mathbb{E}[R|s,a]=\sum_r rp(r|s,a)
$$

* The weight $d_\pi$ is the stationary distribution

An equivalent definition

* Suppose an agent follows a given policy and generate a trajectory with rewards as $(R_{t+1}, R_{t+2},\dots)$
* The average single-step reward along this trajectory is

  $$
  \begin{align}
  \nonumber & \lim_{n\to\infty}\frac1n\mathbb{E}\left[R_{t+1}+R_{t+2}+\cdots+R_{t+n}|S_t=s_0 \right] \\
  \nonumber = & \lim_{n\to\infty}\frac1n \mathbb{E}\left[\sum_{k=1}^n R_{t+k}|S_t=s_0 \right]
  \end{align}
  $$

  where $s_0$ is the starting state of the trajectory.

  An important property is that

  $$
  \begin{align}
  \lim_{n\to\infty}\frac1n\mathbb{E}\left[\sum_{k=1}^n R_{t+k}|S_t=s_0 \right]
  \nonumber & = \lim_{n\to\infty}\frac1n\mathbb{E}\left[\sum_{k=1}^n R_{t+k} \right] \\
  \nonumber & = \sum_sd_\pi(s)r_\pi(s) \\
  \nonumber & = \bar{r}_\pi
  \end{align}
  $$

---

The two metrics are equivalent to each other.

In the discounted case where $\gamma < 1$ , it holds that

$$
\bar{r}_\pi = (1-\gamma)\bar{v}_\pi
$$

## Gradient of the metrics

$$
\nabla_\theta = \sum_{s\in\mathcal{S}}\eta(s)\sum_{a\in\mathcal{A}}\nabla_\theta \pi(a|s,\theta)q_\pi(s,a)
$$

where

* $J(\theta)$ can be $\bar{v}_\pi,\bar{r}_\pi$
* "=" may denote strict equality, approximation, or proportional to.
* $\eta$ is a distribution or weight of the states.

A compact and useful form of the gradient:

$$
\begin{align}
\nonumber \nabla_\theta & = \sum_{s\in\mathcal{S}}\eta(s)\sum_{a\in\mathcal{A}}\nabla_\theta \pi(a|s,\theta)q_\pi(s,a)\\
\nonumber & = \mathbb{E}[\nabla_\theta \ln\pi(A|S,\theta)q_\pi(S,A)]
\end{align}
$$

where $S\sim\eta$ and $A\sim\pi(A|S,\theta)$

we can use samples to approximate the gradient

$$
\nabla_\theta J\approx \nabla_\theta \ln \pi(a|s,\theta)q_\pi(s,a)
$$

We can use softmax function that can normalize the entries in a vector from $(-\infty, +\infty)$ to $(0,1)$

Then, the policy function has the form of

$$
\pi(a|s,\theta)=\frac{e^{h(s,a,\theta)}}{\sum_{a^\prime\in\mathcal{A}} e^{h(s,a^\prime,\theta)}}
$$

where $h(s,a,\theta)$ is another function.

## Gradient-ascent algorithm (REINFORCE)

* The gradient-ascent algorithm maximizing $J(\theta)$ is

  $$
  \begin{align}
  \nonumber \theta_{t+1} & =\theta_t + \alpha \nabla_\theta J(\theta) \\
  \nonumber &=\theta_t+\alpha\mathbb{E}\left[\nabla_\theta \ln\pi(A|S,\theta_t)q_\pi(S,A) \right]
  \end{align}
  $$

* The true gradient can be replaced by a stochastic one:

  $$
  \theta_{t+1}=\theta_t+\alpha\left[\nabla_\theta \ln\pi(a_t|s_t,\theta_t)q_\pi(s_t,a_t) \right]
  $$

* Furthermore, since $q_\pi$ is unknown, it can be approximated:

  $$
  \theta_{t+1}=\theta_t+\alpha\left[\nabla_\theta \ln\pi(a_t|s_t,\theta_t)q_t(s_t,a_t) \right]
  $$

  There are different methods to approximate $q_\pi(s_t,a_t)$

  * Monte-Carlo based method, REINFORCE

The policy gradient method is on-policy.
