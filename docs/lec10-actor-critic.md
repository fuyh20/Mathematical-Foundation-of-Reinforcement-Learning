# Actor-Critic Methods

"actor" refers to policy update. It is called actor is because policies wil applied to take actions.

"critic" refers to policy evaluation or value estimation. It is called critic because it criticizes the policy by evaluating it.

## The simplest actor-critic (QAC)

The stochastic gradient-ascent algorithm is

$$
\theta_{t+1}=\theta_t+\alpha\left[\nabla_\theta \ln\pi(a_t|s_t,\theta_t)q_t(s_t,a_t) \right]
$$

* this algorithm corresponds to actor.
* the algorithm estimating $q_t(s_t,a_t)$ corresponds to critic.

How to get $q_t(s_t,a_t)$ ?

Temporal-difference learning: if TD is used, such kind of algorithms are usually called actor-critic.

---

The simplest actor-critic algorithm (QAC)

**aim** : search for an optimal policy by maximizing $J(\theta)$

At time step $t$ in each episode, do

Generate $a_t$ following $\pi(a|s_t,\theta_t)$ , observe $r_{t+1}, s_{t+1}$ , and then generate $a_{t+1}$ following $\pi(a|s_{t+1},\theta_t)$

Critic (value update):

$$
w_{t+1} = w_t + \alpha_w[r_{t+1}+\gamma q(s_{t+1},a_{t+1},w_t)-q(s_t,a_t,w_t)]\nabla_w q(s_t,a_t,w_t)
$$

Actor (policy update):

$$
\theta_{t+1}=\theta_t + \alpha_\theta\nabla_\theta \ln\pi(a_t|s_t,\theta_t)q(s_t,a_t,w_{t+1})
$$

---

## Advantage actor-critic (A2C)

The core idea is to introduce a baseline to reduce variance.

Property: the policy gradient is invariant to an additional baseline

$$
\begin{align}
\nonumber \nabla_\theta J(\theta)& =\mathbb{E}_{S\sim\eta,A\sim\pi}\left[\nabla_\theta\ln\pi(A|S,\theta_t)q_\pi(S,A) \right] \\
\nonumber & = \mathbb{E}_{S\sim\eta,A\sim\pi}\left[\nabla_\theta\ln\pi(A|S,\theta_t)(q_\pi(S,A)-b(S)) \right] \\
\end{align}
$$

Here, the additional baseline $b(S)$ is a scalar function of $S$ .

Proof is omitted.

The gradient is $\nabla_\theta J(\theta)=\mathbb{E}[X]$ where

$$
X(S,A)\doteq \nabla_\theta \ln\pi(A|S,\theta_t) [q(S,A)-b(S)]
$$

We have

* $\mathbb{E}[X]$ is invariant to $b(S)$
* $\mathrm{var}(X)$ is NOT invariant to $b(S)$

our goal: select an optimal baseline $b$ to minimize $\mathrm{var}(X)$ .

The optimal baseline that can be minimize $\mathrm{var} (X)$ is, for any $s\in\mathcal{S}$

$$
b^*(s)=\frac{\mathbb{E}_{A\sim\pi}\left[\|\nabla_\theta\ln\pi(A|s,\theta_t) \|^2 q(s,A) \right]}{\mathbb{E}_{A\sim\pi}\left[\|\nabla_\theta\ln\pi(A|s,\theta_t) \|^2 \right]}
$$

* Although this baseline is optimal, it is complex.

* We can remove the weight $\|\nabla_\theta\ln\pi(A|s,\theta_t) \|^2$ and select the suboptimal baseline:

  $$
  b(s)=\mathbb{E}_{A\sim\pi}[q(s,A)]=v_\pi(s)
  $$
  
  which is the state value of $s$

When $b(s)=v_\pi(s)$

the gradient-ascent algorithm is

$$
\begin{align}
\nonumber \theta_{t+1} & = \theta_t + \alpha\mathbb{E}\left[\nabla_\theta\ln\pi(A|S,\theta_t)\left[q_\pi(S,A)-v_\pi(S)\right] \right] \\
\nonumber & \doteq \theta_t + \alpha\mathbb{E}\left[\nabla_\theta\ln\pi(A|S,\theta_t)\delta_\pi(S,A)\right]
\end{align}
$$

where

$$
\delta_\pi(S,A)\doteq q_\pi(S,A)-v_\pi(S)
$$

is called the advantage function.

Futhermore, the advantage function is approximation by the TD error:

$$
\delta_t=q_t(s_t,a_t)-v_t(s_t)\to r_{t+1}+\gamma v_t(s_{t+1})-v_t(s_t)
$$

* This approximation is reasonable because

  $$
  \mathbb{E}[q_\pi(S,A)-v_\pi(S)|S=s_t,A=a_t]=\mathbb{E}[R+\gamma v_\pi(S')-v_\pi(S)|S=s_t, A=a_t]
  $$

* Benefit: only need one work network to approximate $v_\pi(s)$ rather than two networks for $q_\pi(s,a)$ and $v_\pi(s)$

A2C is on-policy

## Off-policy actor-critic

By importance sampling, convert it to off-policy.

### Importance sampling

Note that

$$
\mathbb{E}_{X\sim p_0}[X] = \sum_x p_0(x)x=\sum_x p_1(x)\underbrace{\frac{p_0(x)}{p_1(x)}x}_{f(x)}=\mathbb{E}_{X\sim p_1}[f(X)]
$$

Therefore,

$$
\mathbb{E}_{X\sim p_0}[X]\approx \bar{f}=\frac{1}{n}\sum_{i=1}^n f(x_i)=\frac{1}{n}\sum_{i=1}^n \frac{p_0(x_i)}{p_1(x_i)}x_i
$$

* $\frac{p_0(x_i)}{p_1(x_i)}$ is called importance weight.

### The theorem of off-policy policy gradient

* Suppose $\beta$ is the behavior policy that generates experience samples.
* Our aim is to use these samples to update a target policy $\pi$ that can minimize the metric

  $$
  J(\theta) = \sum_{s\in\mathcal{S}}d_\beta(s)v_\pi(s)=\mathbb{E}_{S\sim d_\beta}[v_\pi(S)]
  $$
  
  where $d_\beta$ is the stationary distribution under policy $\beta$

Theorem (Off-policy policy gradient theorem)

In the discounted case where $\gamma\in(0,1)$ , the gradient of $J(\theta)$ is

$$
\nabla_\theta J(\theta) = \mathbb{E}_{S\sim\rho,A\sim\beta}\left[\frac{\pi(A|S,\theta)}{\beta(A|S)}\nabla_\theta\ln\pi(A|S,\theta)q_\pi(S,A) \right]
$$

where $\beta$ is the behavior policy and $\rho$ is a state distribution.

The off-policy policy gradient is also invariant to a baseline $b(s)$

## Deterministic actor-critic (DPG)

Can we use deterministic policies in the policy gradient methods?

Benefit: it can handle continuous action.

Now, the deterministic policy is specifically denoted as

$$
a=\mu(s,\theta)\doteq\mu(s)
$$

* $\mu$ is a mapping from $\mathcal{S}$ to $\mathcal{A}$
* $\mu$ can be represented by, for example, a neural network with the input as $s$ , the output as $a$ , and the parameter as $\theta$ .
* We may write $\mu(s,\theta)$ in short as $\mu(s)$

Consider the metric of average state value in the discounted case:

$$
J(\theta)=\mathbb{E}[v_\mu(s)]=\sum_{s\in\mathcal{S}}d_0(s)v_\mu(s)
$$

where $d_0(s)$ is a probability distribution satisfying $\sum_{s\in\mathcal{S}}d_0(s)=1$

In the discounted case where $\gamma\in(0,1)$ , the gradient of $J(\theta)$ is

$$
\begin{align}
\nonumber \nabla_\theta J(\theta) & = \sum_{s\in\mathcal{S}}\rho_\mu(s)\nabla_\theta\mu(s)(\nabla_a q_\mu(s,a))|_{a=\mu(s)} \\
\nonumber & = \mathbb{E}_{S\sim\rho_\mu}[\nabla_\theta\mu(S)(\nabla_a q_\mu(S,a))|_{a=\mu(S)}]
\end{align}
$$

Here, $\rho_\mu$ is a state distribution.

It is off-policy.

* How to select the function to represent $q(s,a,w)$
  * Linear function
  * Neural networks: deep deterministic policy gradient (DDPG) method.
