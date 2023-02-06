# Value Function Approximation

from tabular representation to function representation

## Motivating examples: curve fitting

state and action values are represented by tables

* advantage: intuitive and easy to analyze
* disadvantage: difficult to handle large or continuous state or action spaces

Consider an example:

* there are one-dimensional states $s_1,\dots,s_{|\mathcal{S}|}$
* Their state values are $v_\pi(s_1),\dots,v_\pi(s_{|\mathcal{S}|})$ , where $\pi$ is given policy.
* Suppose $|\mathcal{S}|$ is very large and we hope to use a simple curve to approximate these dots to save storage.

We can use high-order polynomial curves or other complex curves to fit the dots.

## Algorithm for state value estimation

### objective function

$$
J(w) = \mathbb{E}\left[ \left(v_\pi(S)-\hat{v}(S,w)\right)^2 \right]
$$

* our goal is to find the best $w$ that can minimize $J(w)$

we can use a **uniform distribution** to approximate the probability distributino of $S$ .

In this case, the objective function becomes

$$
J(w)=\mathbb{E}\left[ \left(v_\pi(S)-\hat{v}(S,w)\right)^2 \right] = \frac{1}{\left|\mathcal{S}\right|}\sum_{s\in\mathcal{S}}\left(v_\pi(s)-\hat{v}(s,w) \right)^2
$$

Drawback: we think all of states are equal, but in fact probabily not.

The second way is to use the stationary distribution

* It describes the **long behavior** of a Markov process
* let $\{d_\pi(s)\}_{s\in\mathcal{S}}$ denote the stationary distribution of the Markov process under policy $\pi$ . By definition, $d_\pi(s)\ge 0$ and $\sum_{s\in\mathcal{S}} d_\pi(s) = 1$ .
* The objective function can be rewritten as

$$
J(w)=\mathbb{E}\left[ \left(v_\pi(S)-\hat{v}(S,w)\right)^2 \right] = \sum_{s\in\mathcal{S}}d_\pi(s)\left(v_\pi(s) -\hat{v}(s,w)\right)^2
$$

This function is a weighted squared error.

The converged values can be predicted because they are the entries of $d_\pi$ :

$$
d_\pi^T=d_\pi^T P_\pi
$$

### optimization algorithms

* To minimize the objective function $J(w)$ , we can use the gradient-descent algorithm :

$$
w_{k+1}=w_k - \alpha_k \nabla_w J(w_k)
$$

We can use the stochastic gradient to replace the true gradient:

$$
w_{t+1} = w_t +\alpha_t(v_\pi(s_t) - \hat{v}(s_t,w_t)\nabla_w \hat{v}(s_t,w_t))
$$

$v_\pi$ is unknown, and we can replace $v_\pi(s_t)$ with an approximation so that the algorithm is implementable.

* Monte Carlo learning with function approximation

  Let $g_t$ be the discounted return starting form $s_t$ in the episode. Then, $g_t$ can be used to approximate $v_\pi(s_t)$ . The algorithm becomes

  $$
  w_{t+1} = w_t + \alpha_t(g_t-\hat{v}(s_t, w_t))\nabla_w \hat{v}(s_t,w_t)
  $$

* Second, TD learning with function approximation.

  $r_{t+1}+\gamma \hat{v}(s_{t+1}, w_t)$ can be viewd as an approximation of $v_\pi(s_t)$ . Then, the algorithm becomes

  $$
  w_{t+1} = w_t + \alpha_t(r_{t+1}+\gamma \hat{v}(s_{t+1},w_t)-\hat{v}(s_t, w_t))\nabla_w \hat{v}(s_t,w_t)
  $$

### Selection of function approximators

How to select the function $\hat{v}(s,w)$

* The first approach, which was widely used before, is to use a linear function

  $$
  \hat{v}(s,w)=\phi^T(s)w
  $$

  Here, $\phi$ is the feature vector

* The second approach is to use a neural network as a nonlinear function approximator.

In the linear case where $\hat{v}(s,w)=\phi^T(s)w$ , we have

$$
\nabla_w\hat{v}(s,w)=\phi(s)
$$

Substituting the gradient into the TD algorithm

$$
w_{t+1}=w_t+\alpha_t[r_{t+1}+\gamma \hat{v}(s_{t+1},w_t)-\hat{v}(s_t,w_t)]\nabla_w \hat{v}(s_t,w_t)
$$

yields

$$
w_{t+1}=w_t+\alpha_t[r_{t+1}+\gamma \phi^T(s_{t+1})w_t-\phi^T(s_t)w_t]\phi(s_t)
$$

which is the algorithm of TD learning with linear function approximation. It is called **TD-linear** in our course in short.

* Disadvantages of linear function approximation:
  * Difficult to select appropriate feature vectors.
* Advantages of linear function approximation:
  * The theoretical properties of the TD algorithm in the linear case can be much better understood than in the nonlinear case.
  * Linear function approximation is still powerful in the sense that the tabular representation is merely a special case of linear function approximation.

We next show that the tabular representation is a special case of linear function approximation.

* First, consider the special feature vector for state $s$ :

  $$
  \phi(s)=e_s \in \mathbb{R}^{|\mathcal{S}|}
  $$

  where $e_s$ is a vector with the sth entry as 1 and the others as 0.

* in this case,

  $$
  \hat{s}(s,w)=e_s^Tw=w(s)
  $$
  
  where $w(s)$ is the sth entry of w.

---

Summary of the story

The story of TD learning with value function approximation.

* This story started from the objective function:

  $$
  J(w)=\mathbb{E}\left[\left(v_\pi(S)-\hat{v}(S,w)\right)^2\right]
  $$

  The objective function suggest that it is a policy evaluation problem.

* The gradient-descent algorithm is

  $$
  w_{t+1}=w_t+\alpha_t(v_\pi(s_t)-\hat{v}(s_t,w_t))\nabla_w \hat{v}(s_t,w_t)
  $$

* The true value function, which is unknown, in the algorithm is replaced by an approximation, leading to the algorthm:

  $$
  w_{t+1}=w_t + \alpha_t[r_{t+1}+\gamma \hat{v}(s_{t+1}, w_t)-\hat{v}(s_t,w_t)]\nabla_w \hat{v}(s_t,w_t)
  $$

Note that the above story is not rigorous mathematically.

## Sarsa with function approximation

The Sarsa algorithm with value function approximation is

$$
w_{t+1}=w_t + \alpha_t\left[r_{t+1}+\gamma \hat{q}(s_{t+1},a_{t+1}, w_t) - \hat{q}(s_t,a_t,w_t) \right]\nabla_w\hat{q}(s_t,a_t,w_t)
$$

## Q-learning with function approximation

The q-value update rule is:

$$
w_{t+1}=w_t + \alpha_t\left[r_{t+1}+\gamma\max_{a\in\mathcal{A}(s_{t+1})} \hat{q}(s_{t+1},a,w_t) - \hat{q}(s_t,a_t,w_t) \right]\nabla_w\hat{q}(s_t,a_t,w_t)
$$

which is the same as Sarsa except that $\hat{q}(s_{t+1},a_{t+1},w_t)$ is replaced by $\max_{a\in\mathcal{A}(s_{t+1})}\hat{q}(s_{t+1},a,w_t)$

## Deep Q-learning

Deep Q-learning or deep Q-network (DQN):

The role of neural networks is to be a nonlinear function approximator.

Deep Q-learning aims to minimize the objective function/loss function:

$$
J(w)=\mathbb{E}\left[\left(R+\gamma \max_{a\in\mathcal{A}(S')} \hat{q}(S',a,w) - \hat{q}(S,A,w) \right)^2 \right]
$$

where $(S,A,R,S')$ are random variables.

* This is actually the Bellman optimality error. That is because

  $$
  q(s,a)=\mathbb{E}\left[R_{t+1}+\gamma \max_{a\in\mathcal{A}(S_{t+1})} q(S_{t+1},a) \mid S_t=s, A_t=a \right] \quad \forall s,a
  $$

  The value of $R+\gamma \max_{a\in\mathcal{A}(S')} \hat{q}(S',a,w) - \hat{q}(S,A,w)$ should be zero in the expectation sense.

To do that, we can introduce two network.

* One is a main network representing $\hat{q}(s,a,w)$
* The other is a target network $\hat{q}(s,a,w_T)$

The objective function in this case degenerates to

$$
J(w)=\mathbb{E}\left[\left(R+\gamma \max_{a\in\mathcal{A}(S')} \hat{q}(S',a,w_T) - \hat{q}(S,A,w) \right)^2 \right]
$$

where $w_T$ is the target network parameter.

When $w_T$ is fixed, the gradient of $J$ can be easily obtained as

$$
\nabla_wJ=\mathbb{E}\left[\left(R+\gamma \max_{a\in\mathcal{A}(S')} \hat{q}(S',a,w_T) - \hat{q}(S,A,w) \right)\nabla_w\hat{q}(S,A,w) \right]
$$

---

First technique:

* Two networks, a main network and a target network

Why is it used?

* When calculating the gradient, we has problems.

Implementation details:

* Let $w$ and $w_T$ denote the parameters of the main and target networks, respectively. They are set to be the same initially.
* In every iteration, we draw a mini-batch of samples $\{(s,a,r',s')\}$ from the replay buffer.
* The inputs of the networks include state $s$ and action $a$ . The target output is $y_T\doteq r+\gamma\max_{a\in\mathcal{A}(s')}\hat{q}(s',a,w_T)$ . Then, we directly minimize the TD error or called loss function $\left( y_T-\hat{q}(s,a,w)\right)^2$ over the mini-batch $\{(s,a,y_T)\}$ .

Experience replay

* After we have collected some experience samples, we do NOT use these samples in the order they were collected.
* Instead, we store them in a set, called replay buffer $\mathcal{B}\doteq\{(s,a,r',s')\}$
* Every time we train the neural network, we can draw a mini-batch of random samples from the replay buffer.
* The draw of samples, or called experience replay, should follow a uniform distribution
