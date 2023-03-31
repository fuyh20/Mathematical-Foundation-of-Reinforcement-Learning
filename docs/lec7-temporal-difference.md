# Temporal-Difference Learning

from non-incremental to incremental

## Motivating examples

That is to estimate the mean of a function $v(X)$

$$
w = \mathbb{E}[R+\gamma v(X)]
$$

where $R,X$ are random variables, $\gamma$ is a constant, and $v(\cdot)$ is a function.

Suppose we can obtain samples ${x}$ and ${r}$ of $X$ and $R$. we define

$$
g(w)=w-\mathbb{E}[R+\gamma v(X)]
$$

$$
\begin{align}
\nonumber \tilde{g}(w,\eta) & =w-[r + \gamma v(x)] \\
\nonumber & =(w-\mathbb{E}[R+\gamma v(X)])+(\mathbb{E}[R+\gamma v(X)]-[r + \gamma v(x)]) \\
\nonumber & \doteq g(w)+\eta
\end{align}
$$

Then, the problem becomes a root-finding problem: $g(w)=0$ . The corresponding RM algorithm is

$$
w_{k+1}=w_k- \alpha_k\tilde{g}(w_k, \eta_k) = w_k-\alpha_k[w_k-(r_k + \gamma v(x_k))]
$$

## TD algorithm of state values

The data/experience required by the algorithm:

$(s_0, r_1, s_1, \dots, s_t, r_{t+1}, s_{t+1})$ or $\{(s_t, r_{r+1}, s_{t+1}\}_t$ generated following the given policy $\pi$

The TD learning algorithm is

$$
\begin{align}
& v_{t+1}(s_t)=v_t(s_t)-\alpha_t(s_t)\left[v_t(s_t)-\left[r_t+\gamma v_t(s_{t+1})\right]\right] \\
& v_{t+1}(s)=v_t(s) \quad \forall s \ne s_t
\end{align}
$$

where $t=0,1,2,\dots$ Here, $v_t(s_t)$ is the estimated state value of $v_\pi(s_t)$ , $\alpha_t(s_t)$ is the learning rate of s_t at time $t$ .

* At time $t$ , only the value of the visited state $s_t$ is updated whereas the values of the unvisited states $s\ne s_t$ remain unchanged.

The TD algorithm can be annotated as

$$
\underbrace{v_{t+1}(s_t)}_{\mathrm{new\ estimate}} = \underbrace{v_t(s_t)}_{\mathrm{current\ estimate}} - \alpha_t(s_t) \overbrace{\left[ v_t(s_t)- \underbrace{\left[ r_{t+1}+\gamma v_t(s_{t+1})\right]}_{\mathrm{TD\ target\ }\bar{v}_t}\right]}^{\mathrm{TD\ error\ }\delta_t}
$$

Here,

$$
\bar{v}_t\doteq r_{t+1}+\gamma v(s_{t+1})
$$

is called the TD target.

$$
\delta_t\doteq v(s_t)-\left[ r_{t+1} + \gamma v(s_{t+1}) \right] = v(s_t)-\bar{v}_t
$$

is called the TD error.

It is clear that new estimate $v_{t+1}(s_t)$ is a combination of the current estimate $v_t(s_t)$ and the TD error.

---

First, why is $\bar{v}_t$ is called the TD target?

That is because the algorithm drives $v(s_t)$ towards $\bar{v}_t$

To see that,

$$
\begin{align}
\nonumber & v_{t+1}(s_t) = v_t(s_t) - \alpha_t(s_t)\left[ v_t(s_t) - \bar{v}_t \right] \\
\nonumber \Rightarrow & v_{t+1}(s_t) - \bar{v}_t = v_t(s_t) -\bar{v}_t- \alpha_t(s_t)\left[ v_t(s_t) - \bar{v}_t \right] \\
\nonumber \Rightarrow & v_{t+1}(s_t) - \bar{v}_t = \left[ 1 - \alpha_t(s_t) \right]\left[ v_t(s_t) - \bar{v}_t \right] \\
\nonumber \Rightarrow & \left|v_{t+1}(s_t) - \bar{v}_t\right| = \left| 1 - \alpha_t(s_t) \right|\left| v_t(s_t) - \bar{v}_t \right|
\end{align}
$$

Since $\alpha_t(s_t)$ is a small positive number, we have

$$
0 < 1 - \alpha_t(s_t) < 1
$$

Therefore,

$$
\left|v_{t+1}(s_t) - \bar{v}_t\right| \le \left|v_t(s_t) - \bar{v}_t\right|
$$

which means $v(s_t)$ is driven towards $\bar{v}_t$

---

Second, what is the interpretation of the TD error?

$$
\delta_t = v(s_t) - \left[r_{t+1} + \gamma v(s_{t+1})\right]
$$

It reflects the deficiency between $v_t$ and $v_\pi$ . To see that, denote

$$
\delta_{\pi, t}\doteq v_\pi(s_t) - \left[ r_{t+1} + \gamma v_\pi(s_{t+1}) \right]
$$

Note that

$$
\mathbb{E}[\delta_{\pi,t}|S_t=s_t]=v_\pi(s_t)-\mathbb{E}[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t = s_t] = 0
$$

* if $v_t=v_\pi$ , then $\delta_t$ should be zero (in expectation sense)
* hence, if $\delta_t$ is not zero, then $v_t$ is not equal to $v_\pi$

The TD error can be interpreted as innovation, which means new information obtained from the experience $(s_t, r_{t+1}, s_{t+1})$

---

* TD algorithm solves the Bellman equation of a given policy $\pi$ without model.

a new expression of the Bellman equation

$$
v_\pi(s) = \mathbb{E}[R+\gamma v_\pi(S')|S=s]\quad s\in\mathcal{S}
$$

we use RM algorithm to solve the above equation

$$
g(v(s))=v(s)-\mathbb{E}[R+\gamma v_\pi(S')|s]
$$

so the Bellman equation can be rewrited as

$$
g(v(s))=0
$$

$$
v_{k+1}(s)=v_k(s)-\alpha_k\left( v_k(s)-\left[r_k+\gamma v_\pi(s_k')  \right] \right)
$$

TD learning is online, MC learning is offline.

## Sarsa

It can directly estimate action values.

Suppose we have some experience $\{(s_t,a_t,r_{t+1},s_{t+1},a_{t+1}) \}_t$

We can use the following Sarsa algorithm to estimate the action values:

$$
\begin{align}
\nonumber & q_{t+1}(s_t,a_t)=q_t(s_t,a_t)-\alpha_t(s_t,a_t)\left[ q_t(s_t,a_t)-\left[ r_{t+1}+\gamma q_t(s_{t+1}, a_{t+1}) \right] \right] \\
\nonumber & q_{t+1}(s,a) = q_t(s,a) \quad \forall (s,a) \ne (s_t,a_t)
\end{align}
$$

where $t=0,1,2,\dots$

* $q_t(s_t,a_t)$ is an estimate of $q_\pi(s_t,a_t)$
* $\alpha_t(s_t,a_t)$ is the learning rate depending on $s_t,a_t$

Sarsa is the abbreviation of state-action-reward-state-action.

It is a stochastic approximation algorithm solving the following equation:

$$
q_\pi(s,a)=\mathbb{E}[R+\gamma q_\pi(S',A')|s,a] \quad \forall s, a
$$

## Expected Sarsa and n-step Sarsa

### Expected Sarsa

$$
\begin{align}
\nonumber & q_{t+1}(s_t,a_t)=q_t(s_t,a_t)-\alpha_t(s_t,a_t)\left[ q_t(s_t,a_t)-\left[ r_{t+1}+\gamma\mathbb{E}[ q_t(s_{t+1},A)] \right] \right] \\
\nonumber & q_{t+1}(s,a) = q_t(s,a) \quad \forall (s,a) \ne (s_t,a_t)
\end{align}
$$

where

$$
\mathbb{E}[q_{t+1}(s_{t+1}, A)] = \sum_a \pi_t(a|s_{t+1})q_t(s_{t+1},a)\doteq v_t(s_{t+1})
$$

is the expected value of $q_t(s_{t+1}, a)$ under policy $\pi_t$

### n-step Sarsa

$$
\begin{align}
\nonumber & q_{t+1}(s_t,a_t)=q_t(s_t,a_t)-\alpha_t(s_t,a_t)\left[ q_t(s_t,a_t)-\left[ r_{t+1}+\gamma r_{t+2}+\cdots+\gamma^n q_t(s_{t+n},a_{t+n}) \right] \right] \\
\nonumber & q_{t+1}(s,a) = q_t(s,a) \quad \forall (s,a) \ne (s_t,a_t)
\end{align}
$$

## Q-learning

TD learning of optimal action values

$$
\begin{align}
\nonumber & q_{t+1}(s_t,a_t)=q_t(s_t,a_t)-\alpha_t(s_t,a_t)\left[ q_t(s_t,a_t)-\left[ r_{t+1}+\gamma \max_{a\in\mathcal{A}}q_t(s_{t+1}, a) \right] \right] \\
\nonumber & q_{t+1}(s,a) = q_t(s,a) \quad \forall (s,a) \ne (s_t,a_t)
\end{align}
$$

It aims to solve

$$
q(s,a)=\mathbb{E}\left[ R_{t+1} + \gamma \max_a q(S_{t+1}, a)\mid S_t=s, A_t=a \right]
$$

This is the Bellman optimality equation expressed in terms of action values.

There exist two policies in a TD learning task:

* The behavior policy is used to generate experience samples.
* The target policy is constantly updated toward an optimal policy.

on-policy vs off-policy:

* when the behavior policy is the same as the target policy, such kind of learning is called on-policy.
* when they are different, the learning is called off-policy.

Advantages of off-policy learning

* It can search for optimal policies based on the experience samples generated by any other policies.

How to judge if a TD algorithm is on-policy or off-policy?

* First, check what the algorithm does mathematically.
* Second, check what things are required to implement the algorithm.

Sarsa is on-policy.

Monte Carlo Learning is on-policy.

Q-learning is off-policy.

---

Pseudocode: Optimal policy search by Q-learning (off-policy version)

For each episode ${s_0,a_0,r_1,s_1,a_1,r_2,\dots}$ generated by $\pi_b$ , do

For each step $t=0,1,2,\dots$ of the episode, do

Update q-value:

$$
q_{t+1}(s_t,a_t)=q_t(s_t,a_t)-\alpha_t(s_t,a_t)\left[ q_t(s_t,a_t)-\left[ r_{t+1}+\gamma \max_{a\in\mathcal{A}}q_t(s_{t+1}, a) \right] \right]
$$

Update target policy:

$$
\begin{align}
\nonumber & \pi_{T,t+1}(a|s_t)=1\ \mathrm{if}\ a=\argmax_a q_{t+1}(s_t,a) \\
\nonumber & \pi_{T,t+1}(a|s_t)=0\  \mathrm{otherwise}
\end{align}
$$

---

All the algorithms can be expressed in a unified expression:

$$
q_{t+1}(s_t,a_t)=q_t(s_t,a_t)-\alpha_t(s_t,a_t) [q_t(s_t,a_t)-\bar{q}_t]
$$

where $\bar{q}_t$ is the TD target.

All these algorithms have different TD target, and they are to solve BE or BOE.
