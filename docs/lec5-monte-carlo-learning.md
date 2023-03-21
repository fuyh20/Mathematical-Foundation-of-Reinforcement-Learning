# Monte Carlo Learning

From model-based to model-free

## Motivating example

How can we estimate something **without models**?

The simplest idea: Monte Carlo estimation.

> Example: Flip a coin
>
> The result (either head or tail) is denoted as a random variable $X$
>
> * if the result is head, then $X = +1$
> * if the result is tail, then $X = -1$
>
> The aim is to compute $\mathbb{E}[X]$
>
> Method 1: Model-based
> > Suppose the probabilistic model is known as
> >
> > $$
> > p(X=1)=0.5 \quad p(X=-1)=0.5
> > $$
> >
> > Then by definition
> >
> > $$
> > \mathbb{E}[X]=\sum_x xp(x)=1\times0.5=-1\times0.5=0
> > $$
> >
> > Problem: it may be impossible to know the precise distribution!!
>
> Method 2: Model-free
> > Idea: Flip the coin many times, and then calculate the average of outcomes
> >
> > Suppose we get a sample sequence: $\{x_1,x_2,\dots,x_N \}$
> > Then, then mean can be approximated as
> >
> > $$
> > \mathbb{E}[X]\approx \bar{x} = \frac1N \sum_{j=1}^N x_j
> > $$
> >
> > This is the idea of Monte Carlo estimation

Math: **Law of Large Numbers**

Why we care about mean estimation?

Because state value and action value are defined as expectations of random variables.

## The simplest MC-based RL algorithm

To understand how to convert the policy iteration algorithm to be model-free

Policy iteration has two steps in each iteration:

$$
\begin{cases}
\mathrm{Policy\ evaluation}: v_{\pi_k} = r_{\pi_k}+\gamma P_{\pi_k}v_{\pi_k} \\
\mathrm{Policy\ improvement}: \pi_{k+1} = \argmax_\pi(r_\pi+\gamma P_\pi v_{\pi_k})
\end{cases}
$$

The elementwise form of the policy improvement step is:

$$
\pi_{k+1}(s)=\argmax_\pi\sum_a \pi(a|s)q_{\pi_k}(s,a) \quad s\in\mathcal{S}
$$

The key is $q_{\pi_k}(s,a)$

Two expression of action value:

* Expression 1 requires the model

$$
q_{\pi_k}(s,a)=\sum_r p(r|s,a)r + \gamma \sum_{s'}p(s'|s,a)v_{\pi_k}(s')
$$

* Expression 2 does not require the model

$$
q_{\pi_k}(s,a)=\mathbb{E}[G_t|S_t=s,A_t=a]
$$

Idea to achieve model-free RL: We can use expression 2 to calculate $q_{\pi_k}(s,a)$ based on data (samples or experiences)

The procedure of Monte Carlo estimation of action values:

* Starting from $(s,a)$ , following policy $\pi_k$ , generate an episode.
* The return of this episode is $g(s,a)$
* $g(s,a)$ is a sample of $G_t$ in

$$
q_{\pi_k}(s,a) = \mathbb{E}[G_t|S_t=s,A_t=a]
$$

* Suppose we have a set of episodes and hence $\{g^{(j)}(s,a)\}$ , then

$$
q_{\pi_k}(s,a)=\mathbb{E}[G_t|S_t=s,A_t=a]\approx \frac1N\sum_{i=1}^N g^{(i)}(s,a)
$$

Fundamental idea: when model is unavailable, we can use data.

---

MC Basic algorithm

Given an initial policy $\pi_0$ , there are two steps at the kth iteration

* step 1: policy evaluation. This step is to obtain $q_{\pi_k}(s,a)$ for all $(s,a)$ . Specifically, for each action-value pair $(s,a)$ , run an infinite number of (or sufficiently many) episodes. The average of their returns is used to approximate $q_{\pi_k}(s,a)$
* step 2: policy improvement. This step is to solve $\pi_{k+1}(s) = \argmax_\pi\sum_a\pi(a|s)q_{\pi_k}(s,a)$ for all $s\in\mathcal{S}$ . The greedy optimal policy is $\pi_{k+1}(a_k^*|s)=1$ where $a_k^*=\argmax_a q_{\pi_k}(s,a)$

Exactly the same as the policy iteration algorithm, except Estimate $q_{\pi_k}(s,a)$ directly, instead of solving $v_{\pi_k}(s)$

---

MC Basic is not practical due to low efficiency

The impact of episode length

When the episode length is short, only the states that are close to the target have nonzero state values.

## MC Exploring Starts

use data more efficiently

Visit: every time a state-action pair appears in the episode, it is called a visit of that state-action pair.

Data-efficient methods:

* first-visit method: only the visit that emerge first will be used to estimate the action value.
* every-visit method: every visit will be used to estiamte the action value.

Another aspect is when to update the policy

* The first method is to collect all the episodes starting from a state-pair and use the average return to approximate the action value.
* The second method uses the return of a single episode to approximate the action value. **episode by episode**

## MC $\epsilon$ - Greedy

A policy is called soft if the probability to take any action is positive.

Why introduce soft policies?

* With a soft policy, a few episodes that are sufficiently long can visit every state-action pair for sufficiently many times
* We don't need to have a large number of episodes starting from every state-action pair.

$\epsilon$ - greedy policy

$$
\pi(a|s)=\begin{cases}
    1-\frac{\epsilon}{\left| \mathcal{A}(s) \right|}\left(\left| \mathcal{A}(s) \right|-1\right) & \mathrm{for\ the\ greedy\ action} \\
\frac{\epsilon}{\left| \mathcal{A}(s) \right|} & \mathrm{for\ the\ other\ } \left| \mathcal{A}(s) \right| -1 \ \mathrm{actions}
\end{cases}
$$

where $\epsilon\in[0,1]$ and $|\mathcal{A}(s)|$ is the numebr of actions for $s$

why use $\epsilon$ - greedy? Balance between exploitation and exploration.

The greater $\epsilon$ is, the more model prefers exploration. But this will not get optimal policy.

Now, the policy improvement step is changed to solve

$$
\pi_{k+1}(s)=\argmax_{\pi\in\Pi_\epsilon}\sum_a \pi(a|s)q_{\pi_k}(s,a)
$$

where $\Pi_\epsilon$ denotes the set of all $\epsilon$ - greedy polices with a fixed value of $\epsilon$

The optimal policy here is

$$
\pi_{k+1}(a|s)=\begin{cases}
    1-\frac{\epsilon}{\left| \mathcal{A}(s) \right|}\left(\left| \mathcal{A}(s) \right|-1\right) & a = a_k^* \\
\frac{\epsilon}{\left| \mathcal{A}(s) \right|} & a \ne a_k^*
\end{cases}
$$

When $\epsilon$ increases, the optimality of policy becomes worse.
