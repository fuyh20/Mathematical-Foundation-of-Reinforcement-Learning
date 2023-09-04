# Bellman Equation

In this lecture:

* A core concept: state value
* A fundamental tool: the Bellman equation

## Motivating examples: why return is important?

Return could be used to evaluate policies.

The returns rely on each other. **Bootstrapping**

* the value of one state relies on the values of other states
* a matrix-vector form is useful

## State value

Consider the following single-step process:

$$
S_t\xrightarrow{A_t}R_{t+1}, S_{t+1}
$$

* $t, t+1$ : discrete time instances
* $S_t$ : state at time t
* $A_t$ : the action taken at state $S_t$
* $R_{t+1}$ : the reward obtained after taking $A_t$
* $S_{t+1}$ : the state transited to after taking $A_t$

Note that $S_t, A_t, R_{t+1}$ are all **random variables**

This step is governed by the following probability distributinos:

* $S_t \rightarrow A_t$ is governed by $\pi(A_t = a|S_t = s)$
* $S_t, A_t \rightarrow R_{t+1}$ is governed by $p(R_{t+1} = r|S_t = s, A_t = a)$
* $S_t, A_t \rightarrow S_{t+1}$ is governed by $p(S_{t+1} = s'|S_t = s, A_t = a)$

Consider the following multi-step trajectory:

$$
S_t\xrightarrow{A_t}R_{t+1}, S_{t+1}\xrightarrow{A_{t+1}}R_{t+2}, S_{t+2}\xrightarrow{A_{t+2}}R_{t+3},\dots
$$

The discounted return is

$$
G_t=R_{t+1}+\gamma R_{t+2} + \gamma^2 R_{t+3}+\dots
$$

* $\gamma \in [0,1)$ is a discount rate
* $G_t$ is also random variable since $R_{t+1}, R_{t+2}, \dots$ are random variable.

The expectation (or called expected value or mean) of $G_t$ is defined as the **state-value function** or simply state value:

$$
v_\pi(s)=\mathbb{E}[G_t|S_t = s]
$$

Remarks:

* It is a function of $s$ . It is a conditional expectation with the condition that the state starts from $s$ .
* It is based on the policy $\pi$ . For a different policy, the state value may be different
* It represents the "value" of a state. If the state is greater, then the policy is better because greater cumulative rewards can be obtained.

Return is for one trajectory, but state value is mean of many trajectories's Return starting from one state.

## Bellman equation: Derivation

how to calculate state value? Bellman equation

In a word, the Bellman equation describes the relationship among the values of all states.

Consider a random trajectory:

$$
S_t\xrightarrow{A_t}R_{t+1}, S_{t+1}\xrightarrow{A_{t+1}}R_{t+2}, S_{t+2}\xrightarrow{A_{t+2}}R_{t+3},\dots
$$

The return $G_t$ can be written as:

$$
\begin{align}
\nonumber G_t & = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots \\
\nonumber     & = R_{t+1} + \gamma (R_{t+2} + R_{t+3} + \dots) \\
\nonumber     & = R_{t+1} + \gamma G_{t+1}
\end{align}
$$

Then, it follows from the definition of the state value that

$$
\begin{align}
\nonumber v_\pi(s) & = \mathbb{E}[G_t|S_t=s] \\
\nonumber          & = \mathbb{E}[R_{t+1} + \gamma G_{t+1}| S_t = s] \\
\nonumber          & = \mathbb{E}[R_{t+1}|S_t = s] + \gamma\mathbb{E}[G_{t+1}|S_t = s]
\end{align}
$$

Next, calculate the two terms, respectively.

First, calculate the first term $\mathbb{E}[R_{t+1}|S_t=s]$ :

$$
\begin{align}
\nonumber \mathbb{E}[R_{t+1}|S_t=s] & = \sum_a\pi(a|s)\mathbb{E}[R_{t+1}|S_t=s,A_t=a] \\
\nonumber                           & = \sum_a\pi(a|s)\sum_rp(r|s,a)r
\end{align}
$$

This is the mean of immediate rewards.

Second, calculate the second term $\mathbb{E}[G_{t+1}|S_t=s]$

$$
\begin{align}
\nonumber \mathbb{E}[G_{t+1}|S_t=s] & = \sum_{s'}\mathbb{E}[G_{t+1}|S_t=s, S_{t+1}=s']p(s'|s) \\
\nonumber & = \sum_{s'}\mathbb{E}[G_{t+1}|S_{t+1}=s']p(s'|s) \\
\nonumber & = \sum_{s'}v_\pi(s')p(s'|s) \\
\nonumber & = \sum_{s'}v_\pi(s')\sum_a p(s'|s, a)\pi(a|s)
\end{align}
$$

Note that:

* This is the mean of future rewards
* $\mathbb{E}[G_{t+1}|S_t=s, S_{t+1}=s']=\mathbb{E}[G_{t+1}|S_{t+1}=s']$ due to the memoryless Markov property.

Therefore, we have

$$
\begin{align}
\nonumber v_\pi(s) & = \mathbb{E}[R_{t+1}|S_t=s]+\gamma\mathbb{E}[G_{t+1}|S_t=s] \\
\nonumber & = \sum_a\pi(a|s)\sum_rp(r|s,a)r + \gamma \sum_a\pi(a|s)\sum_{s'}p(s'|s, a)v_\pi(s') \\
\nonumber & = \sum_a\pi(a|s)\left[\sum_rp(r|s,a)r+\gamma\sum_{s'}p(s'|s,a)v_\pi(s') \right],\quad \forall s\in\mathcal{S}
\end{align}
$$

Highlights:

* The above equation is called the Bellman equation, which characterizes the relationship among the state-value functions of different states.
* It consists of two terms: the immediate reward term and the future reward term.
* A set of equations: every state has an equation like this.

symbols in this equation:

* $v_\pi(s)$ and $v_\pi(s')$ are state values to be calculated. Bootstrapping!
* $\pi(a|s)$ is a given policy. Solving the equation is called policy evaluation.
* $p(r|s,a)$ and $p(s'|s,a)$ represent the dynamic model.

## Bellman equation: Matrix-vector form

* How to solve the Bellman equation?
  
  One unknown relies on another unknown.

$$
v_\pi(s) = \sum_a\pi(a|s)\left[\sum_rp(r|s,a)r+\gamma\sum_{s'}p(s'|s,a)v_\pi(s') \right],\quad \forall s\in\mathcal{S}
$$

* The above elementwise form is valid for every state $s\in\mathcal{S}$ . That means there are $|\mathcal{S}|$ equations like this.
* If we put all the equations together, we have a set of linear equations, which can be concisely written in a **matrix-vector form**

Recall that:

$$
v_\pi(s) = \sum_a\pi(a|s)\left[\sum_rp(r|s,a)r+\gamma\sum_{s'}p(s'|s,a)v_\pi(s') \right]
$$

Rewrite the Bellman equation as

$$
v_\pi(s) = r_\pi(s) + \gamma\sum_{s'}p_\pi(s'|s)v_\pi(s')
$$

where

$$
r_\pi(s)\triangleq\sum_a\pi(a|s)\sum_rp(r|s,a)r
$$

$$
p_\pi(s'|s)\triangleq\sum_a\pi(a|s)p(s'|s,a)
$$

Suppose the states could be indexed as $s_i(i=1,\dots,n)$

For state $s_i$, the Bellman equation is

$$
v_\pi(s_i) = r_\pi(s_i) + \gamma\sum_{s_j}p(s_j|s_i)v_\pi(s_j)
$$

Put all these equations for all the states together and rewrite to a matrix-vector form

$$
v_\pi = r_\pi + \gamma P_\pi v_\pi
$$

where

* $v_\pi = [v_\pi(s_1), \dots, v_\pi(s_n)]^T\in \mathbb{R}^n$
* $r_\pi = [r_\pi(s_1), \dots, r_\pi(s_n)]^T\in \mathbb{R}^n$
* $P_\pi \in \mathbb{R}^{n\times n}$ , where $[P_\pi]_{ij} = p_\pi(s_j|s_i)$ , is the state transition matrix

## Bellman equation: Solve the state values

Why to solve state values?

* Given a policy, finding out the corresponding state values is called **policy evaluation**. It is a fundamental problem in RL. It is the foundation to find better policies.
* It is important to understand how to solve the Bellman equation.

solution

* The *closed-form* solution is

$$
v_\pi = (I-\gamma P_\pi)^{-1}r_\pi
$$

* An iterative solution is

$$
v_{k+1} = r_\pi + \gamma P_\pi v_k
$$

This algorithm leads to a sequence $\{v_0, v_1, v_2, \dots\}$ . We can show that

$$
v_k \to v_\pi = (I-\gamma P_\pi)^{-1}, k\to \infty
$$

## Action value

From state value to action value:

* State value: the average return the agent can get starting from a state.
* Action value: the average return the agent can get starting from a state and taking an action.

We can know which action is better through action value.

Definition

$$
q_\pi(s, a) = \mathbb{E}[G_t|S_t = s, A_t = a]
$$

* $q_\pi(s, a)$ is a function of the state-action pair $(s,a)$
* $q_\pi(s, a)$ depends on $\pi$

It follows from the properties of conditional expectation that

$$
\mathbb{E}[G_t|S_t=s] = \sum_a\mathbb{E}[G_t|S_t=s,A_t=a]\pi(s|a)
$$

Hence

$$
v_\pi(s) = \sum_a \pi(s|a) q_\pi(s,a)
$$

Recall that the state value is given by

$$
v_\pi(s) = \sum_a\pi(a|s)\left[\sum_rp(r|s,a)r+\gamma\sum_{s'}p(s'|s,a)v_\pi(s') \right]
$$

By comparing the above two formulas, we have the **action-value function** as

$$
q_\pi(s,a) = \sum_r p(r|s,a)r + \gamma \sum_{s'} p(s'|s,a)v_\pi(s')
$$
