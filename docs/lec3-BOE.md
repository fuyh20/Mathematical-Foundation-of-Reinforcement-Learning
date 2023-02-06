# Optimal Policy and Bellman Optimality Equation

In this lecture:

* Core concepts: optimal state value and optimal policy
* A fundamental tool: the Bellman optimality equation (BOE)

## Optimal policy

The state value could be used to evaluate if a policy is good or not: if

$$
v_{\pi_1}(s) \ge v_{\pi_2}(s), \quad \forall s\in\mathcal{S}
$$

then $\pi_1$ is "better" than $\pi_2$

Definition

A policy $\pi^*$ is optimal if $v_{\pi^*}(s) \ge v_\pi(s)$ for all $s$ and for any other policy $\pi$ .

The definition leads to many questions:

* Does the optimal policy exist?
* Is the optimal policy unique?
* Is the optimal policy stochastic or deterministic?
* How to obtain the optimal policy?

## Bellman optimality equation (BOE)

Bellman optimality equation (elementwise form):

$$
\begin{align}
\nonumber v(s) & = \max_\pi\sum_a\pi(a|s)\left[\sum_rp(r|s,a)r+\gamma\sum_{s'}p(s'|s,a)v(s') \right],\quad \forall s\in\mathcal{S} \\
\nonumber & = \max_\pi\sum_a \pi(a|s)q(s, a) \quad s\in\mathcal{S}
\end{align}
$$

Remarks:

* $p(r|s,a),p(s'|s,a)$ are known.
* $v(s),v(s')$ are unknown and to be calculated.

Bellman optimality equation (matrix-vector form)

$$
v=\max_\pi(r_\pi+\gamma P_\pi v)
$$

where the elements corresponding to $s$ or $s'$ are

$$
[r_\pi]_s\triangleq\sum_a\pi(a|s)\sum_rp(r|s,a)r
$$

$$
[P_\pi]_{s,s'}=p(s'|s)\triangleq\sum_a\pi(a|s)\sum_{s'}p(s'|s,a)
$$

Here $\max_\pi$ is performed elementwise.

## BOE: Maximization on the right-hand side

Bellman optimality equation (elementwise form):

$$
v(s) = \max_\pi\sum_a\pi(a|s)\left[\sum_rp(r|s,a)r+\gamma\sum_{s'}p(s'|s,a)v(s') \right],\quad \forall s\in\mathcal{S}
$$

Bellman optimality equation (matrix-vector form)

$$
v=\max_\pi(r_\pi+\gamma P_\pi v)
$$

How to calculate?

> Example
>
> Consider two variable $x,a\in\mathbb{R}$. Suppose they satisfy
>
> $$
x=\max_a(2x-1-a^2)
> $$
>
> Regardless the value of $x$ , $\max_a(2x-1-a^2)=2x-1$ where the maximization is achieved when $a=0$ . Second, when $a = 0$ , the equation becomes $x=2x-1$ , which leads to $x=1$ . Therefore, $a=0$ and $x=1$ are the solution of the equation.

Fix $v'(s)$ first and solve $\pi$

$$
\begin{align}
\nonumber v_\pi(s) & = \max_\pi\sum_a\pi(a|s)\left[\sum_rp(r|s,a)r+\gamma\sum_{s'}p(s'|s,a)v(s') \right] \\
\nonumber & = \max_\pi\sum_a \pi(a|s)q(s, a) \quad s\in\mathcal{S}
\end{align}
$$

> Example (How to solve $\max_\pi\sum_a\pi(a|s)q(s,a)$ )
>
> Suppose $q_1, q_2, q_3 \in \mathbb{R}$ are given. Find $c_1^*,c_2^*,c_3^*$ solving
>
> $$
> \max_{c_1, c_2, c_3}c_1q_1+c_2q_2+c_3q_3
> $$
>
> where $c_1+c_2+c_3=1$ and $c_1,c_2,c_3\ge0$ . Then, the optimal solution is $c_3^*=1$ , and $c_1^* = c_2^* = 0$ . That is because for any $c_1,c_2,c_3$
>
> $$
> q_3=(c_1+c_2+c_3)q_3\ge c_1q_3+c_2q_2+c_3q_3
> $$
>

Inspired by the above example, considering that $\sum_a\pi(a|s)=1$ , we have

$$
\max_\pi\sum_a\pi(a|s)q(s,a) = \max_{a\in\mathcal{A}(s)}q(s,a)
$$

where the optimality is achieved when

$$
\pi(a|s)=\begin{cases}
    1, & a = a^* \\
    0, & a \ne a^*
\end{cases}
$$

where $a^* = \argmax_a q(s,a)$

## BOE: Rewrite as $v=f(v)$

The BOE is $v = \max_\pi (r_\pi + \gamma P_\pi v)$. Let

$$
f(v) \coloneqq \max_\pi (r_\pi + \gamma P_\pi v)
$$

Then, the Bellman optimality equation becomes

$$
v=f(v)
$$

where

$$
[f(v)]_s = \max_\pi\sum_a \pi(a|s)q(s,a), \quad s\in\mathcal{S}
$$

## Contraction mapping theorem

Some concepts:

* Fixed point: $x\in X$ is a fixed point of $f: X \to X$ if

$$
f(x) = x
$$

* Contraction mapping (or contractive function): $f$ is a contraction mapping if

$$
\|f(x_1)-f(x_2)\| \le \gamma \|x_1-x_2\|
$$

where $\gamma\in(0,1)$

Theorem (Contraction Mapping Theorem)

For any equation that has the form of $x=f(x)$ , if $f$ is a contraction mapping, then

* Existence: there exists a fixed point $x^*$ satisfying $f(x^*)=x$
* Uniqueness: the fixed point $x^*$ is unique
* Algorithm: Consider a sequence ${x_k}$ where $x_{k+1} = f(x_k)$ , then $x_k \to x^*$ as $k\to\infty$ . Moreover, the convergence rate is exponentially fast.

## BOE: solution

Let's come back to the Bellman optimality equation

$$
v=f(v)=\max_\pi(r_\pi+\gamma P_\pi v)
$$

$f$ is contraction mapping

For the BOE $v=f(v)=\max_\pi(r_\pi+\gamma P_\pi v)$ , there always exists a solution $v^*$ and the solution is unique. The solution could be solved iteratively by

$$
v_{k+1}=f(v_k) = \max_\pi(r_\pi + \gamma P_\pi v_k)
$$

This sequence ${v_k}$ converges to $v^*$ exponentially fast given any initial guess $v_0$ . The convergence rate is determined by $\gamma$

The iterative algorithm

Matrix-vector form:

$$
v_{k+1}=f(v_k) = \max_\pi(r_\pi + \gamma P_\pi v_k)
$$

Elementwise form:

$$
\begin{align}
\nonumber v_{k+1}(s) & = \max_\pi\sum_a\pi(a|s)\left[\sum_rp(r|s,a)r+\gamma\sum_{s'}p(s'|s,a)v_k(s') \right] \\
\nonumber & = \max_\pi\sum_a \pi(a|s)q_k(s, a) \\
\nonumber & = \max_{a} q_k(s,a)
\end{align}
$$

Procedure summary:

* For any $s$ , current estimated value $v_k(s)$
* For any $a\in\mathcal{A}(s)$ , calculate $q_k(s,a) = \sum_r p(r|s,a)r + \gamma\sum_{s'}p(s'|s,a)v_k(s')$
* Calculate the greedy policy $\pi_{k+1}$ for $s$ as

  $$
  \pi_{k+1}(a|s)=\begin{cases}
  1 & a = a_k^*(s) \\
  0 & a \ne a_k^*(s)
  \end{cases}
  $$

  where $a_k^*(s)=\argmax_a q_k(s,a)$

* Calculate $v_{k+1}(s)=\max_a q_k(s,a)$

The above algorithm is actually the value iteration algorithm

## BOE: Optimality

Suppose $v^*$ is the solution to the Bellman optimality equation. It satisfies

$$
v^* = \max_\pi(r_\pi + \gamma P_\pi v^*)
$$

Suppose

$$
\pi^* = \argmax_\pi(r_\pi+\gamma P_\pi v^*)
$$

Then

$$
v^* = r_{\pi^*} + \gamma P_{\pi^*}v^*
$$

Therefore, $\pi^*$ is a policy and $v^*=v_{\pi^*}$ is the corresponding state value

## Analyzing optimal polices

What factors determine the optimal policy?

It can be clearly seen from the BOE

$$
v(s) = \max_\pi\sum_a \pi(a|s)\left(\sum_rp(r|s,a)r + \gamma\sum_{s'}p(s'|s,a)v(s') \right)
$$

that there are three factors:

* Reward design: $r$
* System model: $p(s'|s,a)$ , $p(r|s,a)$
* Dicount rate: $\gamma$
* $v(s),v(s'),\pi(a|s)$ are unknowns to be calculated

$\gamma$ is closer to 0, the optimal policy becomes more short-sighted.

If we change $r\to ar+b$ , the optimal policy remains the same.
