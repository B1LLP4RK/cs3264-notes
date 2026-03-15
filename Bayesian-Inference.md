# Bayesian Inference

${toc}

## benefits

- allow **prior knowledge** to be combined with _observed data_
- allows new input to be classified by comgining _prediction of multiple hypotheses_ weighted by beliefs
- can work incrementally

## Bayes' Theorem/Belief Update

- $P(h)$ prior belief of hypothesis $h$
- $P(D|h)$ likelihood of data D given $h$
- $P(D)= \sum \limits_{h \in H} P(D|h)P(h)$ marginal likelihood/evidence of D
- $P(h|D)$ posterior belief of $h$ given D

### MAP hypothesis

- Maxium a Posterior

$$
\begin{align*}
h_{\text{MAP}} &= \arg \max \limits_{h \in H} P(h|D)
  \\
&= \arg \max \limits_{h \in H} \frac{P(D|h) P(h)}{P(D)}
  \\
&= \arg \max \limits_{h \in H} P(D|h) P(h)
\end{align*}
$$

- Maximum Likelihood

$$
h_{\text{ML}} = \arg \max \limits_{h \in H} P(D|h)
$$

- note if $P(h) = P(h') \  \forall h,h' \in H$, $h_{\text{MAP}} = h_{\text{ML}}$

### Basic Probability formulas

- Chain rule

$$
P(A_1,...,A_n) = \prod_{i=1}^{n}P(A_i|A_1,...,A_{i-1})
$$

- Inclusion exclusion principle

$$
P(\bigcup_{i=1}^{n}A_i ) = \sum \limits_{1 \leq i \leq n}P(A_i) - \sum \limits_{1 \leq i \leq j \leq n} P(A_i, A_j) + ...
$$

- Marginalization
  - if events $A_1, ..., A_n$ are mutually exclusive and $\sum \limits_{i=1}^{n} P(A_i) = 1$

$$
P(B) = \sum \limits_{i=1}^{n}P(B|A_i)P(A_i)
$$

### Brute force MAP hypothesis Learner

1. For each hypothesis $h \in H$ compute $P(h|D)$
2. Output $h_\text{MAP}$ with highest $P(h|D)$

### Relation to Concept learning

- assumptions
  1. input instances $x_d$ are fixed
  2. $P(h) = \frac{1}{|H|}\ \forall h \in H$

- $P(D|h)$ is
  - 1 if $h$ is consistent with $D$
  - 0 if $h$ is not consistent with $D$

- Then $P(h|D)$ is
  - $\frac{1}{|VS_{H,D}|}$ if $h$ is consistent
  - 0 otherwise
- Thus all consistent hypothesis is a MAP hypothesis

### Learning a Continuous Valued Function

- $h_\text{ML} = \arg \min \limits_{h \in H}\frac{1}{2}\sum \limits_{d \in D}(t_d - h(x_d))^2$ for continous valued function
- where $D=\{<x_d, t_d>\}$ where $t_d$ is a noisy output
  - means $t_d = f(x_d) + \epsilon_d$ where $\epsilon_d \sim N(0, \sigma^2)$
  - derived based on conditional independence, constant removal, etc

## Neural Network for computing probability

- $h_\text{ML} =\arg \max \limits_{h \in H} \sum \limits_{d \in D}^{}t_d \ln h(x_d) + (1- t_d)\ln (1-h(x_d))$
- $c:X \to \{ 0, 1 \}$ and $D = \{ <x_d, t_d> \}$ where $t_d=c(x_d)$

$$
\begin{align*}
  P(D|h) &= \prod_{d \in D}P(t_d, x_d|h)\\
&= \prod_{d \in D}P(t_d| x_d,h)P(x_d| h)\\
&= \prod_{d \in D}P(t_d| x_d,h)P(x_d)\\
&= \prod_{d \in D}h(x_d)^{t_d}(1-h(x_d))^{1-td} P(x_d)\\
&= \sum_{d \in D}\ln h(x_d)^{t_d}+ \ln (1-h(x_d))^{1-td} + \ln P(x_d)\\
&= \sum_{d \in D}t_d\ln h(x_d)+ (1 - t_d)\ln (1-h(x_d)) + \ln P(x_d)\\
\end{align*}
$$

$$
\begin{align*}
\therefore \arg \max \limits_{h \in H} P(D|h) &= \arg \max \limits_{h \in H} \sum_{d \in D}t_d\ln h(x_d)+ (1 - t_d)\ln (1-h(x_d)) + \ln P(x_d)\\
&=\arg \max \limits_{h \in H} \sum_{d \in D}t_d\ln h(x_d)+ (1 - t_d)\ln (1-h(x_d))
\end{align*}
$$

### apply to Gradient ascent in Sigmoid unit

- $U_d = \sum_{d \in D}t_d\ln h(x_d)+ (1 - t_d)\ln (1-h(x_d))$

$$
\begin{align*}
\frac{\partial U_d}{\partial w_i} &=\sum_{d \in D} \frac{\partial U_d}{\partial h(x_d)}\frac{\partial h(x_d)}{\partial w_i}\\
&=\sum_{d \in D} \frac{\partial (t_d\ln h(x_d)+ (1 - t_d)\ln (1-h(x_d)))}{\partial h(x_d)}\frac{\partial h(x_d)}{\partial w_i}\\
\end{align*}
$$

## Minimum Description Length Principle

- definition

$$
h_\text{MDL} = \arg \min \limits_{h \in H} L_c(h) + L_c(D|h)
$$

- $L_c(h)$ is the encoding of $h$ using $c$

- prevents overfitting

- if $-\log_{2}h$ is the length required to describe $h$, then $h_\text{MAP} = h_\text{MDL}$

$$
\begin{align*}
h_\text{MAP} &= \arg \max \limits_{h \in H} P(h|D)\\
&= \arg \max \limits_{h \in H} \frac{P(D|h)P(h)}{P(D)}\\
&= \arg \max \limits_{h \in H} P(D|h)P(h)\\
&= \arg \max \limits_{h \in H}\log_{2} P(D|h) + \log_{2}  P(h)\\
&= \arg \min \limits_{h \in H}-\log_{2} P(D|h) + -\log_{2}  P(h)\\
&= h_\text{MDL} \end{align*}
$$

## Bayes-Optimal Classifier

- $\arg \max \limits_{t \in T} P(t|D)= \arg \max \limits_{t \in T} \sum \limits_{h \in H} P(h)P(t|h)$

## Gibbs Classifer

- $\sum \limits_{h \in H}$ is costly if $H$ is large, for Baeys Optimal classifer
- so sample $h$
- tbh i don't understand this

## Naive Bayes Classifer

- consider target function $c:X \to T$
- $x = (x_1,...x_n)^\intercal,\ \forall  x \in X$

$$
\begin{align*}
t_\text{MAP} &=\arg \max \limits_{t \in T}P(t|x_1,...,x_n)\\
&=\arg \max \limits_{t \in T} \frac{P(x_1,...,x_n|t)P(t)}{P(x_1,...,x_n)}\\
&=\arg \max \limits_{t \in T} P(x_1,...,x_n|t)P(t)\\
&=\arg \max \limits_{t \in T}\prod_{i =1}^{n} P(x_i|t)P(t)
\end{align*}
$$

- the last conversion dependent on conditional indespendence given classification

- Cons
  - Moderate or large training data required
  - Input attributes should be conditionally independent given classification

- Naive Bayes Algorithm
  - approximate probabilities with $D$
    - for each $t \in T$
      - get $\hat{P}(t)$ that is approximate of $P(t)$
      - for each $x_i \forall i \in \{ 1,...,n \}$
        - get $\hat{P}(x_i|t)$ that is approximate of $P(x_i|t)$
  - Classify new instances
    - $t_{NB} =\arg \max \limits_{t \in T} \prod_{i =1}^{n} \hat{P}(x_i|t)\hat{P}(t)$

- conditional independence often violated
  - but we don't care as long as we get same classification. In other words,

$$
\arg \max \limits_{t \in T} \prod_{i =1}^{n} \hat{P}(x_i|t)\hat{P}(t) = \arg \max \limits_{t \in T}P(x_1,...x_n|t)P(t)
$$

- there might not be nough data such that for some $i$, $\hat{P}(x_i|t) = 0$
  - meaning $\prod_{i =1}^{n} \hat{P}(x_i|t)\hat{P}(t)=0$
  - making the algorithm not consider that $t$ as classification if instance being classified has that value of $x_i$
  - solution: use bayesian estimate where $m$ is weight and $p$ is prior belief

$$
\hat{P}(x_i|t) = \frac{|D_{txi}|+mp}{|D_t|+m}
$$

## Expectation Maximization

- when to use?
  - Data is only partially observable
  - unsupervised clustering
  - Supervised learning
- applications
  - training bayesian belief network
  - Unsupervised clustering
- generate data
  - each instance $x_d$ is created by
  - select one of $M$ Gaussian distribution, with **uniform** probability
  - sampling an instance from that Gaussian

### EM for estimating M means

- given
  - there are $M$ Gaussian distribution
    - the distribution has same variance $\sigma^2$
    - the means are unknown
  - the data $x_d \in X$ are generated by selecting a Gaussian and selecting
- determine Maxmum likelihood means of $<\mu_1, ...,\mu _n>$
- consider full description as $<x_d, z_{d1}, z_{d2}>$
  - $z_{dm}$ is unobservable and is value 1 if $x_d$ is from $m$th Gaussian and otherwise 0
  - $x_d$ is observable

### EM algorithm

- Pick random initial $h = <\mu_1, \mu_2>$ then iterate
- Expectation step
  - Calculate $E[z_{dm}]$ of each $z_{dm}$ asuming $h$ is correct
    - it refers to the probability that data $x_d$ is from mth gaussian

$$
\begin{align*}
E[z_{dm}] &= P( \mu _m |x_d) \\
&= \frac{P(x_d|\mu _m)P(\mu _m)}{P(x_d)}\\
&= \frac{P(x_d|\mu _m)P(\mu _m)}{\sum \limits_{i=1}^{M}P(x_d|\mu _i )P(\mu _i)}\\
\end{align*}
$$

- since $P(\mu _i) = \frac{1}{M} = P(\mu _m)$

$$
\frac{P(x_d|\mu _m)P(\mu _m)}{\sum \limits_{i=1}^{M}P(x_d|\mu _i )P(\mu _i)} = \frac{P(x_d|\mu _m)}{\sum \limits_{i=1}^{M}P(x_d|\mu _i )}
$$

- Maximization step
  - assume value taken by each $z_{dm}$ is the $E[z_{dm}]$
  - replace $h$ by $h'= <\mu _1', ...,\mu _M'>$

$$
\mu _m' = \frac{\sum \limits_{d \in D}^{}x_dE[z_{dm}]}{\sum \limits_{d \in D}^{}E[z_{dm}]}
$$

- tbh i don't know why the steps are E and M steps respectively
- the algorithm converges to local ML hypothesis.
  - to $E[\ln p(D|h')]$

### general EM

- Given
  - Observed data $\{ x_d \}_{d \in D}$
  - Unobserved data $\{ z_d \}_{d \in D}$
    - where $z_d = <z_{d1}, ... z_{dM}>$
  - parameterized probability distribution $p(D|h)$
    - $D = \{ d \}$ where $d=<x_d, z_d>$
    - $h$ comprises the parameters
- determine ML hypothesis $h'$ that maximizes $E[\ln p(D|h')]$
