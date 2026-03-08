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
