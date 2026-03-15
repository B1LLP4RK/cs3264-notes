# Neural Networks

## Perceptron Unit

- $o(x_1, x_2,...,x_n)$ =
  - 1 if $w_0 + w_1x_1 + ... + w_nx_n >  0$
  - -1 if $w_0 + w_1x_1 + ... + w_nx_n \leq  0$
- decision surface
  - the line that divdes the positive data from negative
  - only exist for linearly separable data
- training
  - set a random initial weight vector $w$
  - for each datapoint d
    - $w_i = w_i+\eta (t-o)x_i$

## Gradient descent

- for a simpler linear unit (not perceptron)
  - $o = w \cdot x$
- let loss function sum of squared error
  - $L(w) = \frac{1}{2}\sum \limits_{d \in D}^{} (t_d-o_d)^2$
- then gradient
  - $\nabla L(w) = [\frac{\partial L(w)}{\partial w_i}, ...]$
  - $\frac{\partial L(w)}{\partial w_i} =-\sum \limits_{d \in  D}^{} (t_d-o_d)x_i$
- start with a random $w$
  - then for each $w_i$ do
  - $w_i = w_i - \eta \frac{\partial L(w)}{\partial w_i} = w_i + \eta \sum \limits_{d \in  D}^{} (t_d-o_d)x_i$

## Comparison

| Perceptron                       | Gradient ascent   |
| -------------------------------- | ----------------- |
| required data linearly separable | data can be noisy |

## Stochastic Gradient descent

- similar with GD. choose random initial $w$
- for each data $d$ chosen randomly
  - $L_d(w) = \frac{1}{2}(t_d-o_d)^2$
  - $w = w - \eta \nabla L_d(w)$

## Sigmoid Unit

- $o(x_1, x_2, ..., x_n) = \sigma(w \cdot X)$
  - $X = (x_0, x_1 , ...,x_n)$ where $x_0=1$
  - sigmoid function $\sigma (x) =\frac{1}{1+e^{-x}}$
    - has propoerty

  $$
  \frac{\Delta \sigma (x)}{\Delta x} = \sigma (x) \times (1- \sigma (x))
  $$

- Loss Gradient

$$
\frac{\partial L_d}{\partial w_i}= -\sum \limits_{d \in D}^{} (t_d-o_d)o_d(1-o_d)x_{id}
$$
