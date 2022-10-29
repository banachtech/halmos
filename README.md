# Project Erdos

Aim of project Erdos is to create a reinforcement learning model for automated trading of cryptocurrencies. The model is inspired by [Mnih et al (2015)](https://www.nature.com/articles/nature14236). 

## Data
Dataset consists of 30 minute interval price and order-book data of 9 cryptocurrencies, namely, ADA, ATOM, BCH, BNB, BTC, DOGE, ETH, SOL and XRP. The raw data has open, high, low and close prices and ask10 and bid10 of order book information in respect of each cryptocurrency. We transform the raw data into features as below.

## Features
The features (i.e. the preprocessed state vector $\phi(s_t)$ referred to in the paper) *in respect of each currency* are

1. mean orderbook ratio: 
$$\delta_t = \frac{1}{N} \sum_{s=t-N+1}^t \log(ask10_s) - \log(bid10_s)$$ 
2. intra-period volatility: 
$$\nu_t = \frac{1}{N} \sum_{s=t-N+1}^t \log(high_s) - \log(low_s)$$
3. mean of returns: 
$$\mu_t = \frac{1}{N} \sum_{s=t-N+1}^t \log(close_s) - \log(close_{s-1})$$
4. std dev of returns: 
$$\sigma_t = \sqrt{\frac{1}{N-1} \sum_{s=t-N+1}^t (\log(close_s/close_{s-1}) - \mu_t)^2}$$
5. relative strength indicator:
$$\rho_t = \frac{up_t}{up_t + dn_t}$$
$$up_0 = 0, up_s = \frac{N-1}{N} up_{s-1} + \frac{1}{N} \mathbb{1}(close_s > close_{s-1})$$
$$dn_0 = 0, dn_s = \frac{N-1}{N} dn_{s-1} + \frac{1}{N} \mathbb{1}(close_s < close_{s-1})$$
where $\mathbb{1}(A)$ is an indicator function taking value $1$ if $A$ is true and value $0$ otherwise.
6. bollinger band indicator:
$$\beta_t = \text{tanh}(\frac{close_t - a_t}{b_t})$$
where
$$a_t = \frac{1}{N} \sum_{s=t-N+1}^t p_s$$
$$b_t = \sqrt{\frac{1}{N-1} \sum_{s=t-N+1}^t (p_s - a_t)^2}$$
and
$$p_s = \frac{close_s + high_s + low_s}{3}$$
7. true range indicator:
$$\kappa_t = \frac{\max(high_t, close_{t-1}) - \min(low_t, close_{t-1})}{close_t}$$

$N$ is the observation or lookback period and is a hyper parameter that needs to be determined through experimentation. $N=48$ is a good starting value. The tuple $x_t = (\delta_t, \nu_t, \mu_t, \sigma_t, \rho_t, \beta_t, \kappa_t)$ makes the feature vector in respect of each cryptocurrency. The final feature vector is a concatenation of all feature vectors (so it is of dimension $9\times 7 = 63$).

## Action
The action space in this set-up is a 9 element vector $a$, each component representing investment in a cryptocurrency. $a$ is constrained such that $a_i \in [-1,1]$ and $\Vert a \Vert_1 \leq 1$.

## Reward
Reward for taking action $a_t$ at time $t$ is defined as

$$r_t = \sum_{j=1}^9 a_{i,t} (\frac{close_{t+1}}{close_t} - 1) - c \sum_{j=1}^9 \vert a_{i,t} - a_{i,t-1} \vert $$

where $c$ is the transaction cost. $c$ is an input variable to the problem. For this exercise, we take $c=0.0002$.

## Q-Function
The Q-function $Q:\mathbb{R^{72}} \mapsto \mathbb{R}$ is modelled in two ways. Note that 72 comes from concatenating the aggregate feature vector and the action vector ($=9\times7 + 9$).
### Neural Network
This is similar to the model used by the paper. We define a fully connected neural network with 3 hidden layers with 64, 32 and 16 neurons respectively. We use relu as activation function for all layers except the output layer. The input layer obviously has 72 neurons taking in the concatenated vector of aggregate feature vector and the action vector. The output layer has one neuron and no activation function.
### Gaussian RBF
We use a Gaussian radial basis network with $M$ units, where $M$ needs to be determined through experimentation. $M=10$ is a good starting point. The functional form of the network is as follows:
$$G(z) = b_1^Tz + b_0 + \sum_{j=1}^M a_j e^{\frac{-\Vert z - \xi_j\Vert_2^2}{2h^2}}$$
where $z$ is the combined feature and action vector i.e. $z \in \mathbb{R}^{72}$ and $a_j,b_0 \in \mathbb{R}, h \in \mathbb{R}^+, \xi_j \in \mathbb{R}^{72}$. Note the positivity constraint on $h$.

## Training
As mentioned in the paper, we maintain two sets of parameters $\theta^-$ and $\theta$ (following the notation of the paper) of Q-function. These are upated as per the algorithm of the paper. We maintain a experience memory or buffer (from which samples are drawn for gradient calculation and updating of the Q-function parameters) of length 960 and batch or sample size of 24. 
However, we depart from the paper in a few ways:
1. We just have a single episode i.e the entire training dataset.
2. There is no terminal state.
3. We use an exponential averaging scheme to update parameter $\theta^-$ after every $C=48$ updates of $\theta$. We set $\theta^- \leftarrow (1-\tau)\theta^- + \tau \theta$, where the scale parameter is annealed as $\tau_n = \frac{1}{\log(n)}$. $n is the iteration number of the update.

## Implementation
Preferred language for implementation is [Julia](https://docs.julialang.org/en/v1/). [Flux](https://fluxml.ai/Flux.jl/stable/) is the most popular light weight machine learning package for Julia. [Zygote](https://fluxml.ai/Zygote.jl/latest/) is a popular package for gradient calculations through automatic differentiation. [Optim](https://julianlsolvers.github.io/Optim.jl/stable/) is a powerful convex optimisation library.

## Deliverables
1. A Julia program (i.e. a .jl script file) that 
  - reads in raw data in matrix form of open, close, high, low, ask10 and bid10 stored in csv file
  - reads in model parameters stored in json file
  - computes action vector and appends the action vector with timestamp into actions.csv file with cryptocurrency name as headers
  - updates the model parameters
2. A Jupyter notebook with the testing and development work with meaningful comments accompanying the code.

## Timeframe
Expected timeframe for completion of the project is 3 weeks, assuming 3 days of work per week.

## References
Reinforcement learning:

1. Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.
2. [Berkeley RL Bootcamp] (https://sites.google.com/view/deep-rl-bootcamp/lectures)

Deep learning:

3. Strang, Gilbert (2019). Linear Algebra and Learning from Data. Wellesley-Cambridge Press.
4. Geron, Aurelien (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition. O'Reilly Media, Inc.
