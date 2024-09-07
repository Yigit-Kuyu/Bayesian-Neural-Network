# INFO
 In machine learning, **frequentist methods** optimize objective functions using observed data, while **Bayesian methods** use prior knowledge to estimate posterior distributions and quantify uncertainty. The difference can be seen in the following figure. Basically, we play with distributions, not "approximate points".

![AV_Module](https://github.com/Yigit-Kuyu/Bayesian-Neural-Network/blob/main/BNN.jpg)

# Topics in Bayesian Methods

1) **Probability Distributions**

   - **[Gaussian (Normal) Distribution:](https://github.com/Yigit-Kuyu/Bayesian-Neural-Network/blob/main/GaussDist.py)** Commonly used to model environmental noise and uncertainty in value functions.
   
   - **[Beta Distribution:](https://github.com/Yigit-Kuyu/Bayesian-Neural-Network/blob/main/BetaDist.py)** A continuous probability distribution defined on the interval [0, 1], frequently employed in Bayesian statistics to represent probabilities, such as the likelihood of success.
   
   - **[Dirichlet Distribution:](https://github.com/Yigit-Kuyu/Bayesian-Neural-Network/blob/main/DirichletDist.py)** Serves as a prior distribution for categorical variables in DL.
   
   - **Bernoulli Distribution:** Ideal for modeling binary outcomes in DL tasks, such as the success or failure of an action.

2) **Bayesian Methods in Machine Learning**

   - **[Bayesian Inference Techniques in BNN:](https://github.com/Yigit-Kuyu/Bayesian-Neural-Network/blob/main/BNN.py)** A framework that applies Bayesian inference (given below) to neural network architectures, allowing for uncertainty quantification in predictions.
       - **Variational Inference (VI):** A technique that approximates the true posterior distribution with a simpler variational distribution (e.g., Gaussian), optimizing it by minimizing the Evidence Lower Bound (ELBO).
       - **MCMC (Markov Chain Monte Carlo):** A method that samples from the posterior distribution using a random walk to generate a Markov chain.
       - **Bayes by Backpropagation:** A stochastic gradient-based method that updates posterior distributions of network weights using backpropagation. 


