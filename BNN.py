import torch
import numpy as np
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.nn as nn
from pyro.infer import MCMC, NUTS
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from pyro.infer.autoguide import AutoNormal
from pyro.infer import Predictive
from pyro.optim import Adam



np.random.seed(42)



# Noisy observations from a sinusoidal function.
def generate_sinusoidal_data():
    """Generate noisy observations from a sinusoidal function."""
    x_obs = np.hstack([np.linspace(-0.2, 0.2, 500), np.linspace(0.6, 1, 500)])
    noise = 0.02 * np.random.randn(x_obs.shape[0])
    y_obs = x_obs + 0.3 * np.sin(2 * np.pi * (x_obs + noise)) + 0.3 * np.sin(4 * np.pi * (x_obs + noise)) + noise

    x_true = np.linspace(-0.5, 1.5, 1000)
    y_true = x_true + 0.3 * np.sin(2 * np.pi * x_true) + 0.3 * np.sin(4 * np.pi * x_true)
    
    return x_obs, y_obs, x_true, y_true


x_obs, y_obs, x_true, y_true = generate_sinusoidal_data()
# Set plot limits and labels
xlims = [-0.5, 1.5]
ylims = [-1.5, 2.5]

# Create plot
# Create a figure and axis for plotting
fig, ax = plt.subplots(figsize=(10, 5))

# Plot the true function as a blue line
ax.plot(x_true, y_true, 'b-', linewidth=3, label="True function")

# Plot the observations as black points
ax.plot(x_obs, y_obs, 'ko', markersize=4, label="Observations")

# Set the x and y limits for the plot
ax.set_xlim(xlims)
ax.set_ylim(ylims)

# Label the axes
ax.set_xlabel("X", fontsize=30)
ax.set_ylabel("Y", fontsize=30)

# Add a legend to the plot
ax.legend(loc=4, fontsize=15, frameon=False)

# The graph shows the true underlying function (in blue) and the noisy observations (in black).
# The observations are concentrated in certain areas of the true function, indicating that they do not cover the entire range of the function.
# This suggests that the model may have limited information about the true function, as the observations are not evenly distributed across it.
# The presence of noise in the observations can also affect the model's ability to accurately learn the true function.

plt.show()





class BNN(PyroModule):
    # This class defines BNN using Pyro, which allows for probabilistic programming.
    # We show weights and biases explicitly to define their distributions, which is essential for Bayesian inference.
    # Normally, we can build a network without showing them, but in a Bayesian context, we need to specify their prior distributions.
    
    def __init__(self, in_dim=1, out_dim=1, hid_dim=5, prior_scale=10.):
        super().__init__()

        self.activation = nn.Tanh()  # Activation function introduces non-linearity; alternatives include ReLU, Sigmoid, etc.
        # We can choose not to use an activation function, but it may limit the network's ability to learn complex patterns.
        
        # Direct sampling from Gaussian distributions without reparameterization (No lambda, no reparameterization)
        self.layer1 = PyroModule[nn.Linear](in_dim, hid_dim)  # Input to hidden layer
        self.layer2 = PyroModule[nn.Linear](hid_dim, out_dim)  # Hidden to output layer

        # Set layer parameters as random variables (Prior distributions)
        self.layer1.weight = PyroSample(dist.Normal(0., prior_scale).expand([hid_dim, in_dim]).to_event(2))
        self.layer1.bias = PyroSample(dist.Normal(0., prior_scale).expand([hid_dim]).to_event(1))
        self.layer2.weight = PyroSample(dist.Normal(0., prior_scale).expand([out_dim, hid_dim]).to_event(2))
        self.layer2.bias = PyroSample(dist.Normal(0., prior_scale).expand([out_dim]).to_event(1))
        # Normal distribution is used for weights and biases as it is a common choice for prior distributions; alternatives include Uniform or Laplace distributions.

    def forward(self, x, y=None):
        x = x.reshape(-1, 1)
        x = self.activation(self.layer1(x))
        mu = self.layer2(x).squeeze()  # mu represents the predicted output of the network
        # We sample "sigma" using a Gamma distribution because "sigma" quantifies the uncertainty or variability in the observations around the predicted mean(mu).
        # Sampling "sigma" from a distribution (e.g., Gamma distribution), we allow the model to learn the uncertainty in the noise level from the data, while "mu" is the predicted output of the network. 
        # In a Bayesian framework, we treat "sigma" as a random variable rather than a fixed parameter, which is essential for Bayesian inference.
        
        # In the below line, the Gamma distribution is often used as a prior for variance (or precision, which is the inverse of variance) 
        # because it is a conjugate prior for the normal distribution. This means that the posterior distribution will also be a Gamma distribution, simplifying the Bayesian updating process.
        sigma = pyro.sample("sigma", dist.Gamma(.5, 1))  # "sigma" (standart deviation): The name of the random variable being sampled.
                                                        # dist.Gamma(.5, 1): The Gamma distribution with shape parameter 0.5 and scale parameter 1. 
        # Alternatives for sigma could include using a Normal distribution, but Gamma is preferred for modeling positive values.

        
        with pyro.plate("data", x.shape[0]):
            # "obs1" is the observed data, which is assumed to follow a normal distribution centered around the predicted mean "mu".
            # "sigma * sigma" represents the variance of the normal distribution, where "sigma"(standart deviation) is a random variable sampled from a Gamma distribution.
            # "obs=y" indicates that the actual observed values are provided, allowing the model to condition on these observations during inference.   
            obs1 = pyro.sample("obs", dist.Normal(mu, sigma * sigma), obs=y)  # the likelihood of the observed data
        return mu
    

# This function runs a Markov Chain Monte Carlo (MCMC) simulation to sample from the posterior distribution of the Bayesian Neural Network (BNN) model.
# It generate predictions based on these samples.
def run_mcmc(x_obs, y_obs, num_samples=50, seed=42):  # Define the function with input data, number of samples, and seed
    print("Running MCMC with the following parameters:")
    print(f"Number of samples: {num_samples}")
    '''
    50 samples from the posterior distribution of all model parameters.
    Parameters Sampled:
    Weights for layer 1 and layer 2
    Biases for layer 1 and layer 2
    Sigma (the noise parameter)
    '''
    print(f"Random seed: {seed}")

    model = BNN()  # Instantiate the Bayesian Neural Network model
    print("Bayesian Neural Network model instantiated.")

    # Set Pyro random seed for reproducibility
    pyro.set_rng_seed(seed)  # Ensures that the random numbers generated are the same across runs
    print("Pyro random seed set for reproducibility.")

    # Define Hamiltonian Monte Carlo (HMC) Kernel for the MCMC process:
    # NUTS is a sophisticated MCMC method that automatically tunes the step size and avoids the need for manual tuning.
    # It is particularly effective for sampling from complex posterior distributions, such as those found in Bayesian Neural Networks (BNNs).
    # In the context of our BNN, the NUTS kernel will help us sample from the posterior distributions of the model parameters,
    # including the weights and biases of the neural network, which are initialized as normal distributions.
    # The kernel will also sample the uncertainty parameter "sigma", which quantifies the noise in the observations.
    # By using NUTS, we can efficiently explore the parameter space and obtain samples that reflect the uncertainty in our model.
    nuts_kernel = NUTS(model, jit_compile=False)  # NUTS (No-U-Turn Sampler) is a specific MCMC method
    
    # Print the model parameters and their uncertainty
    print("Model parameters and their uncertainty:")
    for name, value in model.named_parameters():
        uncertainty = value.stddev.data.numpy() if hasattr(value, 'stddev') else 'N/A'
        print(f"{name}: {value.data.numpy()}, Uncertainty: {uncertainty}")
    print("NUTS kernel defined for MCMC.")

    # Define MCMC sampler
    mcmc = MCMC(nuts_kernel, num_samples=num_samples)  # Create an MCMC object with the defined kernel and number of samples
    print("MCMC sampler created.")

    # Convert data to PyTorch tensors
    x_train = torch.from_numpy(x_obs).float()  # Convert input observations from NumPy array to PyTorch tensor
    y_train = torch.from_numpy(y_obs).float()  # Convert output observations from NumPy array to PyTorch tensor
    print("Input and output observations converted to PyTorch tensors.")

    # Run MCMC
    first_run=0
    if first_run==1:
        mcmc.run(x_train, y_train)  # Execute the MCMC sampling process with the training data
        # Get the samples
        samples_mcmc = mcmc.get_samples()
        print("MCMC samples obtained.")
        if samples_mcmc is not None:  # Check if samples are not None
            torch.save(samples_mcmc, 'mcmc_samples.pth')  # Save the MCMC samples to a file for later use
        else:
            print("Error: MCMC samples are None, cannot save to file.")
            return None, None
    else:
        try:
            samples_mcmc = torch.load('mcmc_samples.pth')  # Load the saved MCMC samples from file
            print("MCMC sampling process executed.")
        except FileNotFoundError:
            print("Error: 'mcmc_samples.pth' not found. Please run MCMC to generate samples.")
            return None, None

    # Plot the samples
    plt.figure(figsize=(10, 5))
    plt.plot(samples_mcmc['sigma'].detach().numpy(), label='Sigma')
    plt.xlabel('Sample Index')
    plt.ylabel('Sigma')
    plt.title('Posterior Samples of Sigma')
    # The graph displays the posterior samples of the uncertainty parameter "sigma" across different samples. 
    # A higher value of sigma indicates greater uncertainty in the model's predictions, while lower values suggest more confidence. 
    # Observing the distribution of sigma can help assess how well the model captures the noise in the observations.
    plt.legend()
    plt.show()

    return samples_mcmc, model




# Custom guide for Bayes by Backprop using reparameterization trick
def guide_bbb(model):
    for name, param in model.named_parameters():
        if 'weight' in name or 'bias' in name:
            loc = pyro.param(f"{name}_loc", torch.randn(param.shape))
            scale = pyro.param(f"{name}_scale", torch.ones(param.shape), constraint=dist.constraints.positive)
            pyro.sample(name, dist.Normal(loc, scale).to_event(param.dim()))

def run_bayes_by_backprop(x_obs, y_obs, num_iterations=1000, num_samples=50, seed=42):
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    model = BNN()  # Instantiate BNN 
    
    # Guide is no longer needed for Bayesian Backpropagation
    # Optimizer for backpropagation (now we directly update weights via Bayes' theorem)
    optimizer = Adam({"lr": 0.01})

    x_train = torch.from_numpy(x_obs).float()
    y_train = torch.from_numpy(y_obs).float()

    # Use the custom guide for Bayes by Backprop
    #guide = guide_bbb
    guide = lambda *args: guide_bbb(model)

    # Define SVI with Trace_ELBO
    svi_bbp = SVI(model, guide, optimizer, loss=Trace_ELBO())

    # Training loop
    for step in range(num_iterations):
        loss = svi_bbp.step(x_train, y_train)  # Include num_samples to avoid TypeError
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss}")

    # Generate samples from the posterior
    predictive = pyro.infer.Predictive(model, num_samples=num_samples)
    samples_bbp = predictive(x_train)  # Get predictions without 'obs'
    samples_bbp = {key: value for key, value in samples_bbp.items() if key != 'obs'}  # Remove 'obs' from the samples
    print("Variational Inference samples obtained.")
    print("Bayes by Backpropagation samples obtained.")
    
    return samples_bbp, model



# The aim of this code is to perform variational inference on BNN model.
# It aims to approximate the posterior distribution of the model parameters using a variational approach,
# similar to how the run_mcmc function samples from the posterior distribution using Markov Chain Monte Carlo (MCMC).
# This function will optimize the variational parameters to minimize the difference between the true posterior
# and the variational distribution, allowing for efficient inference in the BNN.
def run_vi(x_obs, y_obs, num_samples=50, seed=42):  # Define the function for variational inference
    print("Running Variational Inference with the following parameters:")
    print(f"Number of samples: {num_samples}") # 50 samples from the posterior distribution of all model parameters.
    '''
    Parameters Sampled:
    Weights for layer 1 and layer 2
    Biases for layer 1 and layer 2
    Sigma (the noise parameter)
    '''
    print(f"Random seed: {seed}")

    model = BNN()  # Instantiate the Bayesian Neural Network model
    print("Bayesian Neural Network model instantiated.")

    # Set Pyro random seed for reproducibility
    pyro.set_rng_seed(seed)  # Ensures that the random numbers generated are the same across runs
    print("Pyro random seed set for reproducibility.")

    # Define the variational guide
    guide = AutoNormal(model)  # Using AutoNormal for automatic variational inference
    # AutoNormal is a variational inference method that automatically determines the variational distribution.
    # Alternatives include using custom guides or other methods like Mean-Field Variational Inference.
    print("Variational guide defined.")

    # Define the optimizer
    optimizer = ClippedAdam({"lr": 0.01})  # Using ClippedAdam optimizer
    # ClippedAdam is used to prevent large updates that can destabilize training.
    # Alternatives include the standard Adam optimizer, but it may not handle large gradients as effectively.
    print("Optimizer defined.")

    # Set up the inference, objective function is the evidence lower bound (ELBO)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())  # Stochastic Variational Inference
    print("SVI setup complete.")

    # Convert data to PyTorch tensors
    x_train = torch.from_numpy(x_obs).float()  # Convert input observations from NumPy array to PyTorch tensor
    y_train = torch.from_numpy(y_obs).float()  # Convert output observations from NumPy array to PyTorch tensor
    print("Input and output observations converted to PyTorch tensors.")

    # Run variational inference 
    for step in range(num_samples):
        loss = svi.step(x_train, y_train)  # Execute the variational inference step
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss}")  # Print loss every 10 steps
    print("Variational Inference process executed.")
    
    
    # Generate samples from the posterior
    predictive = pyro.infer.Predictive(model, num_samples=num_samples)
    samples_vi = predictive(x_train)  # Get predictions without 'obs'
    samples_vi = {key: value for key, value in samples_vi.items() if key != 'obs'}  # Remove 'obs' from the samples
    print("Variational Inference samples obtained.")

    '''
    # Generate samples from the posterior (parameters only)
    param_store = pyro.get_param_store()
    samples_vi = {name: param.detach().clone() for name, param in param_store.items()}
    print("Variational Inference samples obtained.")
    '''

    return samples_vi, model


def predict_with_uncertainty(model, samples, x_test):
    predictive = Predictive(model=model, posterior_samples=samples)
    predictions = predictive(x_test)
    
    # Extract the relevant predictions (assuming the output is named 'obs')
    y_pred = predictions['obs']
    
    # Calculate mean and standard deviation
    mean_prediction = y_pred.mean(dim=0)
    std_prediction = y_pred.std(dim=0)
    
    return mean_prediction, std_prediction



# Run MCMC
samples1, model = run_mcmc(x_obs, y_obs)
print(f"Shape of samples1: { {key: value.shape for key, value in samples1.items()} }")  

# Prepare test data
x_test = torch.from_numpy(x_true).float().unsqueeze(1)

# Make predictions
mean_pred, std_pred = predict_with_uncertainty(model, samples1, x_test)

# Plot results
plt.figure(figsize=(12, 6))

# Plot the true function
plt.plot(x_true, y_true, 'g-', linewidth=2, label='True function')
# The true function represents the actual underlying relationship 
# we are trying to model. It serves as a benchmark for evaluating the performance of our Bayesian Neural Network (BNN). 

# Plot the observations
plt.plot(x_obs, y_obs, 'ko', markersize=4, label='Observations')
# Observations are the noisy data points collected from the true function. 
# They are crucial for training the model, but their noise can lead to 
# inaccuracies in predictions. 
# The distribution of these points can indicate areas where the model 
# may struggle to learn effectively.

# Plot the mean prediction
plt.plot(x_test, mean_pred, 'b-', linewidth=2, label='Mean prediction')
# The mean prediction is the BNN's estimate of the output given the input. 
# It represents the model's best guess of the true function based on the learned parameters. 
# However, it does not capture the uncertainty inherent in the predictions.

# Fill between the credible interval
plt.fill_between(x_test.squeeze(), mean_pred - 2*std_pred, mean_pred + 2*std_pred, 
                 alpha=0.3, color='blue', label='95% Credible Interval')
# The 95% credible interval provides a range within 
# which we expect the true output to lie with 95% probability. 
# This interval accounts for the uncertainty in the model's predictions,
# highlighting areas where the model is more or less confident.

plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('BNN Predictions with Uncertainty (HMC)')
plt.show()


# Run Variational Inference
samples2, model = run_vi(x_obs, y_obs, num_samples=50, seed=42)
samples2_reshaped = {key: value.squeeze(1) for key, value in samples2.items()}
print(f"Shape of samples2: { {key: value.shape for key, value in samples2_reshaped.items()} }")  



mean_pred, std_pred = predict_with_uncertainty(model, samples2_reshaped, x_test)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(x_true, y_true, 'g-', linewidth=2, label='True function')
plt.plot(x_obs, y_obs, 'ko', markersize=4, label='Observations')
plt.plot(x_test, mean_pred, 'b-', linewidth=2, label='Mean prediction')
plt.fill_between(x_test.squeeze(), mean_pred - 2*std_pred, mean_pred + 2*std_pred, 
                 alpha=0.3, color='blue', label='95% Credible Interval')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('BNN Predictions with Uncertainty (Variational Inference)')
plt.show()

samples3, model = run_bayes_by_backprop(x_obs, y_obs, num_samples=50, seed=42)
samples3_reshaped = {key: value.squeeze(1) for key, value in samples3.items()}
print(f"Shape of samples3: { {key: value.shape for key, value in samples3_reshaped.items()} }")  


mean_pred, std_pred = predict_with_uncertainty(model, samples3_reshaped, x_test)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(x_true, y_true, 'g-', linewidth=2, label='True function')
plt.plot(x_obs, y_obs, 'ko', markersize=4, label='Observations')
plt.plot(x_test, mean_pred, 'b-', linewidth=2, label='Mean prediction')
plt.fill_between(x_test.squeeze(), mean_pred - 2*std_pred, mean_pred + 2*std_pred, 
                 alpha=0.3, color='blue', label='95% Credible Interval')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('BNN Predictions with Uncertainty (Bayes by Backpropagation)')
plt.show()
