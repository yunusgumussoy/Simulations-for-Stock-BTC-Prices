# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 18:39:39 2024

@author: Yunus
"""
# Markov Chain Monte Carlo (MCMC) methods for predicting and forecasting Bitcoin prices

# pip install tensorflow tensorflow-probability

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import yfinance as yf
import matplotlib.pyplot as plt

# Step 1: Download Bitcoin data
btc_data = yf.download("BTC-USD", start="2015-01-01", end="2024-01-01")
returns = btc_data['Adj Close'].pct_change().dropna()

# Step 2: Define the model using TFP
def model_fn():
    # Prior distributions
    mu = tfp.distributions.Normal(loc=0.0, scale=1.0)
    sigma = tfp.distributions.HalfNormal(scale=1.0)
    return mu, sigma

# Step 3: Define the target log probability function
def target_log_prob_fn(mu, sigma):
    return tf.reduce_sum(tfp.distributions.Normal(loc=mu, scale=sigma).log_prob(returns))

# Step 4: Set up the MCMC sampler
num_samples = 2000
num_burnin_steps = 1000

@tf.function
def run_mcmc():
    # Initialize the Markov chain
    mu_current = tf.zeros(())
    sigma_current = tf.ones(())
    
    # Define the transition kernel
    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn,
        step_size=0.1,
        num_leapfrog_steps=3)

    # Run the MCMC
    samples, _ = tfp.mcmc.sample_chain(
        num_results=num_samples,
        num_burnin_steps=num_burnin_steps,
        current_state=[mu_current, sigma_current],
        kernel=kernel)

    return samples

# Step 5: Run the MCMC
samples = run_mcmc()

# Extract mu and sigma samples
mu_samples = samples[0]
sigma_samples = samples[1]

# Step 6: Forecasting future returns
future_steps = 30
future_returns = np.random.normal(np.mean(mu_samples), np.mean(sigma_samples), future_steps)

# Step 7: Calculate future prices
last_price = btc_data['Adj Close'][-1]
future_prices = [last_price * (1 + r) for r in future_returns]

# Step 8: Create future dates
last_date = btc_data.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps)

# Step 9: Plotting
plt.figure(figsize=(12, 6))
plt.plot(btc_data['Adj Close'], label='Historical Prices')
plt.plot(future_dates, future_prices, label='Forecasted Prices', color='red')
plt.title('Bitcoin Price Forecast using MCMC (TFP)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Display future prices
print("Forecasted Future Prices:")
for date, price in zip(future_dates, future_prices):
    print(f"{date.date()}: {price:.2f}")
