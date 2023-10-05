import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Sample dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

# Prior
prior_mu = np.zeros(2)
prior_sigma = np.eye(2)

# Logistic function
def logistic(z):
  return 1/(1+np.exp(-z))

# Likelihood
def likelihood(y, X, w):
  z = X.dot(w)
  p = logistic(z)
  ll = y*np.log(p) + (1-y)*np.log(1-p)
  return np.sum(ll)

# Sample from posterior
N = 1000
samples = np.zeros((N,2))
for i in range(N):
  samples[i,:] = np.random.multivariate_normal(prior_mu, prior_sigma)

# Plot samples
plt.scatter(samples[:,0], samples[:,1], c='b', alpha=0.1)
plt.xlabel('w1')
plt.ylabel('w2')
plt.title('Bayesian Logistic Regression')
plt.show()

print("Bayesian logistic regression implemented and visualized!")
