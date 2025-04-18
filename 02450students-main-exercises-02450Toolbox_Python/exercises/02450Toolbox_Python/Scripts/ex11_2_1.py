# exercise 11_2_1
import numpy as np
import matplotlib.pyplot as plt

# Number of data objects
N = 1000

# Number of attributes
M = 1

# x-values to evaluate the histogram
x = np.linspace(-10, 10, 50)

# Allocate variable for data
X = np.empty((N, M))

# Mean and covariances
m = np.array([1, 3, 6])
s = np.array([1, 0.5, 2])

# Draw samples from mixture of gaussians
c_sizes = np.random.multinomial(N, [1.0 / 3, 1.0 / 3, 1.0 / 3])
for c_id, c_size in enumerate(c_sizes):
    X[
        c_sizes.cumsum()[c_id] - c_sizes[c_id] : c_sizes.cumsum()[c_id], :
    ] = np.random.normal(m[c_id], np.sqrt(s[c_id]), (c_size, M))


# Plot histogram of sampled data
plt.figure()
plt.hist(X, x)
plt.show()

print("Ran Exercise 11.2.1")
