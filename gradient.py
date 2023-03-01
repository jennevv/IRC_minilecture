from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize
import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation
matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\\FFmpeg\\bin\\ffmpeg.exe'

def f(x):
    '''Objective function'''
    return 0.5*(x[0] - 4.5)**2 + 2.5*(x[1] - 2.3)**2

def df(x):
    '''Gradient of the objective function'''
    return np.array([x[0] - 4.5, 5*(x[1] - 2.3)])

def steepest_descent(gradient, x0 = np.zeros(2), alpha = 0.01, max_iter = 10000, tolerance = 1e-10):
    '''
    Steepest descent with constant step size alpha.

    Args:
      - gradient: gradient of the objective function
      - alpha: line search parameter (default: 0.01)
      - x0: initial guess for x_0 and x_1 (default values: zero) <numpy.ndarray>
      - max_iter: maximum number of iterations (default: 10000)
      - tolerance: minimum gradient magnitude at which the algorithm stops (default: 1e-10)

    Out:
      - results: <numpy.ndarray> of size (n_iter, 2) with x_0 and x_1 values at each iteration
    '''

    # Initialize the iteration counter
    iter_count = 1

    # Prepare list to store results at each iteration
    results = np.array([])

    # Evaluate the gradient at the starting point
    gradient_x = gradient(x0)
    print(gradient_x)
    # Gradient values in a list
    gradients = gradient_x

    # Set the initial point
    x = x0
    results = np.append(results, x, axis=0)

    # Iterate until the gradient is below the tolerance or maximum number of iterations is reached
    # Stopping criterion: inf norm of the gradient (max abs)
    while any(abs(gradient_x) > tolerance) and iter_count < max_iter:

        # Update the current point by moving in the direction of the negative gradient
        x = x - alpha * gradient_x

        # Store the result
        results = np.append(results, x, axis=0)

        # Evaluate the gradient at the new point
        gradient_x = gradient(x)

        # add gradient value to list
        gradients = np.append(gradients,gradient_x)

        # Increment the iteration counter
        iter_count += 1

    # Return the points obtained at each iteration
    return results.reshape(-1, 2),gradients.reshape(-1,2)

alpha = 0.4
estimate,gradients = steepest_descent(df, x0 = np.array([-9, -9]), alpha=alpha)

print('Final results: {}'.format(estimate[-1]))
print('N° iterations: {}'.format(len(estimate)))

X, Y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
Z = f(np.array([X, Y]))

# Minimizer
result = minimize(
    f, np.zeros(2), method='trust-constr', jac=df)

min_x0, min_x1 = np.meshgrid(result.x[0], result.x[1])
min_z = f(np.stack([min_x0, min_x1]))


# Plot

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
levels = [x**2 for x in range(22)]
ax.contour(X, Y, Z, levels)
ax.scatter(min_x0,min_x1,color = 'red')
ax.scatter(-9.1,-9.1,color = 'red')
ax.set_xlabel('$x_{0}$')
ax.set_ylabel('$x_{1}$')

def animation_function(i):
    if i == 0:
        ax.scatter(min_x0, min_x1, color='red')
        ax.scatter(-9.1, -9.1, color='red')
        ax.annotate("", (estimate[i, 0], estimate[i, 1]), ((estimate[i, 0] - alpha * gradients[0,0], estimate[i, 1])),
                    arrowprops={'arrowstyle': '<-', 'color': 'black', 'linewidth': 2})
        ax.annotate("", (estimate[i, 0], estimate[i, 1]), ((estimate[i, 0] , estimate[i, 1] - alpha * gradients[0, 1])),
                    arrowprops={'arrowstyle': '<-', 'color': 'black', 'linewidth': 2})
    else:
        ax.annotate("", (estimate[i-1,0], estimate[i-1,1]), ((estimate[i, 0], estimate[i, 1])),
                arrowprops={'arrowstyle': '<-', 'color': 'C3', 'linewidth': 2})

animation = FuncAnimation(fig,
                          func = animation_function,
                          frames = np.arange(0, len(estimate)-1, 1),
                          interval = 500)


f = 'steepestdescent_03.mp4'
animation.save(f)

