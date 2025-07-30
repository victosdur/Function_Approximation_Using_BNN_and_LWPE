import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# generate data using a function depending their domain and number of samples
def generate_training_data(f, x_range, num_samples):
    x = np.linspace(*x_range, num_samples)
    y = np.array([f(xi) for xi in x])
    return x, y

def interpolation_tf(x, y, x_points):
    # Get indexes of the left-most values
    indexs = tf.searchsorted(x, x_points) - 1 
    
    # Clamping to avoid out-of-range indexes
    indexs = tf.clip_by_value(indexs, 0, tf.shape(x)[0] - 2) 
    
    # Extract values of x and y at the extremes of the intervals found.
    x1 = tf.gather(x, indexs)
    x2 = tf.gather(x, indexs + 1)
    y1 = tf.gather(y, indexs)
    y2 = tf.gather(y, indexs + 1)
    
    # Calculate linear interpolation
    y_points = y1 + (y2 - y1) * (x_points - x1) / (x2 - x1)
    
    # Handle the exact case: if xs_torch matches exactly some value of x, we use its image
    mask_exact = tf.reduce_any(tf.equal(tf.expand_dims(x_points, axis=1), x), axis=1)
    exact_indices = tf.searchsorted(x, x_points)
    exact_indices = tf.clip_by_value(exact_indices, 0, tf.shape(y)[0] - 1)
    exact_values = tf.gather(y, exact_indices)
    
    y_points = tf.where(mask_exact, exact_values, y_points)
    
    return y_points

def plot_and_save_approximation(model, x_train, y_train, x_points, y_points, domain, title, filename, fontsize=14):
    plt.figure(figsize=(8, 6))
    plt.plot(x_train, model(tf.expand_dims(x_train, axis=1)), 'r-', alpha=0.5, label="BNN(x)")
    plt.plot(x_train, y_train, 'g-', alpha=0.5, label="$f(x)$")
    plt.scatter(x_points, y_points, color="red", label="BNN base points")
    plt.xlabel('x', fontsize=fontsize)
    plt.ylabel('y', fontsize=fontsize)
    plt.xlim((domain[0], domain[1]))
    plt.title(title, fontsize=fontsize+1)
    plt.legend(loc="lower left", fontsize=fontsize-2, framealpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()