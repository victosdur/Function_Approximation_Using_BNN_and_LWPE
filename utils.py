import numpy as np
import tensorflow as tf

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