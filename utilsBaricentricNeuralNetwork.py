import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, activations
from tensorflow .keras.layers import Layer


# the BaricentricNeuralNetwork: Tensorflow
class BaricentricLayerTf(Layer):
    def __init__(self, points, **kwargs):
        super(BaricentricLayerTf, self).__init__(**kwargs)
        
        # Separate the input(x-coordinates and output values(y-values).
        # self.x_coords = tf.constant([p[0] for p in points], dtype=tf.float32)
        # self.y_values = tf.constant([p[1] for p in points], dtype=tf.float32)
        self.x_coords = points[:,0]
        self.y_values = points[:,1]

    def call(self, x):
        output = tf.zeros_like(x)  # Initialize output
        num_segments = len(self.x_coords) - 1  # Number of segments created by the points
        
        for i in range(num_segments):
            # Extract x_i, x_{i+1}, y_i, y_{i+1}
            x_i, x_next = self.x_coords[i], self.x_coords[i + 1]
            b_i, b_next = self.y_values[i], self.y_values[i + 1]
            
            # Barycentric coordinates t = (x - x_i) / (x_i+1 - x_i)
            t = (x_i - x) / (x_i - x_next)
            
            # Define contributions by segment
            relu1up = activations.relu(t)  #
            relu2up = activations.relu(1-relu1up)
            stepup = tf.cast((-t) >= 0, dtype=tf.float32)
            if i == 0:
                stepup = tf.cast((-t) > 0, dtype=tf.float32)
            else:
                stepup = tf.cast((-t) >= 0, dtype=tf.float32)
            
            relu1ab = activations.relu(1-t)      
            relu2ab = activations.relu(1-relu1ab)
            stepab = tf.cast((t-1) >= 0, dtype=tf.float32)
            if i == num_segments - 1:
                stepab = tf.cast((t-1) > 0, dtype = tf.float32)
            else:
                stepab = tf.cast((t-1) >= 0, dtype=tf.float32)
            
            # Output for this segment
            segment_output = (relu2up - stepup) * b_i + (relu2ab - stepab) * b_next
            
            # Add the segment contribution to the total output
            output += segment_output

        return output


class BaricentricNetworkTf(Model):
    def __init__(self, points, **kwargs):
        super(BaricentricNetworkTf, self).__init__(**kwargs)
        self.layer = BaricentricLayerTf(points)

    def call(self, x):
        return self.layer(x)