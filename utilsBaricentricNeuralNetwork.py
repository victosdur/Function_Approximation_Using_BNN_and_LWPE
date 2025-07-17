import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, activations
from tensorflow .keras.layers import Layer


# the BaricentricNeuralNetwork: Tensorflow
class BaricentricSigmaLayerTf(Layer):
    def __init__(self, points, **kwargs):
        super(BaricentricSigmaLayerTf, self).__init__(**kwargs)
        
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
            #hiddenlayer1
            relu1h1 = activations.relu(t)  #
            step1 = tf.cast((-t) > 0, dtype=tf.float32)
            step2 = tf.cast((t - 1) > 0, dtype=tf.float32)
            relu2h1 = activations.relu(1-t) 

            relu1h2 = activations.relu(1-relu1h1-2*step1-2*step2)
            relu2h2 = activations.relu(1-relu2h1-2*step1-2*step2)
            
            # Output for this segment
            segment_output = relu1h2* b_i + relu2h2 * b_next
            # Add the segment contribution to the total output
            output += segment_output
        return output


class BaricentricSigmaNetworkTf(Model):
    def __init__(self, points, **kwargs):
        super(BaricentricSigmaNetworkTf, self).__init__(**kwargs)
        self.layer = BaricentricSigmaLayerTf(points)

    def call(self, x):
        return self.layer(x)