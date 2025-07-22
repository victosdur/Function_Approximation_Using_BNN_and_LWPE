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
    



def stepestrella(x):
    return tf.cast(x > 0, dtype=tf.float32)

class CustomDense(tf.keras.layers.Layer):
    def __init__(self, weights, biases, activations):
        super(CustomDense, self).__init__()
        self.weights = tf.constant(weights, dtype=tf.float32)
        self.biases = tf.constant(biases, dtype=tf.float32)
        self.activations = activations  # lista de funciones

    def call(self, inputs):
        z = tf.matmul(inputs, self.weights)
        z = tf.add(z, self.biases)
        if len(self.activations) == 1:
            return self.activations(z)
        else:
            out = tf.stack([self.activations[i](z[:, i]) for i in range(z.shape[1])], axis=1)
        return out
    
class BaricentricNetworkSegment(Model):
    def __init__(self, a, b, fa, fb, **kwargs):
        super(BaricentricNetworkSegment, self).__init__(**kwargs)
        # Layer 1: 4 neurons
        w1 = [[1, -1, 1, -1]]  # input_dim=1, output_dim=4
        b1 = [0, 0, -1, 1]
        act1 = [tf.nn.relu, stepestrella, stepestrella, tf.nn.relu]

        self.layer1 = CustomDense(tf.transpose(w1), b1, act1)

        # Layer 2: 2 neurons
        w2 = [[-1, 0], [-2, -2], [-2, -2], [0, -1]]  # 4x2
        b2 = [1.0, 1.1]
        act2 = tf.nn.relu

        self.layer2 = CustomDense(w2, b2, act2)

        # Layer 3: 1 neuron
        w3 = [[fa], [fb]]  # 2x1
        b3 = [0]
        act3 = tf.identity

        self.layer3 = CustomDense(w3, b3, act3)

    def call(self, x):
        t = (x - a) / (b - a)
        x = self.layer1(t)
        x = self.layer2(t)
        x = self.layer3(t)
        return x
    
    class SumOfSubNetworks(tf.keras.Model):
        def __init__(self, num_subnets=3):
            super(SumOfSubNetworks, self).__init__()
            self.subnets = [FixedSubNetwork() for _ in range(num_subnets)]

        def call(self, inputs):
            outputs = [subnet(inputs) for subnet in self.subnets]
            return tf.add_n(outputs)