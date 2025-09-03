import numpy as np
from tensorflow.keras import Model # for version 3.9.13 (the one everything has been tested). For 3.10.0 and above, use `from tensorflow import Model`
import tensorflow as tf


def stepestrella(x):
    return tf.cast(x > 0, dtype=tf.float32)

class CustomDense(tf.keras.layers.Layer):
    def __init__(self, weights, biases, activations):
        super(CustomDense, self).__init__()
        self.fixed_weights = tf.constant(weights, dtype=tf.float32)
        self.fixed_biases = tf.constant(biases, dtype=tf.float32)
        self.activations = activations  # lista de funciones
    def call(self, inputs):
        z = tf.matmul(inputs, self.fixed_weights)
        z = tf.add(z, self.fixed_biases)
        if isinstance(self.activations, list):
            out = tf.stack([self.activations[i](z[:, i]) for i in range(z.shape[1])], axis=1)
        else:
            out = self.activations(z)
        return out
    
class BaricentricNetworkSegment(Model):
    def __init__(self, a, b, fa, fb, **kwargs):
        super(BaricentricNetworkSegment, self).__init__(**kwargs)
        self.a = a
        self.b = b
        self.fa = fa
        self.fb = fb
        # Layer 1: 4 neurons
        w1 = [[1., -1., 1., -1.]]  # input_dim=1, output_dim=4
        b1 = [0., 0., -1., 1.]
        act1 = [tf.nn.relu, stepestrella, stepestrella, tf.nn.relu]

        self.layer1 = CustomDense(w1, b1, act1)

        # Layer 2: 2 neurons
        w2 = [[-1., 0.], [-2., -2.], [-2., -2.], [0., -1.]]  # 4x2
        b2 = [1.0, 1.0]
        act2 = tf.nn.relu

        self.layer2 = CustomDense(w2, b2, act2)

        # Layer 3: 1 neuron
        w3 = tf.stack([fa, fb])[:, tf.newaxis]  # 2x1
        b3 = [0.]
        act3 = tf.identity

        self.layer3 = CustomDense(w3, b3, act3)

    def call(self, x):
        t = (x - self.a) / (self.b - self.a)
        z = self.layer1(t)
        z = self.layer2(z)
        z = self.layer3(z)
        return z
    
class BaricentricNetwork(Model):
    def __init__(self, points):
        super(BaricentricNetwork, self).__init__()
        self.x_coords = points[:,0]
        self.y_values = points[:,1]
        num_segments = len(self.x_coords) - 1 
        self.subnets = [BaricentricNetworkSegment(a = self.x_coords[i], b = self.x_coords[i+1], fa = self.y_values[i], fb = self.y_values[i+1]) for i in range(num_segments)]
    
    def call(self, x):
        outputs = [subnet(x) for subnet in self.subnets]  # List of outputs from each subnet

        # 1st way, divided by num of active subnets
        outputs_stack = tf.stack(outputs, axis=0)  # Shape: [num_subnets, batch_size]
        active_mask = tf.cast(outputs_stack > 0, dtype=tf.float32)  
        sum_outputs = tf.reduce_sum(outputs_stack, axis=0)
        num_active = tf.reduce_sum(active_mask, axis=0)  # [batch_size]
        # Avoid division by zero
        num_active = tf.maximum(num_active, 1.0)
        # Final prediction: average if multiple subnets are active (joining point, otherwise just the active one)
        final_output = sum_outputs / num_active  # [batch_size]
        
        return final_output
