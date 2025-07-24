import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

fontsize=16
    
def plot_persistent_barcode(tensor, nameSave=None, ax=None):
    num_bars = tensor.shape[0]

    if ax is None:
        fig, ax = plt.subplots()

    for i, (birth, death) in enumerate(tensor):
        ax.plot([birth, death], [i, i], 'b', lw=4)  # Línea azul para cada barra
    
    ax.set_xlabel("Birth-Death Interval", fontsize=12)
    ax.set_ylabel("Index", fontsize=12)

    ax.set_title("Persistence barcode", fontsize=12)

    ax.grid(True, linestyle='--', alpha=0.6)

    if ax is None and nameSave is not None:
        plt.savefig(nameSave, dpi=300, bbox_inches="tight")

# function for compute PE (without tensorflow, just for test the development in tensorflow) from persistence barcode
def computePersistenceEntropy(persistentBarcode):
    l=[]
    for i in persistentBarcode:
        l.append(i[1]-i[0])
    L = sum(l)
    p=l/L
    entropia=-np.sum(p*np.log(p))
    return entropia # round(entropia,4)

# Persistent entropy calculation in TensorFlow
def persistent_entropy(D):
    persistence = tf.experimental.numpy.diff(D)
    persistence = tf.boolean_mask(tf.abs(persistence), tf.math.is_finite(persistence))
    
    P = tf.reduce_sum(persistence)
    probabilities = persistence / P
    
    # Ensures that a probability of zero will result in a logarithm of zero as well
    log_prob = tf.zeros_like(probabilities)
    mask = probabilities > 0
    log_prob = tf.tensor_scatter_nd_update(log_prob, tf.where(mask), tf.math.log(probabilities[mask]))
    
    return -tf.reduce_sum(probabilities * log_prob)

# Length-weighted persistent entropy calculation in TensorFlow
def length_weighted_persistent_entropy(D):
    persistence = tf.experimental.numpy.diff(D)
    persistence = tf.boolean_mask(tf.abs(persistence), tf.math.is_finite(persistence))
    
    P = tf.reduce_sum(persistence)
    probabilities = persistence / P
    
    # Ensures that a probability of zero will result in a logarithm of zero as well
    log_prob = tf.zeros_like(probabilities)
    mask = probabilities > 0
    log_prob = tf.tensor_scatter_nd_update(log_prob, tf.where(mask), tf.math.log(probabilities[mask]))
    
    return -tf.reduce_sum(persistence * log_prob)
    
# Persistent entropy loss function in TensorFlow
class PersistentEntropyLoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stat_fn = persistent_entropy
    
    def call(self, X, Y):
        stat_ref = self.stat_fn(X)
        stat_aprox = self.stat_fn(Y)
        return tf.abs(stat_aprox - stat_ref)

# Length-weighted persistent entropy loss function in TensorFlow
class LengthWeightedPersistentEntropyLoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stat_fn = length_weighted_persistent_entropy
    
    def call(self, X, Y):
        stat_ref = self.stat_fn(X)
        stat_aprox = self.stat_fn(Y)
        return tf.abs(stat_aprox - stat_ref)


# function for lower star persistencia diagram differentiable inside tensorflow
def LowerStarsSimplex(simplextree, filtration_values, dimensions, homology_coeff_field, persistence_dim_max):
    simplextree.reset_filtration(-np.inf, 0)

    # Assign new filtration values
    for i in range(simplextree.num_vertices()):
        simplextree.assign_filtration([i], filtration_values[i])
    simplextree.make_filtration_non_decreasing()
    
    # Compute persistence diagram
    simplextree.compute_persistence(homology_coeff_field=homology_coeff_field, persistence_dim_max=persistence_dim_max)
    
    # Get vertex pairs for optimization. First, get all simplex pairs
    pairs = simplextree.lower_star_persistence_generators()
    
    L_indices = []
    for dimension in dimensions:
    
        finite_pairs = pairs[0][dimension] if len(pairs[0]) >= dimension+1 else np.empty(shape=[0,2])
        finite_pairs = np.vstack((finite_pairs,[np.argmin(filtration_values).item(),np.argmax(filtration_values).item()]))
        essential_pairs = pairs[1][dimension] if len(pairs[1]) >= dimension+1 else np.empty(shape=[0,1])
        
        finite_indices = np.array(finite_pairs.flatten(), dtype=np.int32)
        essential_indices = np.array(essential_pairs.flatten(), dtype=np.int32)

        L_indices.append((finite_indices, essential_indices))

    return L_indices

class LowerStarLayer(tf.keras.layers.Layer):
    def __init__(self, simplextree, homology_dimensions=[0], min_persistence=None, homology_coeff_field=11, persistence_dim_max=0, **kwargs):
        super().__init__(**kwargs)
        self.dimensions  = homology_dimensions
        self.simplextree = simplextree
        self.min_persistence = min_persistence if min_persistence is not None else [0. for _ in range(len(self.dimensions))]
        self.hcf = homology_coeff_field
        self.pdm = persistence_dim_max
        assert len(self.min_persistence) == len(self.dimensions)
        
    def call(self, filtration_values):
        indices = LowerStarsSimplex(self.simplextree, filtration_values.numpy(), self.dimensions, self.hcf, self.pdm)
        # Get persistence diagrams
        self.dgms = []
        for idx_dim, dimension in enumerate(self.dimensions):
            finite_dgm = tf.reshape(tf.gather(filtration_values, indices[idx_dim][0]), [-1,2])
            essential_dgm = tf.reshape(tf.gather(filtration_values, indices[idx_dim][1]), [-1,1])
            min_pers = self.min_persistence[idx_dim]
            if min_pers >= 0:
                persistent_indices = tf.where(tf.math.abs(finite_dgm[:,1]-finite_dgm[:,0]) > min_pers)
                self.dgms.append((tf.reshape(tf.gather(finite_dgm, indices=persistent_indices),[-1,2]), essential_dgm))
            else:
                self.dgms.append((finite_dgm, essential_dgm))
        return self.dgms
        
