import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import entropy
import scipy.sparse as sparse
from ripser import ripser

fontsize=16
    
# function for calculate persistence diagramas using LowerStar filtration (not torch).
def calculatePersistenceDiagrams_LowerStar(t,x):
    N = x.shape[0]
    I = np.arange(N-1)
    J = np.arange(1, N)
    V = np.maximum(x[0:-1], x[1::])
    # Add vertex birth times along the diagonal of the distance matrix
    I = np.concatenate((I, np.arange(N)))
    J = np.concatenate((J, np.arange(N)))
    V = np.concatenate((V, x))
    #Create the sparse distance matrix
    D = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    dgms = ripser(D, maxdim=3, distance_matrix=True)['dgms'] # doesn't matter the maxdim as there is only diagram for dimension 0 in the lowerstar filtration.
    return dgms

# function for obtain persistence diagrama of specific dimension.
def obtainDiagramDimension(Diagrams,dimension):
    dgm=Diagrams[dimension]
    dgm = dgm[dgm[:, 1]-dgm[:, 0] > 1e-3, :]
    return dgm

# function for remove infinity values for persistence diagram.
def limitDiagramLowerStar(Diagram,maximumFiltration):
    infinity_mask = np.isinf(Diagram)
    Diagram[infinity_mask] = maximumFiltration + 1
    return Diagram

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

# function for compute PE from persistence barcode
def computePersistenceEntropy(persistentBarcode):
    l=[]
    for i in persistentBarcode:
        l.append(i[1]-i[0])
    L = sum(l)
    p=l/L
    entropia=-np.sum(p*np.log(p))
    return entropia # round(entropia,4)

# Persistent entropy calculation in TensorFlow
def persistent_entropy_tf(D):
    persistence = tf.experimental.numpy.diff(D)
    persistence = tf.boolean_mask(tf.abs(persistence), tf.math.is_finite(persistence))
    
    P = tf.reduce_sum(persistence)
    probabilities = persistence / P
    
    # Ensures that a probability of zero will result in a logarithm of zero as well
    log_prob = tf.zeros_like(probabilities)
    mask = probabilities > 0
    log_prob = tf.tensor_scatter_nd_update(log_prob, tf.where(mask), tf.math.log(probabilities[mask]))
    
    return -tf.reduce_sum(probabilities * log_prob)

def persistent_entropy_lim_tf(D):
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
class PersistentEntropyLossTF(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stat_fn = persistent_entropy_tf
    
    def call(self, X, Y):
        stat_ref = self.stat_fn(X)
        stat_aprox = self.stat_fn(Y)
        return tf.abs(stat_aprox - stat_ref)

class PersistentEntropyLossLimTF(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stat_fn = persistent_entropy_lim_tf
    
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
        
# function for plot signal data and his persistent diagram of specific dimension.
def plotSignal_PersistentDiagram(t,signal,dimension):
    dgms = calculatePersistenceDiagrams_LowerStar(t,signal)    
    dgm0 = obtainDiagramDimension(dgms,dimension)
    allgrid = np.unique(dgm0.flatten())
    allgrid = allgrid[allgrid < np.inf]
    xs = np.unique(dgm0[:, 0])
    ys = np.unique(dgm0[:, 1])
    ys = ys[ys < np.inf]

    #Plot the time series and the persistence diagram
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.plot(t, signal,'-o')
    ax = plt.gca()
    ax.set_yticks(allgrid)
    ax.set_xticks([])
    plt.grid(linewidth=1, linestyle='--')
    plt.title("Señal")
    plt.xlabel("t")

    plt.subplot(122)
    ax = plt.gca()
    ax.set_yticks(ys)
    ax.set_xticks(xs)
    plt.grid(linewidth=1, linestyle='--')
    plot_diagrams(dgm0, size=50)
    plt.title(f"Persistence Diagram, dimension = {dimension}")
    plt.show()

# function for plot peristence diagrams.
def plot_diagrams(
    diagrams,
    plot_only=None,
    title=None,
    xy_range=None,
    labels=None,
    colormap="default",
    size=20,
    ax_color=np.array([0.0, 0.0, 0.0]),
    diagonal=True,
    lifetime=False,
    legend=True,
    show=False,
    ax=None
):
    """A helper function to plot persistence diagrams. 

    Parameters
    ----------
    diagrams: ndarray (n_pairs, 2) or list of diagrams
        A diagram or list of diagrams. If diagram is a list of diagrams, 
        then plot all on the same plot using different colors.
    plot_only: list of numeric
        If specified, an array of only the diagrams that should be plotted.
    title: string, default is None
        If title is defined, add it as title of the plot.
    xy_range: list of numeric [xmin, xmax, ymin, ymax]
        User provided range of axes. This is useful for comparing 
        multiple persistence diagrams.
    labels: string or list of strings
        Legend labels for each diagram. 
        If none are specified, we use H_0, H_1, H_2,... by default.
    colormap: string, default is 'default'
        Any of matplotlib color palettes. 
        Some options are 'default', 'seaborn', 'sequential'. 
        See all available styles with

        .. code:: python

            import matplotlib as mpl
            print(mpl.styles.available)

    size: numeric, default is 20
        Pixel size of each point plotted.
    ax_color: any valid matplotlib color type. 
        See [https://matplotlib.org/api/colors_api.html](https://matplotlib.org/api/colors_api.html) for complete API.
    diagonal: bool, default is True
        Plot the diagonal x=y line.
    lifetime: bool, default is False. If True, diagonal is turned to False.
        Plot life time of each point instead of birth and death. 
        Essentially, visualize (x, y-x).
    legend: bool, default is True
        If true, show the legend.
    show: bool, default is False
        Call plt.show() after plotting. If you are using self.plot() as part 
        of a subplot, set show=False and call plt.show() only once at the end.
    """

    ax = ax or plt.gca()
    plt.style.use(colormap)

    xlabel, ylabel = "Birth", "Death"

    if not isinstance(diagrams, list):
        # Must have diagrams as a list for processing downstream
        diagrams = [diagrams]

    if labels is None:
        # Provide default labels for diagrams if using self.dgm_
        labels = ["$H_{{{}}}$".format(i) for i , _ in enumerate(diagrams)]

    if plot_only:
        diagrams = [diagrams[i] for i in plot_only]
        labels = [labels[i] for i in plot_only]

    if not isinstance(labels, list):
        labels = [labels] * len(diagrams)

    # Construct copy with proper type of each diagram
    # so we can freely edit them.
    diagrams = [dgm.astype(np.float32, copy=True) for dgm in diagrams]

    # find min and max of all visible diagrams
    concat_dgms = np.concatenate(diagrams).flatten()
    has_inf = np.any(np.isinf(concat_dgms))
    finite_dgms = concat_dgms[np.isfinite(concat_dgms)]

    # clever bounding boxes of the diagram
    if not xy_range:
        # define bounds of diagram
        ax_min, ax_max = np.min(finite_dgms), np.max(finite_dgms)
        x_r = ax_max - ax_min

        # Give plot a nice buffer on all sides.
        # ax_range=0 when only one point,
        buffer = 1 if xy_range == 0 else x_r / 5

        x_down = ax_min - buffer / 2
        x_up = ax_max + buffer

        y_down, y_up = x_down, x_up
    else:
        x_down, x_up, y_down, y_up = xy_range

    yr = y_up - y_down

    if lifetime:

        # Don't plot landscape and diagonal at the same time.
        diagonal = False

        # reset y axis so it doesn't go much below zero
        y_down = -yr * 0.05
        y_up = y_down + yr

        # set custom ylabel
        ylabel = "Lifetime"

        # set diagrams to be (x, y-x)
        for dgm in diagrams:
            dgm[:, 1] -= dgm[:, 0]

        # plot horizon line
        ax.plot([x_down, x_up], [0, 0], c=ax_color)

    # Plot diagonal
    if diagonal:
        ax.plot([x_down, x_up], [x_down, x_up], "--", c=ax_color)

    # Plot inf line
    if has_inf:
        # put inf line slightly below top
        b_inf = y_down + yr * 0.95
        ax.plot([x_down, x_up], [b_inf, b_inf], "--", c="k", label=r"$\infty$")

        # convert each inf in each diagram with b_inf
        for dgm in diagrams:
            dgm[np.isinf(dgm)] = b_inf

    # Plot each diagram
    for dgm, label in zip(diagrams, labels):

        # plot persistence pairs
        ax.scatter(dgm[:, 0], dgm[:, 1], size, label=label, edgecolor="none")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    ax.set_xlim([x_down, x_up])
    ax.set_ylim([y_down, y_up])
    ax.set_aspect('equal', 'box')

    if title is not None:
        ax.set_title(title)

    if legend is True:
        ax.legend(loc="lower right")

    if show is True:
        plt.show()