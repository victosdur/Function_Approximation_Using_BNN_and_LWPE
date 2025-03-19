This repository contains data and experiments associated to the paper: "Barycentric Neural Networks for Function Approximation: A Green AI Approach Using Persistent Entropy Loss". European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases 2025 [ECML PKDD 2025](https://ecmlpkdd.org/2025/).

## Usage

To run the Jupyter notebooks correctly, you need to create a Python virtual environment (This has been developed with Python3.9.13):

```bash
virtualenv -p python env
```

Next, activate the virtual environment (we use the source command because we are working in WSL):

```bash
env\Scripts\activate
```

Finally, install the required libraries:

```bash
pip install numpy pandas matplotlib tqdm gudhi tensorflow
```

If Jupyter Notebook cannot find a kernel, you may need to install ipykernel. Run the following command to resolve this issue:

```bash
pip install ipykernel
```

## Repository structure

- `utilsBaricentricNeuralNetwork.py`: Contains the implementation of the Baricentric Neural Network, as proposed in the paper, available TensorFlow framework.

- `utilsTopology.py`: Provides the necessary tools for topology and topological data analysis used in the experiments.

- `utils.py`: Contains  necessary function to run some of the following experiments (notebooks).

- `RepresentCPLF_BNN.ipynb`: Includes experiments on representing certain CPLFs (Continuous Piecewise Linear Functions) using the proposed Baricentric Neural Network.

- `AproxContinuousFunction_BNN.ipynb`: Contains experiments on approximating continuous functions using the proposed Baricentric Neural Network.

- `Compare_PELoss`: Contains comparison between different variants of persistent entropy (normal persistent entropy and limit persistent entropy, which is our variant), in order to detect which is the best for the usage of it as loss function.

- `OptimizePoints.ipynb`: Contains experiments aimed at optimizing the fixed points used to construct the Baricentric Neural Network that approximates continuous functions in the Tensorflow framework using persistent entropy as loss function. Also compare the persistent entropy-based loss function proposed with the MSE loss function.

- `FiguresPaper`: Contains the code to generate illustrative figures that appear in the paper. These figures are saved in the `figures` folder, along with other figures generated in the experimental notebooks that have been commented.

- `BNNastypicalNN.ipynb`: Shows a simple example of how to compute the BNN as a typical neural network using the Dense layers of Tensorflow frameworks.

## Citation and Reference

If you want to use our code for your experiments, please cite our paper.

For further information, please contact us at: vtoscano@us.es
