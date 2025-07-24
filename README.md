This repository contains data and experiments associated to the paper: Toscano-Duran,V., Gonzalez-Diaz,R., and Gutiérrez-Naranjo, M.A., "Barycentric Neural Networks for Function Approximation: A Green AI Approach Using Persistent Entropy Loss".

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

- `utilsBaricentricNeuralNetwork.py`: It contains the implementation of the proposed Barycentric Neural Network (BNN) in TensorFlow framework.

- `utilsTopology.py`: It provides the topological tools necessary for the experiments.

- `utils.py`: It contains some necessary functions used in the experiments.

- `RepresentCPLF_BNN.ipynb`: It includes experiments on representing certain continuous piecewise linear functions (CPLFs) using the proposed barycentric neural network.

- `AproxContinuousFunction_BNN.ipynb`: It contains experiments on approximating continuous functions using the proposed Baricentric Neural Network.

- `Compare_PELoss`: It contains a comparison between different variants of persistent entropy, such as normal persistent entropy (PE) and length-weighted persistent entropy (LWPE), which we have developed. The aim is to determine which variant is best suited to use as a loss function.

- `ApproximatingFunctionsUsingBNN_ComparisonMSEandLWPE.ipynb`: This contains experiments aimed at optimising the base points used to construct the barycentric neural network that approximates continuous functions in the TensorFlow framework using the LWPE loss function. It also compares the proposed persistent entropy-based loss function (LWPE) with the mean squared error (MSE) loss function.

- `experimentsComparingLossFunction.py`: It contains experiments aimed at optimising the base points used to construct the barycentric neural network that approximates continuous functions. It compares different loss functions (mean squared error (MSE), root mean squared error (RMSE), mean absolute error (MAE), logcosh and our persistent entropy variant (LWPE)).

- `results`: It contains the results of the different loss functions at the time the previous script was run.

- `FiguresPaper`: This contains the code used to generate the illustrative figures that appear in the paper. These figures are saved in the 'figures' folder, alongside other figures that have been commented on and generated in the experimental notebooks.


## Citation and Reference

If you want to use our code for your experiments, please cite our paper.

For further information, please contact us at: vtoscano@us.es
