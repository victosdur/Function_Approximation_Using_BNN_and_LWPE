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

- `utilsBaricentricNeuralNetwork.py`: Contains the implementation of the Baricentric Neural Network, as proposed in the paper, available in TensorFlow framework.

- `utilsTopology.py`: Provides the necessary topological tools used in the experiments.

- `utils.py`: Contains necessary function used in the experiments.

- `RepresentCPLF_BNN.ipynb`: Includes experiments on representing certain CPLFs (Continuous Piecewise Linear Functions) using the proposed Baricentric Neural Network.

- `AproxContinuousFunction_BNN.ipynb`: Contains experiments on approximating continuous functions using the proposed Baricentric Neural Network.

- `Compare_PELoss`: Contains comparison between different variants of persistent entropy (normal persistent entropy (PE) and length weighted persistent entropy (LWPE), which is our variant), in order to detect which is the best for the usage of it as loss function.

- `ApproximatingFunctionsUsingBNN_ComparisonMSEandLWPE.ipynb`: Contains experiments aimed at optimizing the fixed points used to construct the Baricentric Neural Network that approximates continuous functions in the Tensorflow framework using LWPE as loss function. Also compare the persistent entropy-based loss function proposed with the MSE loss function.

- `experimentsComparingLossFunction.py`: Contains experiments aimed at optimizing the fixed points used to construct the Baricentric Neural Network that approximates continuous functions in the Tensorflow framework, comparing different loss functions (MSE, RMSE, MAE, Logcosh, Persistent Entropy (LWPE)).

- `results`: Contains the results of the different loss function at the time of running previous script.

- `FiguresPaper`: Contains the code to generate illustrative figures that appear in the paper. These figures are saved in the `figures` folder, along with other figures generated in the experimental notebooks that have been commented.


## Citation and Reference

If you want to use our code for your experiments, please cite our paper.

For further information, please contact us at: vtoscano@us.es
