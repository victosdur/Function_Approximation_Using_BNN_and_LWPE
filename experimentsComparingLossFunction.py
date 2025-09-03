import tensorflow as tf
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import gudhi as gd
from utilsTopology import *
from utilsBaricentricNeuralNetwork import *
from utils import *
import yfinance as yf

fontsize=16

plt.rcParams.update({
    "axes.titlesize": 14,    
    "axes.labelsize": fontsize,   
})

def train_model(x_train, y_train, loss_name, layer, dgmRef, num_points_aprox, num_iter=50):
    
    #data 1
    # x_points = tf.Variable(tf.cast(tf.linspace(-10,10,num_points_aprox), dtype=tf.float32) ,trainable=True)
    x_points = tf.Variable([-10,-6.5,-3.3,-0.2,3.,6.1,9.2,10],trainable=True)
    #data 2
    # x_points = tf.Variable(tf.cast(tf.linspace(0,len(y_train)-1,num_points_aprox), dtype=tf.float32) ,trainable=True)
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)


    history = {
        "epoch": [],
        "loss": [],
        "lossValue": [],
        "MSE": [],
        "RMSE": [],
        "MAE": [],
        "LogCosh": [],
        "LWPE": [],
    }
    best_loss_value = float('inf')
    for epoch in tqdm(range(num_iter), desc=f"Training with {loss_name}"):
        with tf.GradientTape() as tape:
            tape.watch(x_points)
            y_points = interpolation_tf(x_train, y_train, x_points)
            points = tf.stack([x_points, y_points], axis=1)
            model = BaricentricNetwork(points)
            y_aprox = model(tf.expand_dims(x_train,axis=1))
            dgmsAprox = layer.call(y_aprox)
            dgmAprox = dgmsAprox[0][0]

            if loss_name == "LWPE":
                loss_value = loss_functions[loss_name](dgmRef, dgmAprox)
            elif loss_name == "RMSE":
                loss_value = tf.sqrt((loss_functions[loss_name](y_train, y_aprox)))
            else:
                loss_value = loss_functions[loss_name](y_train, y_aprox)

        gradients = tape.gradient(loss_value, [x_points])
        # print(gradients)
        gradients[0] = tf.tensor_scatter_nd_update(gradients[0], [[0], [num_points_aprox - 1]], [0.0, 0.0])


        if epoch == 0:
            initial_model = model
            initial_x_points = tf.identity(x_points)
            initial_y_points = tf.identity(y_points)
        
        optimizer.apply_gradients(zip(gradients, [x_points]))
        if loss_value.numpy().item() < best_loss_value:
            best_loss_value = loss_value.numpy().item()
            best_x_points = tf.identity(x_points)
            best_y_points = tf.identity(y_points)
            best_model = model
            best_iter = epoch
        if epoch == num_iter - 1:
            last_model = model
            last_x_points = tf.identity(x_points)
            last_y_points = tf.identity(y_points)

        mse_val = loss_functions["MSE"](y_train, y_aprox).numpy()
        mae_val = loss_functions["MAE"](y_train, y_aprox).numpy()
        rmse_val = tf.sqrt(loss_functions["RMSE"](y_train, y_aprox)).numpy()
        logcosh_val = loss_functions["LogCosh"](y_train, y_aprox).numpy()
        LWPE_val = loss_functions["LWPE"](dgmRef,dgmAprox).numpy().item()
    
        history["epoch"].append(epoch)
        history["loss"].append(loss_name)
        if loss_name == "LWPE":
            history["lossValue"].append(loss_value.numpy().item())
        else:
            history["lossValue"].append(loss_value.numpy())
        history["MSE"].append(mse_val)
        history["MAE"].append(mae_val)
        history["RMSE"].append(rmse_val)
        history["LogCosh"].append(logcosh_val)
        history["LWPE"].append(LWPE_val)

    plot_and_save_approximation(model=initial_model,x_train=x_train,y_train=y_train,x_points=initial_x_points,y_points=initial_y_points,
                         domain=domain,filename=f'results/InitialApproximation_{loss_name}_{dataName}.png',fontsize=fontsize)
    plot_and_save_approximation(model=best_model,x_train=x_train,y_train=y_train,x_points=best_x_points,y_points=best_y_points,
                         domain=domain,filename=f'results/BestApproximation_{loss_name}_{dataName}.png',fontsize=fontsize)
    plot_and_save_approximation(model=last_model,x_train=x_train,y_train=y_train,x_points=last_x_points,y_points=last_y_points,
                         domain=domain,filename=f'results/LastApproximation_{loss_name}_{dataName}.png',fontsize=fontsize)

    return pd.DataFrame(history)

# data 1
domain =[-10,10]
num_points=250
dataName="SyntheticWithNoise"
function=np.sin
x_train, y_train = generate_training_data(function, (domain[0], domain[1]), num_points)
np.random.seed(7)
ruido = np.random.normal(0, 0.05, size=x_train.shape)
y_train = y_train + ruido
x_train = tf.constant(x_train,dtype=tf.float32)
y_train = tf.constant(y_train,dtype=tf.float32)
num_points_optimize = 8

# #data 2
# data = yf.download('GLD', start='2023-01-01', end='2024-01-01')  # ETF de oro
# dataName="economic"
# prices = data['Close'].values
# domain =[0,prices.shape[0]-1]
# num_points=len(prices)
# x_train, y_train = np.arange(0, len(prices)), prices.reshape(-1)
# x_train = tf.constant(x_train,dtype=tf.float32)
# y_train = tf.constant(y_train,dtype=tf.float32)
# num_points_optimize = 30

stbase = gd.SimplexTree()
for i in range(num_points - 1):
    stbase.insert([i, i + 1], -1e10)
layer = LowerStarLayer(simplextree=stbase)
dgmsRef = layer.call(tf.Variable(y_train))
dgmRef = dgmsRef[0][0]
PERef=persistent_entropy(dgmRef)
LWPEref=length_weighted_persistent_entropy(dgmRef)

 
distances = tf.abs(dgmRef[:, 0] - dgmRef[:, 1])  # (n,)
top_x_indices = tf.argsort(distances, direction='DESCENDING')[:int(num_points_optimize/2)]
dgmRefFilt = tf.gather(dgmRef, top_x_indices)
PERefFilt=persistent_entropy(dgmRefFilt)
LWPERefFilt=length_weighted_persistent_entropy(dgmRefFilt)

loss_functions = {
    "LWPE": LengthWeightedPersistentEntropyLoss(),
    "MSE": tf.keras.losses.MeanSquaredError(),
    "RMSE": tf.keras.losses.MeanSquaredError(),
    "MAE": tf.keras.losses.MeanAbsoluteError(),
    "LogCosh": tf.keras.losses.LogCosh(),

}

resultados = {}
for name, loss_fn in loss_functions.items():
    print(f"Entrenando con la función de pérdida: {name}")
    df_result = train_model(x_train, y_train, name, layer, dgmRefFilt, num_points_optimize)
    resultados[name] = df_result
    print(f"Resultados guardados para perdida {name}")

with pd.ExcelWriter(f"results/learning_curves_comparison_{dataName}.xlsx") as writer:
    for name, df in resultados.items():
        df.to_excel(writer, sheet_name=name, index=False)

print(resultados)

fig,axes = plt.subplots(nrows=2,ncols=3,figsize=(12, 12))
for name, df in resultados.items():
    axes[0,0].plot(df["epoch"], df["MSE"],"--", label=f"{name}")
    axes[0,1].plot(df["epoch"], df["RMSE"],"--", label=f"{name}")
    axes[0,2].plot(df["epoch"], df["MAE"],"--", label=f"{name}")
    axes[1,0].plot(df["epoch"], df["LogCosh"],"--", label=f"{name}")
    axes[1,1].plot(df["epoch"], df["LWPE"],"--", label=f"{name}")

axes[0,0].set_title("Learning curves - MSE")
axes[0,1].set_title("Learning curves - RMSE")
axes[0,2].set_title("Learning curves - MAE")
axes[1,0].set_title("Learning curves - LogCosh")
axes[1,1].set_title("Learning curves - LWPE")
axes[0,0].set_xlabel("Epochs")
axes[0,1].set_xlabel("Epochs")  
axes[0,2].set_xlabel("Epochs")
axes[1,0].set_xlabel("Epochs")
axes[1,1].set_xlabel("Epochs")
axes[0,0].set_ylabel("MSE")
axes[0,1].set_ylabel("RMSE")
axes[0,2].set_ylabel("MAE")
axes[1,0].set_ylabel("LogCosh")
axes[1,1].set_ylabel("LWPE")
# plt.suptitle("Learning Curves Comparison", fontsize=16)
axes[1,1].legend(title="Loss function", bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=fontsize)
axes[1,2].remove()

plt.subplots_adjust(wspace=0.3, hspace=0.35)
plt.savefig(f"results/learning_loss_curves_{dataName}.png")
plt.show()
