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

def train_model(x_train, y_train, loss_name, layer, dgmRef, num_points_aprox, num_iter=50):
    
    #data 1
    # x_points = tf.Variable([-10,-6.5,-3.3,-0.2,3.,6.1,9.2,10],trainable=True)
    #data 2
    # x_points = tf.Variable([0,15.,35.,50.,65.,75.,85.,95,110,130,150,175,200,220,235,250],trainable=True)
    x_points = tf.Variable(tf.cast(tf.linspace(0,len(y_train)-1,num_points_aprox), dtype=tf.float32) ,trainable=True)
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)


    history = {
        "epoch": [],
        "loss": [],
        "mse": [],
        "rmse": [],
        "mae": [],
        "logcosh": [],
        "entropyLim": [],
    }

    for epoch in tqdm(range(num_iter), desc=f"Training with {loss_name}"):
        with tf.GradientTape() as tape:
            tape.watch(x_points)
            y_points = interpolation_tf(x_train, y_train, x_points)
            points = tf.stack([x_points, y_points], axis=1)
            model = BaricentricSigmaNetworkTf(points)
            y_aprox = model(x_train)
            dgmsAprox = layer.call(y_aprox)
            dgmAprox = dgmsAprox[0][0]

            if loss_name == "persistent_entropy":
                loss_value = loss_functions[loss_name](dgmRef, dgmAprox)
            elif loss_name == "rmse":
                loss_value = tf.sqrt((loss_functions[loss_name](y_train, y_aprox)))
            else:
                loss_value = tf.reduce_mean(loss_functions[loss_name](y_train, y_aprox))

        gradients = tape.gradient(loss_value, [x_points])
        # print(gradients)
        gradients[0] = tf.tensor_scatter_nd_update(gradients[0], [[0], [num_points_aprox - 1]], [0.0, 0.0])
        optimizer.apply_gradients(zip(gradients, [x_points]))

        # Calcular métricas
        mse_val = tf.reduce_mean(tf.keras.losses.MSE(y_train, y_aprox)).numpy()
        mae_val = tf.reduce_mean(tf.keras.losses.MAE(y_train, y_aprox)).numpy()
        rmse_val = tf.sqrt(tf.reduce_mean(tf.square(y_train - y_aprox))).numpy()
        logcosh_val = tf.reduce_mean(tf.keras.losses.logcosh(y_train, y_aprox)).numpy()
        entropyLim_val = loss_functions["persistent_entropy"](dgmRef,dgmAprox).numpy().item()

        # Guardar en histórico
        history["epoch"].append(epoch)
        history["loss"].append(loss_name)
        history["mse"].append(mse_val)
        history["mae"].append(mae_val)
        history["rmse"].append(rmse_val)
        history["logcosh"].append(logcosh_val)
        history["entropyLim"].append(entropyLim_val)

    return pd.DataFrame(history)


# data 1
# domain =[-10,10]
# num_points=250
# function=np.sin
# x_train, y_train = generate_training_data(function, (domain[0], domain[1]), num_points)
# np.random.seed(7)
# ruido = np.random.normal(0, 0.05, size=x_train.shape)
# y_train = y_train + ruido
# x_train = tf.constant(x_train,dtype=tf.float32)
# y_train = tf.constant(y_train,dtype=tf.float32)
# num_points_optimize = 8

#data 2
data = yf.download('GLD', start='2023-01-01', end='2024-01-01')  # ETF de oro
prices = data['Close'].values
domain =[0,prices.shape[0]-1]
num_points=len(prices)
x_train, y_train = np.arange(0, len(prices)), prices.reshape(-1)
x_train = tf.constant(x_train,dtype=tf.float32)
y_train = tf.constant(y_train,dtype=tf.float32)
num_points_optimize = 16

# Topología base
stbase = gd.SimplexTree()
for i in range(num_points - 1):
    stbase.insert([i, i + 1], -1e10)
layer = LowerStarLayer(simplextree=stbase)
dgmsRef = layer.call(tf.Variable(y_train))
dgmRef = dgmsRef[0][0]
entropyRef=persistent_entropy_tf(dgmRef)
entropyRefLim=persistent_entropy_lim_tf(dgmRef)

 
distances = tf.abs(dgmRef[:, 0] - dgmRef[:, 1])  # (n,)
top_x_indices = tf.argsort(distances, direction='DESCENDING')[:int(num_points_optimize/2)]
dgmRefFilt = tf.gather(dgmRef, top_x_indices)
entropyRefFilt=persistent_entropy_tf(dgmRefFilt)
entropyRefFiltLim=persistent_entropy_lim_tf(dgmRefFilt)

# Lista de funciones de pérdida
loss_functions = {
    "persistent_entropy": PersistentEntropyLossLimTF(),
    "mse": tf.keras.losses.MeanSquaredError(),
    "rmse": tf.keras.losses.MeanSquaredError(),
    "mae": tf.keras.losses.MeanAbsoluteError(),
    "logcosh": tf.keras.losses.LogCosh(),

}

# Entrenamiento y almacenamiento
resultados = {}
for name, loss_fn in loss_functions.items():
    print(f"Entrenando con la función de pérdida: {name}")
    df_result = train_model(x_train, y_train, name, layer, dgmRefFilt, num_points_optimize)
    resultados[name] = df_result
    print(f"Resultados guardados para perdida {name}")

# Guardar a Excel
with pd.ExcelWriter("learning_curves_comparison.xlsx") as writer:
    for name, df in resultados.items():
        df.to_excel(writer, sheet_name=name, index=False)

print(resultados)

# # Visualización
fig,axes = plt.subplots(nrows=2,ncols=3,figsize=(12, 12))
for name, df in resultados.items():
    axes[0,0].plot(df["epoch"], df["mse"],"--", label=f"{name}")
    axes[0,1].plot(df["epoch"], df["rmse"],"--", label=f"{name}")
    axes[0,2].plot(df["epoch"], df["mae"],"--", label=f"{name}")
    axes[1,0].plot(df["epoch"], df["logcosh"],"--", label=f"{name}")
    axes[1,1].plot(df["epoch"], df["entropyLim"],"--", label=f"{name}")

axes[0,0].set_title("Learning curves - MSE")
axes[0,1].set_title("Learning curves - RMSE")
axes[0,2].set_title("Learning curves - MAE")
axes[1,0].set_title("Learning curves - LogCosh")
axes[1,1].set_title("Learning curves - Persistent Entropy Lim")
axes[0,0].set_xlabel("Epochs")
axes[0,1].set_xlabel("Epochs")  
axes[0,2].set_xlabel("Epochs")
axes[1,0].set_xlabel("Epochs")
axes[1,1].set_xlabel("Epochs")
axes[0,0].set_ylabel("MSE")
axes[0,1].set_ylabel("RMSE")
axes[0,2].set_ylabel("MAE")
axes[1,0].set_ylabel("LogCosh")
axes[1,1].set_ylabel("Persistent Entropy Lim")
plt.suptitle("Curvas de aprendizaje - Comparación de funciones de pérdida")
axes[0,0].legend(title="Loss function")
axes[0,1].legend(title="Loss function")
axes[0,2].legend(title="Loss function")
axes[1,0].legend(title="Loss function")
axes[1,1].legend(title="Loss function")
axes[1,2].remove()

plt.subplots_adjust(wspace=0.2, hspace=0.35)
plt.savefig("loss_curves.png")
plt.show()
