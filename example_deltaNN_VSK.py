import tensorflow as tf
import numpy as np
import custom_tf
from scipy.stats import qmc
from test_disc_functions.basic_disc_functions import discfuncs_dict
from custom_tf.rbf_archetypes import rbfs_dict
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse.linalg import cg
import matplotlib
# matplotlib.use('TkAgg', force=True)  # Uncomment if you want to use this option
import matplotlib.pyplot as plt

save_pics = True
saving_folder = 'data/results_example_deltaNN_VSK'

random_seed = 42

# SET THE RANDOM SEED
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
tf.keras.utils.set_random_seed(random_seed)
tf.config.experimental.enable_op_determinism()

# NUMBER OF INTERPOLATION PTS
N = 1089
# NUMBER OF PTS PER AXIS (FOR THE TEST GRID)
neval = 60

# GENERATE HALTON POINTS FOR TRAINING/INTERPOLATION
sampler = qmc.Halton(d=2, scramble=False)
H1 = sampler.random(N + 1)
X = H1[1:(N + 1), :]

# GRID FOR TEST DATA
X1test, X2test = np.meshgrid(np.linspace(0, 1, neval), np.linspace(0, 1, neval))
Xtest = np.hstack([X1test.flatten().reshape(X1test.size, 1), X2test.flatten().reshape(X2test.size, 1)])

# TEST FUNCTION
f = discfuncs_dict['dfunc_basic_004']

# OUTPUTS
y = f(X)
ytest = f(Xtest)

# NORMALIZATION OF y
yscaler = MinMaxScaler()
yscaler.fit(y.reshape(-1, 1))

y_original = y.copy()
ytest_original = ytest.copy()

y = yscaler.transform(y.reshape(-1, 1)).flatten()
ytest = yscaler.transform(ytest.reshape(-1, 1)).flatten()

# PSIMODEL (example of simple deltaNN, just using deltaLayers into a Keras sequential model)
model_psi = tf.keras.models.Sequential(
    layers=[
        tf.keras.layers.Dense(512, activation='elu', kernel_initializer='glorot_normal', dtype=tf.float64),
        tf.keras.layers.Dense(512, activation='elu', kernel_initializer='glorot_normal', dtype=tf.float64),
        custom_tf.layers.DiscontinuityDense(32, activation='elu', kernel_initializer='glorot_normal',
                                            dtype=tf.float64),
        custom_tf.layers.DiscontinuityDense(32, activation='elu', kernel_initializer='glorot_normal',
                                            dtype=tf.float64),
        tf.keras.layers.Dense(256, activation='elu', kernel_initializer='glorot_normal', dtype=tf.float64),
        tf.keras.layers.Dense(256, activation='elu', kernel_initializer='glorot_normal', dtype=tf.float64),
        custom_tf.layers.DiscontinuityDense(16, activation='elu', kernel_initializer='glorot_normal',
                                            dtype=tf.float64),
        custom_tf.layers.DiscontinuityDense(16, activation='elu', kernel_initializer='glorot_normal',
                                            dtype=tf.float64),
        tf.keras.layers.Dense(128, activation='elu', kernel_initializer='glorot_normal', dtype=tf.float64),
        custom_tf.layers.DiscontinuityDense(16, activation='elu', kernel_initializer='glorot_normal',
                                            dtype=tf.float64),
        tf.keras.layers.Dense(128, activation='elu', kernel_initializer='glorot_normal', dtype=tf.float64),
        tf.keras.layers.Dense(1, activation='linear', kernel_initializer='glorot_normal', dtype=tf.float64)
    ]
)

# BUILD THE MODEL:
model_psi.build(X.shape)

# RBFs
rbf_class = rbfs_dict['matern_c2']
# Possible choices:
# 1. rbfs_dict['matern_c2']
# 2. rbfs_dict['gaussian']

shape_param = 0.12

# OPTIMIZER
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.0005)
EPOCHS = 2000

# VSKFitter
model = custom_tf.model_archetypes.VSKFitter(
    model_psi=model_psi,
    rbf_centers_coo=X,
    shape_param=shape_param * np.ones(N),
    rbf=rbf_class()
)

# COMPILE THE VSKFitter MODEL
model.compile(optimizer=OPTIMIZER, loss='mse', metrics=['mae'])
# TRAIN THE VSKFitter MODEL
history = model.fit(
    X, y,
    epochs=EPOCHS,
    batch_size=N,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=75),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=550, restore_best_weights=True),
        tf.keras.callbacks.TerminateOnNaN()
    ]
)

K = model._kernel_matrix().numpy()
reg_factor = 1e-3
c0 = model._coefficients().numpy().flatten()

cf_psi = cg(K + np.eye(*K.shape) * reg_factor, y, x0=c0, atol=1e-12)[0]

Knopsi = model._kernel_matrix(with_psi=False).numpy()
cf_nopsi = cg(Knopsi + np.eye(*Knopsi.shape) * reg_factor, y, atol=1e-12)[0]

psi = model._model_psi(Xtest).numpy().flatten()
ypred = model(Xtest).numpy().flatten()

ypred_interpol = model._kernel_eval(Xtest).numpy() @ np.expand_dims(cf_psi, axis=-1)
ypred_interpol = ypred_interpol.flatten()

ynopsi_interpol = model._kernel_eval(Xtest, with_psi=False).numpy() @ np.expand_dims(cf_nopsi, axis=-1)
ynopsi_interpol = ynopsi_interpol.flatten()

# ------------------------------------- ERRORS ----------------------------
# ypred_interpol
ae = np.abs(ytest - ypred_interpol)
mse = ((ytest - ypred_interpol) ** 2).mean()
rmse = np.sqrt(mse)
# ynopsi_interpol (classic interpolation)
ae_nopsi = np.abs(ytest - ynopsi_interpol)
mse_nopsi = ((ytest - ynopsi_interpol) ** 2).mean()
rmse_nopsi = np.sqrt(mse_nopsi)
# ypred = VSKFitter's call otherwise
ae_nointerpol = np.abs(ytest - ypred)
mse_nointerpol = ((ytest - ypred) ** 2).mean()
rmse_nointerpol = np.sqrt(mse_nointerpol)
# psi (only deltaNN)
ae_psionly = np.abs(ytest - psi)
mse_psionly = ((ytest - psi) ** 2).mean()
rmse_psionly = np.sqrt(mse_psionly)
# ------------------------------------------------------------------------------

# -------------------------- PLOTS ---------------------------------------------
fig_vsk = plt.figure(figsize=(10, 10))
ax_vsk = fig_vsk.add_subplot(111, projection="3d")
surf_vsk = ax_vsk.plot_surface(X1test, X2test, ypred.reshape(neval, neval), cmap='viridis')
ax_vsk.set_title('VSK Prediction')
fig_vsk.colorbar(surf_vsk)  # , shrink=0.5, aspect=5)

fig_vsk_top, ax_vsk_top = plt.subplots()
fig_vsk_top.set_figheight(10)
fig_vsk_top.set_figwidth(10)
cont_vsk_top = ax_vsk_top.contourf(X1test, X2test, ypred.reshape(neval, neval), cmap='viridis', levels=100)
ax_vsk_top.set_title('VSK Prediction')
fig_vsk_top.colorbar(cont_vsk_top)  # , shrink=0.5, aspect=5)

fig_vsk_ipol = plt.figure(figsize=(10, 10))
ax_vsk_ipol = fig_vsk_ipol.add_subplot(111, projection="3d")
surf_vsk_ipol = ax_vsk_ipol.plot_surface(X1test, X2test, ypred_interpol.reshape(neval, neval), cmap='viridis')
ax_vsk_ipol.set_title('VSK Interpolation')
fig_vsk_ipol.colorbar(surf_vsk_ipol)  # , shrink=0.5, aspect=5)

fig_vsk_ipol_top, ax_vsk_ipol_top = plt.subplots()
fig_vsk_ipol_top.set_figheight(10)
fig_vsk_ipol_top.set_figwidth(10)
cont_vsk_ipol_top = ax_vsk_ipol_top.contourf(X1test, X2test, ypred_interpol.reshape(neval, neval),
                                             cmap='viridis', levels=100
                                             )
ax_vsk_ipol_top.set_title('VSK Interpolation')
fig_vsk_ipol_top.colorbar(cont_vsk_ipol_top)  # , shrink=0.5, aspect=5)

fig_ipol = plt.figure(figsize=(10, 10))
ax_ipol = fig_ipol.add_subplot(111, projection="3d")
surf_ipol = ax_ipol.plot_surface(X1test, X2test, ynopsi_interpol.reshape(neval, neval), cmap='viridis')
ax_ipol.set_title('Classic Interpolation')
fig_ipol.colorbar(surf_ipol)  # , shrink=0.5, aspect=5)

fig_ipol_top, ax_ipol_top = plt.subplots()
fig_ipol_top.set_figheight(10)
fig_ipol_top.set_figwidth(10)
cont_ipol_top = ax_ipol_top.contourf(X1test, X2test, ynopsi_interpol.reshape(neval, neval), cmap='viridis',
                                     levels=100
                                     )
ax_ipol_top.set_title('Classic Interpolation')
fig_ipol_top.colorbar(cont_ipol_top)  # , shrink=0.5, aspect=5)

fig_original = plt.figure(figsize=(10, 10))
ax_original = fig_original.add_subplot(111, projection="3d")
surf_original = ax_original.plot_surface(X1test, X2test, ytest.reshape(neval, neval), cmap='viridis')
ax_original.set_title('Target Function')
fig_original.colorbar(surf_original)  # , shrink=0.5, aspect=5)

fig_original_top, ax_original_top = plt.subplots()
fig_original_top.set_figheight(10)
fig_original_top.set_figwidth(10)
cont_original_top = ax_original_top.contourf(X1test, X2test, ytest.reshape(neval, neval), cmap='viridis',
                                             levels=100
                                             )
ax_original_top.set_title('Target Function')
fig_original_top.colorbar(cont_original_top)  # , shrink=0.5, aspect=5)

fig_psi = plt.figure(figsize=(10, 10))
ax_psi = fig_psi.add_subplot(111, projection="3d")
surf_psi = ax_psi.plot_surface(X1test, X2test, psi.reshape(neval, neval), cmap='viridis')
ax_psi.set_title('Psi Function')
fig_psi.colorbar(surf_psi)  # , shrink=0.5, aspect=5)

fig_psi_top, ax_psi_top = plt.subplots()
fig_psi_top.set_figheight(10)
fig_psi_top.set_figwidth(10)
cont_psi_top = ax_psi_top.contourf(X1test, X2test, psi.reshape(neval, neval), cmap='viridis', levels=100)
ax_psi_top.set_title('Psi Function')
fig_psi_top.colorbar(cont_psi_top)  # , shrink=0.5, aspect=5)

fig_ae = plt.figure(figsize=(10, 10))
ax_ae = fig_ae.add_subplot(111, projection="3d")
surf_ae = ax_ae.plot_surface(X1test, X2test, ae.reshape(neval, neval), cmap='Reds')
ax_ae.set_title('Absolute Error (VSK Interpol.)')
fig_ae.colorbar(surf_ae)  # , shrink=0.5, aspect=5)

fig_ae_top, ax_ae_top = plt.subplots()
fig_ae_top.set_figheight(10)
fig_ae_top.set_figwidth(10)
cont_ae_top = ax_ae_top.contourf(X1test, X2test, ae.reshape(neval, neval), cmap='Reds', levels=100)
ax_ae_top.set_title('Absolute Error (VSK Interpol.)')
fig_ae_top.colorbar(cont_ae_top)  # , shrink=0.5, aspect=5)

fig_ae_nopsi = plt.figure(figsize=(10, 10))
ax_ae_nopsi = fig_ae_nopsi.add_subplot(111, projection="3d")
surf_ae_nopsi = ax_ae_nopsi.plot_surface(X1test, X2test, ae_nopsi.reshape(neval, neval), cmap='Reds')
ax_ae_nopsi.set_title('Absolute Error (Classic Interpol.)')
fig_ae_nopsi.colorbar(surf_ae_nopsi)  # , shrink=0.5, aspect=5)

fig_ae_top_nopsi, ax_ae_top_nopsi = plt.subplots()
fig_ae_top_nopsi.set_figheight(10)
fig_ae_top_nopsi.set_figwidth(10)
cont_ae_top_nopsi = ax_ae_top_nopsi.contourf(X1test, X2test, ae_nopsi.reshape(neval, neval), cmap='Reds', levels=100)
ax_ae_top_nopsi.set_title('Absolute Error (Classic Interpol.)')
fig_ae_top_nopsi.colorbar(cont_ae_top_nopsi)  # , shrink=0.5, aspect=5)

fig_ae_psionly = plt.figure(figsize=(10, 10))
ax_ae_psionly = fig_ae_psionly.add_subplot(111, projection="3d")
surf_ae_psionly = ax_ae_psionly.plot_surface(X1test, X2test, ae_psionly.reshape(neval, neval), cmap='Reds')
ax_ae_psionly.set_title('Absolute Error (Psi Function)')
fig_ae_psionly.colorbar(surf_ae_psionly)  # , shrink=0.5, aspect=5)

fig_ae_top_psionly, ax_ae_top_psionly = plt.subplots()
fig_ae_top_psionly.set_figheight(10)
fig_ae_top_psionly.set_figwidth(10)
cont_ae_top_psionly = ax_ae_top_psionly.contourf(X1test, X2test, ae_psionly.reshape(neval, neval), cmap='Reds',
                                                 levels=100)
ax_ae_top_psionly.set_title('Absolute Error (Psi Function)')
fig_ae_top_psionly.colorbar(cont_ae_top_psionly)  # , shrink=0.5, aspect=5)

if save_pics:
    fig_vsk.savefig(f'{saving_folder}/vsk_pred.pdf')
    fig_vsk_top.savefig(f'{saving_folder}/vsk_pred_topview.pdf')
    fig_vsk_ipol.savefig(f'{saving_folder}/vsk_interpolation.pdf')
    fig_vsk_ipol_top.savefig(f'{saving_folder}/vsk_interpolation_topview.pdf')
    fig_ipol.savefig(f'{saving_folder}/classic_interpolation.pdf')
    fig_ipol_top.savefig(f'{saving_folder}/classic_interpolation_topview.pdf')
    fig_original.savefig(f'{saving_folder}/target_func.pdf')
    fig_original_top.savefig(f'{saving_folder}/target_func_topview.pdf')
    fig_psi.savefig(f'{saving_folder}/psi_func.pdf')
    fig_psi_top.savefig(f'{saving_folder}/psi_func_topview.pdf')
    fig_ae.savefig(f'{saving_folder}/abserr.pdf')
    fig_ae_top.savefig(f'{saving_folder}/abserr_topview.pdf')
    fig_ae_nopsi.savefig(f'{saving_folder}/abserr_nopsi.pdf')
    fig_ae_top_nopsi.savefig(f'{saving_folder}/abserr_nopsi_topview.pdf')
    fig_ae_psionly.savefig(f'{saving_folder}/abserr_psionly.pdf')
    fig_ae_top_psionly.savefig(f'{saving_folder}/abserr_psionly_topview.pdf')

else:
    plt.show()





