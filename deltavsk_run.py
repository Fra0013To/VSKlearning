import tensorflow as tf
from custom_tf.model_archetypes import VSKFitter, load_vskmodel
from custom_tf.layers import DiscontinuityDense, DenseResidualLayer
from custom_tf.psimodels_archetypes import psimodel_generator, psimodels_allowed
from custom_tf.optimizers_archetypes import optimizers_dict
from custom_tf.rbf_archetypes import rbfs_dict, GaussianRBF, MaternC2RBF
from test_disc_functions.basic_disc_functions import discfuncs_dict
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse.linalg import cg
from scipy.stats import qmc
import time
import argparse
import os
import pickle

import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt

default_data_path = 'data/experiment_results'

timenow = time.localtime()
ID = time.strftime('%Y%m%d_%H%M%S', timenow)
rs = int(time.strftime('%Y%m%d', timenow)) + int(time.strftime('%H%M%S', timenow))

custom_objects = {
    'VSKFitter': VSKFitter,
    'DiscontinuityDense': DiscontinuityDense,
    'DenseResidualLayer': DenseResidualLayer,
    'GaussianRBF': GaussianRBF,
    'MaternC2RBF': MaternC2RBF
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-df', '--discontinuous_function', type=str, default='dfunc_basic_004',
                        help=f'List of discontinuous functions allowed: {list(discfuncs_dict.keys())}'
                        )
    parser.add_argument('-rbf', '--radialbasisfunction', type=str, default='matern_c2',
                        help=f'List of RBFs allowed: {list(rbfs_dict.keys())}'
                        )
    parser.add_argument('-sp', '--rbf_shape_parameter', type=float, default=1.,
                        help=f'Default value: 1.')  # TODO: extend to arrays
    parser.add_argument('-pm', '--psi_model', type=str, default='deltaresnet_000',
                        help=f'List of psi_models allowed: {psimodels_allowed}'
                        )
    parser.add_argument('-o', '--optimizer', type=str, default='adam_lr1e-4',
                        help=f'List of optimizers allowed: {list(optimizers_dict.keys())}'
                        )
    parser.add_argument('-e', '--training_epochs', type=int, default=5000,
                        help='Default value: 5000')
    parser.add_argument('-epo', '--training_epochs_psimodel_only', type=int, default=2500,
                        help='Default value: 2500')
    parser.add_argument('-bpo', '--minibatch_size_psimodel_only', type=int, default=32,
                        help='Default value: 32')
    parser.add_argument('-N', '--num_centers', type=int, default=1000,
                        help='Default value: 1000')
    parser.add_argument('-dN', '--distribution_num_centers', type=str, default='halton',
                        help='Default value: halton. Alternatively: uniform.')
    parser.add_argument('-ntg', '--num_test_grid', type=int, default=60,
                        help='Default value: 60')
    parser.add_argument('-lrf', '--lr_factor', type=float, default=0.5,
                        help='Default value: 0.5')
    parser.add_argument('-lrp', '--lr_patience', type=float, default=75,
                        help='Default value: 75')
    parser.add_argument('-esp', '--es_patience', type=float, default=550,
                        help='Default value: 550')
    parser.add_argument('-regf', '--regularization_factor', type=float, default=1e-3,
                        help='Default value: 1e-3')
    parser.add_argument('-cgt', '--cg_tolerance', type=float, default=1e-12,
                        help='Default value: 1e-12')
    parser.add_argument('-rs', '--random_seed', type=int, default=rs,
                        help='Default value: generated as (YYYYMMDD + HHmmss) if the program is run at HH:mm:ss of the day DD/MM/YYYY.')
    parser.add_argument('-ponc', '--psimodel_only_nocoeffs', action='store_true',
                        help='Default value: False (i.e., True if the option is given)')
    parser.add_argument('-tp', '--trainperc_psimodel_only', type=float, default=.8,
                        help='Default value: 0.8')
    parser.add_argument('-nndf', '--notnormalized_discontinuous_function', action='store_true',
                        help='Default value: False (i.e., True if the option is given)')
    parser.add_argument('-s', '--save_results', action='store_true',
                        help='Default value: False (i.e., True if the option is given)')
    parser.add_argument('-spl', '--save_plots', action='store_true',
                        help='Default value: False (i.e., True if the option is given)')
    parser.add_argument('-shpl', '--show_plots', action='store_true',
                        help='Default value: False (i.e., True if the option is given)')
    parser.add_argument('-sf', '--saving_folder', type=str, default=default_data_path,
                        help=f'Default value: {default_data_path}')
    parser.add_argument('-lpm', '--load_psimodel', type=str, default=None,
                        help=f'No Default Value. You must specify the ID of a pretrained model, saved in the folder given by the option --loading_folder_psimodel (-lfpm). Better if you use also the same random seed')
    parser.add_argument('-lfpm', '--loading_folder_psimodel', type=str, default=default_data_path,
                        help=f'Default value: {default_data_path}')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()

    random_seed = args.random_seed

    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    tf.keras.utils.set_random_seed(
        random_seed
    )
    tf.config.experimental.enable_op_determinism()

    N = args.num_centers
    neval = args.num_test_grid

    if args.distribution_num_centers == 'halton':
        sampler = qmc.Halton(d=2, scramble=False)
        H1 = sampler.random(N + 1)
        X = H1[1:(N + 1), :]
    elif args.distribution_num_centers == 'uniform':
        X = np.random.rand(N, 2)
    else:  # same distribution of 'halton'
        sampler = qmc.Halton(d=2, scramble=False)
        H1 = sampler.random(N + 1)
        X = H1[1:(N + 1), :]

    # GRID FOR TEST DATA
    X1test, X2test = np.meshgrid(np.linspace(0, 1, neval), np.linspace(0, 1, neval))
    Xtest = np.hstack([X1test.flatten().reshape(X1test.size, 1), X2test.flatten().reshape(X2test.size, 1)])

    # TEST FUNCTION
    f = discfuncs_dict[args.discontinuous_function]

    # OUTPUTS
    y = f(X)
    ytest = f(Xtest)

    if not args.notnormalized_discontinuous_function:
        yscaler = MinMaxScaler()
        yscaler.fit(y.reshape(-1, 1))

        y_original = y.copy()
        ytest_original = ytest.copy()

        y = yscaler.transform(y.reshape(-1, 1)).flatten()
        ytest = yscaler.transform(ytest.reshape(-1, 1)).flatten()

    # PSIMODEL
    if args.load_psimodel is not None and args.psimodel_only_nocoeffs:
        model_psi = tf.keras.models.load_model(
            f'{args.loading_folder_psimodel}/{args.load_psimodel}/vskmodel/model_psi.keras',
            custom_objects=custom_objects
        )
        with open(f'{args.loading_folder_psimodel}/{args.load_psimodel}/training_history.pkl', 'rb') as file:
            history = pickle.load(file)
    else:
        model_psi = psimodel_generator(args.psi_model)
        model_psi.build(X.shape)
    # print(model_psi.get_weights())

    # RBFs
    rbf_class = rbfs_dict[args.radialbasisfunction]
    shape_param = args.rbf_shape_parameter

    # OPTIMIZER
    OPTIMIZER = optimizers_dict[args.optimizer]

    # TRAINING PARAMETERS
    if args.psimodel_only_nocoeffs:
        EPOCHS = args.training_epochs_psimodel_only
    else:
        EPOCHS = args.training_epochs

    if args.psimodel_only_nocoeffs:
        n_train = int(np.round(args.trainperc_psimodel_only * N))
        inds = np.random.permutation(N)
        inds_train = inds[:n_train]
        inds_val = inds[n_train:]

        Xtrain = X[inds_train, :]
        Xval = X[inds_val, :]
        ytrain = y[inds_train]
        yval = y[inds_val]

        if args.load_psimodel is None:
            model_psi.compile(optimizer=OPTIMIZER, loss='mse', metrics=['mae'])

            history = model_psi.fit(
                Xtrain, ytrain,
                epochs=EPOCHS,
                batch_size=args.minibatch_size_psimodel_only,
                validation_data=(Xval, yval),
                callbacks=[
                    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=args.lr_factor,
                                                         patience=args.lr_patience),
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.es_patience,
                                                     restore_best_weights=True
                                                     ),
                    tf.keras.callbacks.TerminateOnNaN()
                ]
            )

        model = VSKFitter(
            model_psi=model_psi,
            rbf_centers_coo=X,
            shape_param=shape_param * np.ones(N),
            rbf=rbf_class()
        )

    else:
        # VSKFitter
        model = VSKFitter(
            model_psi=model_psi,
            rbf_centers_coo=X,
            shape_param=shape_param * np.ones(N),
            rbf=rbf_class()
        )

        model.compile(optimizer=OPTIMIZER, loss='mse', metrics=['mae'])
        history = model.fit(
            X, y,
            epochs=EPOCHS,
            batch_size=N,
            callbacks=[
                tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=args.lr_factor, patience=args.lr_patience),
                tf.keras.callbacks.EarlyStopping(monitor='loss', patience=args.es_patience, restore_best_weights=True),
                tf.keras.callbacks.TerminateOnNaN()
            ]
        )

    if args.load_psimodel is not None:
        tot_epochs = len(history['loss'])
    else:
        tot_epochs = len(history.history['loss'])

    K = model._kernel_matrix().numpy()
    reg_factor = args.regularization_factor
    if args.psimodel_only_nocoeffs:
        # WE SET c0 TO ZERO BECAUSE WE DO NOT COMPUTE COEFFICIENTS "PAIRED" WTH THE PSI-MODEL TRAINING
        c0 = np.zeros_like(model._coefficients().numpy().flatten())
    else:
        c0 = model._coefficients().numpy().flatten()

    cf_psi = cg(K + np.eye(*K.shape) * reg_factor, y, x0=c0, atol=args.cg_tolerance)[0]

    Knopsi = model._kernel_matrix(with_psi=False).numpy()
    cf_nopsi = cg(Knopsi + np.eye(*Knopsi.shape) * reg_factor, y, atol=args.cg_tolerance)[0]

    psi = model._model_psi(Xtest).numpy().flatten()
    if args.psimodel_only_nocoeffs:
        ypred = psi
    else:
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
    # ynopsi_interpol (calssic interpolation)
    ae_nopsi = np.abs(ytest - ynopsi_interpol)
    ae_nopsi_mean = ae_nopsi.mean()
    ae_nopsi_std = ae_nopsi.std()
    ae_nopsi_min = ae_nopsi.min()
    ae_nopsi_max = ae_nopsi.max()

    mse_nopsi = ((ytest - ynopsi_interpol) ** 2).mean()
    rmse_nopsi = np.sqrt(mse_nopsi)
    # ypred
    if not args.psimodel_only_nocoeffs:  # ypred = psi  if "-ponc" option, VSKFitter's call otherwise
        ae_nointerpol = np.abs(ytest - ypred)
        ae_nointerpol_mean = ae_nointerpol.mean()
        ae_nointerpol_std = ae_nointerpol.std()
        ae_nointerpol_min = ae_nointerpol.min()
        ae_nointerpol_max = ae_nointerpol.max()
        del ae_nointerpol
        mse_nointerpol = ((ytest - ypred) ** 2).mean()
        rmse_nointerpol = np.sqrt(mse_nointerpol)
    # psi (only deltaNN)
    ae_psionly = np.abs(ytest - psi)
    ae_psionly_mean = ae_psionly.mean()
    ae_psionly_std = ae_psionly.std()
    ae_psionly_min = ae_psionly.min()
    ae_psionly_max = ae_psionly.max()

    mse_psionly = ((ytest - psi) ** 2).mean()
    rmse_psionly = np.sqrt(mse_psionly)

    # ----------------------------------------------------------------------------
    # ----------------------------------- SSIM -----------------------------------
    # ------- ypred_interpol -------
    maxy = np.max([ytest.max(), ypred_interpol.max()])
    miny = np.min([ytest.min(), ypred_interpol.min()])
    rangey = maxy - miny

    ytest_as_img = ((ytest - miny) / rangey).reshape((neval, neval))
    ytest_as_img_ssim = np.expand_dims(ytest_as_img, axis=0)
    ytest_as_img_ssim = tf.cast(np.expand_dims(ytest_as_img_ssim, axis=-1), dtype=model._tf_dtype)

    ypred_interpol_as_img = ((ypred_interpol - miny) / rangey).reshape((neval, neval))
    ypred_interpol_as_img_ssim = np.expand_dims(ypred_interpol_as_img, axis=0)
    ypred_interpol_as_img_ssim = tf.cast(np.expand_dims(ypred_interpol_as_img_ssim, axis=-1), dtype=model._tf_dtype)

    ssim = tf.image.ssim(ytest_as_img_ssim, ypred_interpol_as_img_ssim, max_val=1.).numpy()[0]
    del ytest_as_img_ssim, ytest_as_img, ypred_interpol_as_img, ypred_interpol_as_img_ssim
    # ------- ynopsi_interpol (calssic interpolation) -------
    maxy = np.max([ytest.max(), ynopsi_interpol.max()])
    miny = np.min([ytest.min(), ynopsi_interpol.min()])
    rangey = maxy - miny

    ytest_as_img = ((ytest - miny) / rangey).reshape((neval, neval))
    ytest_as_img_ssim = np.expand_dims(ytest_as_img, axis=0)
    ytest_as_img_ssim = tf.cast(np.expand_dims(ytest_as_img_ssim, axis=-1), dtype=model._tf_dtype)

    ynopsi_interpol_as_img = ((ynopsi_interpol - miny) / rangey).reshape((neval, neval))
    ynopsi_interpol_as_img_ssim = np.expand_dims(ynopsi_interpol_as_img, axis=0)
    ynopsi_interpol_as_img_ssim = tf.cast(np.expand_dims(ynopsi_interpol_as_img_ssim, axis=-1), dtype=model._tf_dtype)

    ssim_nopsi = tf.image.ssim(ytest_as_img_ssim, ynopsi_interpol_as_img_ssim, max_val=1.).numpy()[0]
    del ytest_as_img_ssim, ytest_as_img, ynopsi_interpol_as_img, ynopsi_interpol_as_img_ssim
    # ------- ypred = psi  if "-ponc" option, VSKFitter's call otherwise
    if not args.psimodel_only_nocoeffs:
        maxy = np.max([ytest.max(), ypred.max()])
        miny = np.min([ytest.min(), ypred.min()])
        rangey = maxy - miny

        ytest_as_img = ((ytest - miny) / rangey).reshape((neval, neval))
        ytest_as_img_ssim = np.expand_dims(ytest_as_img, axis=0)
        ytest_as_img_ssim = tf.cast(np.expand_dims(ytest_as_img_ssim, axis=-1), dtype=model._tf_dtype)

        ynointerpol_as_img = ((ypred - miny) / rangey).reshape((neval, neval))
        ynointerpol_as_img_ssim = np.expand_dims(ynointerpol_as_img, axis=0)
        ynointerpol_as_img_ssim = tf.cast(np.expand_dims(ynointerpol_as_img_ssim, axis=-1), dtype=model._tf_dtype)

        ssim_nointerpol = tf.image.ssim(ytest_as_img_ssim, ynointerpol_as_img_ssim, max_val=1.).numpy()[0]
        del ytest_as_img_ssim, ytest_as_img, ynointerpol_as_img, ynointerpol_as_img_ssim
    # ------- psi (only deltaNN) -------
    maxy = np.max([ytest.max(), psi.max()])
    miny = np.min([ytest.min(), psi.min()])
    rangey = maxy - miny

    ytest_as_img = ((ytest - miny) / rangey).reshape((neval, neval))
    ytest_as_img_ssim = np.expand_dims(ytest_as_img, axis=0)
    ytest_as_img_ssim = tf.cast(np.expand_dims(ytest_as_img_ssim, axis=-1), dtype=model._tf_dtype)

    psi_as_img = ((psi - miny) / rangey).reshape((neval, neval))
    psi_as_img_ssim = np.expand_dims(psi_as_img, axis=0)
    psi_as_img_ssim = tf.cast(np.expand_dims(psi_as_img_ssim, axis=-1),
                                              dtype=model._tf_dtype)

    ssim_psionly = tf.image.ssim(ytest_as_img_ssim, psi_as_img_ssim, max_val=1.).numpy()[0]
    del ytest_as_img_ssim, ytest_as_img, psi_as_img, psi_as_img_ssim

    if args.save_results:
        if not os.path.exists(f'{args.saving_folder}/experiments_deltavsk.csv'):
            with open(f'{args.saving_folder}/experiments_deltavsk.csv', 'w') as file:
                print(
                    'ID, predictor, rs, disc_func, psi_model, rbf, shape_param, pts_dist, ' +
                    'mae, std_ae, max_ae, min_ae, ' +
                    'mse, rmse, ssim, ' +
                    'optimizer, N, n_eval, max_epochs, tot_epochs, lr_factor, lr_pat, es_pat, psimodel_only, ' +
                    'loaded_psi, batch_psimod_only, trainperc_psimod_only, reg_fac',
                    file=file
                )
        loaded_psi = args.load_psimodel
        if loaded_psi is None:
            loaded_psi = False

        tot_str = ''
        str_interpol = (
            f'{ID}, vsk_interpol, {random_seed}, {args.discontinuous_function}, {args.psi_model}, {args.radialbasisfunction}, ' +
            f'{shape_param}, {args.distribution_num_centers}, {ae.mean()}, {ae.std()}, {ae.max()}, {ae.min()}, {mse}, {rmse}, ' +
            f'{ssim}, {args.optimizer}, {N}, {neval}, ' +
            f'{EPOCHS}, {tot_epochs}, {args.lr_factor}, {args.lr_patience}, {args.es_patience}, ' +
            f'{args.psimodel_only_nocoeffs}, {loaded_psi}, {args.minibatch_size_psimodel_only}, ' +
            f'{args.trainperc_psimodel_only}, {reg_factor}'
        )
        tot_str = tot_str + str_interpol

        if not args.psimodel_only_nocoeffs:
            tot_str = tot_str + '\n'
            str_nointerpol = (
                f'{ID}, vsk_nointerpol, {random_seed}, {args.discontinuous_function}, {args.psi_model}, {args.radialbasisfunction}, ' +
                f'{shape_param}, {args.distribution_num_centers}, {ae_nointerpol_mean}, {ae_nointerpol_std}, {ae_nointerpol_max}, {ae_nointerpol_min}, {mse_nointerpol}, {rmse_nointerpol}, ' +
                f'{ssim_nointerpol}, {args.optimizer}, {N}, {neval}, ' +
                f'{EPOCHS}, {tot_epochs}, {args.lr_factor}, {args.lr_patience}, {args.es_patience}, ' +
                f'{args.psimodel_only_nocoeffs}, {loaded_psi}, {args.minibatch_size_psimodel_only}, ' +
                f'{args.trainperc_psimodel_only}, {reg_factor}'
            )
            tot_str = tot_str + str_nointerpol

        tot_str = tot_str + '\n'
        str_psi = (
                f'{ID}, psi, {random_seed}, {args.discontinuous_function}, {args.psi_model}, {args.radialbasisfunction}, ' +
                f'{shape_param}, {args.distribution_num_centers}, {ae_psionly_mean}, {ae_psionly_std}, {ae_psionly_max}, {ae_psionly_min}, {mse_psionly}, {rmse_psionly}, ' +
                f'{ssim_psionly}, {args.optimizer}, {N}, {neval}, ' +
                f'{EPOCHS}, {tot_epochs}, {args.lr_factor}, {args.lr_patience}, {args.es_patience}, ' +
                f'{args.psimodel_only_nocoeffs}, {loaded_psi}, {args.minibatch_size_psimodel_only}, ' +
                f'{args.trainperc_psimodel_only}, {reg_factor}'
        )
        tot_str = tot_str + str_psi

        tot_str = tot_str + '\n'
        str_nopsi = (
                f'{ID}, classic_interpol, {random_seed}, {args.discontinuous_function}, {args.psi_model}, {args.radialbasisfunction}, ' +
                f'{shape_param}, {args.distribution_num_centers}, {ae_nopsi_mean}, {ae_nopsi_std}, {ae_nopsi_max}, {ae_nopsi_min}, {mse_nopsi}, {rmse_nopsi}, ' +
                f'{ssim_nopsi}, {args.optimizer}, {N}, {neval}, ' +
                f'{EPOCHS}, {tot_epochs}, {args.lr_factor}, {args.lr_patience}, {args.es_patience}, ' +
                f'{args.psimodel_only_nocoeffs}, {loaded_psi}, {args.minibatch_size_psimodel_only}, ' +
                f'{args.trainperc_psimodel_only}, {reg_factor}'
        )
        tot_str = tot_str + str_nopsi
        with open(f'{args.saving_folder}/experiments_deltavsk.csv', 'a') as file:
            print(tot_str, file=file)

        os.makedirs(f'{args.saving_folder}/{ID}')

        results_dict = {
            'X': X,
            'Xtest': Xtest,
            'y': y,
            'ytest': ytest,
            'ypred': ypred,
            'ypred_interpol': ypred_interpol,
            'ynopsi_interpol': ynopsi_interpol,
        }

        args_dict = vars(args)
        results_dict.update(args_dict)

        with open(f'{args.saving_folder}/{ID}/results.pkl', 'wb') as file:
            pickle.dump(results_dict, file)

        with open(f'{args.saving_folder}/{ID}/training_history.pkl', 'wb') as file:
            if args.load_psimodel is not None:
                pickle.dump(history, file)
            else:
                pickle.dump(history.history, file)

        model._export(f'{args.saving_folder}/{ID}', 'vskmodel')

    if (args.save_plots and args.save_results) or args.show_plots:

        if not args.psimodel_only_nocoeffs:
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
        cont_ae_top_psionly = ax_ae_top_psionly.contourf(X1test, X2test, ae_psionly.reshape(neval, neval), cmap='Reds', levels=100)
        ax_ae_top_psionly.set_title('Absolute Error (Psi Function)')
        fig_ae_top_psionly.colorbar(cont_ae_top_psionly)  # , shrink=0.5, aspect=5)

        if args.save_plots and args.save_results:

            if not args.psimodel_only_nocoeffs:
                fig_vsk.savefig(f'{args.saving_folder}/{ID}/vsk_pred.pdf')
                fig_vsk_top.savefig(f'{args.saving_folder}/{ID}/vsk_pred_topview.pdf')
            fig_vsk_ipol.savefig(f'{args.saving_folder}/{ID}/vsk_interpolation.pdf')
            fig_vsk_ipol_top.savefig(f'{args.saving_folder}/{ID}/vsk_interpolation_topview.pdf')
            fig_ipol.savefig(f'{args.saving_folder}/{ID}/classic_interpolation.pdf')
            fig_ipol_top.savefig(f'{args.saving_folder}/{ID}/classic_interpolation_topview.pdf')
            fig_original.savefig(f'{args.saving_folder}/{ID}/target_func.pdf')
            fig_original_top.savefig(f'{args.saving_folder}/{ID}/target_func_topview.pdf')
            fig_psi.savefig(f'{args.saving_folder}/{ID}/psi_func.pdf')
            fig_psi_top.savefig(f'{args.saving_folder}/{ID}/psi_func_topview.pdf')
            fig_ae.savefig(f'{args.saving_folder}/{ID}/abserr.pdf')
            fig_ae_top.savefig(f'{args.saving_folder}/{ID}/abserr_topview.pdf')
            fig_ae_nopsi.savefig(f'{args.saving_folder}/{ID}/abserr_nopsi.pdf')
            fig_ae_top_nopsi.savefig(f'{args.saving_folder}/{ID}/abserr_nopsi_topview.pdf')
            fig_ae_psionly.savefig(f'{args.saving_folder}/{ID}/abserr_psionly.pdf')
            fig_ae_top_psionly.savefig(f'{args.saving_folder}/{ID}/abserr_psionly_topview.pdf')

        if args.show_plots:
            plt.show()
