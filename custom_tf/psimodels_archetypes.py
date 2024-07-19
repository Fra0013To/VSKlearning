import tensorflow as tf
import custom_tf

psimodels_allowed = [
    'test00_vskfitter',
    'deltaresnet_000'
]


def psimodel_generator(psimodel_name):

    if psimodel_name == 'test00_vskfitter':
        psimodel = tf.keras.models.Sequential(
            layers=[
                tf.keras.layers.Dense(128, activation='swish', kernel_initializer='glorot_normal', dtype=tf.float64),
                tf.keras.layers.Dense(128, activation='swish', kernel_initializer='glorot_normal', dtype=tf.float64),
                custom_tf.layers.DiscontinuityDense(8, activation='swish', kernel_initializer='glorot_normal', dtype=tf.float64),
                tf.keras.layers.Dense(128, activation='swish', kernel_initializer='glorot_normal', dtype=tf.float64),
                tf.keras.layers.Dense(128, activation='swish', kernel_initializer='glorot_normal', dtype=tf.float64),
                custom_tf.layers.DiscontinuityDense(8, activation='swish', kernel_initializer='glorot_normal', dtype=tf.float64),
                tf.keras.layers.Dense(128, activation='swish', kernel_initializer='glorot_normal', dtype=tf.float64),
                tf.keras.layers.Dense(1, activation='linear', kernel_initializer='glorot_normal', dtype=tf.float64)
            ]
        ),
    elif psimodel_name == 'deltaresnet_000':
        psimodel = tf.keras.models.Sequential(
            layers=[
                tf.keras.layers.Dense(128, activation='elu', kernel_initializer='glorot_normal', dtype=tf.float64),
                tf.keras.layers.Dense(128, activation='elu', kernel_initializer='glorot_normal', dtype=tf.float64),
                custom_tf.layers.DenseResidualLayer(128, activation='elu', kernel_initializer='glorot_normal',
                                                    dtype=tf.float64),
                tf.keras.layers.Dense(128, activation='elu', kernel_initializer='glorot_normal', dtype=tf.float64),
                custom_tf.layers.DenseResidualLayer(128, activation='elu', kernel_initializer='glorot_normal',
                                                    dtype=tf.float64),
                custom_tf.layers.DiscontinuityDense(16, activation='elu', kernel_initializer='glorot_normal',
                                                    dtype=tf.float64),
                tf.keras.layers.Dense(128, activation='elu', kernel_initializer='glorot_normal', dtype=tf.float64),
                custom_tf.layers.DenseResidualLayer(128, activation='elu', kernel_initializer='glorot_normal',
                                                    dtype=tf.float64),
                custom_tf.layers.DiscontinuityDense(16, activation='elu', kernel_initializer='glorot_normal',
                                                    dtype=tf.float64),
                tf.keras.layers.Dense(128, activation='elu', kernel_initializer='glorot_normal', dtype=tf.float64),
                custom_tf.layers.DenseResidualLayer(128, activation='elu', kernel_initializer='glorot_normal',
                                                    dtype=tf.float64),
                custom_tf.layers.DiscontinuityDense(16, activation='elu', kernel_initializer='glorot_normal',
                                                    dtype=tf.float64),
                tf.keras.layers.Dense(128, activation='elu', kernel_initializer='glorot_normal', dtype=tf.float64),
                tf.keras.layers.Dense(1, activation='linear', kernel_initializer='glorot_normal', dtype=tf.float64)
            ]
        )
    else:
        psimodel = tf.keras.models.Sequential(
            layers=[
                tf.keras.layers.Dense(128, activation='elu', kernel_initializer='glorot_normal', dtype=tf.float64),
                tf.keras.layers.Dense(128, activation='elu', kernel_initializer='glorot_normal', dtype=tf.float64),
                custom_tf.layers.DenseResidualLayer(128, activation='elu', kernel_initializer='glorot_normal',
                                                    dtype=tf.float64),
                tf.keras.layers.Dense(128, activation='elu', kernel_initializer='glorot_normal', dtype=tf.float64),
                custom_tf.layers.DenseResidualLayer(128, activation='elu', kernel_initializer='glorot_normal',
                                                    dtype=tf.float64),
                custom_tf.layers.DiscontinuityDense(16, activation='elu', kernel_initializer='glorot_normal',
                                                    dtype=tf.float64),
                tf.keras.layers.Dense(128, activation='elu', kernel_initializer='glorot_normal', dtype=tf.float64),
                custom_tf.layers.DenseResidualLayer(128, activation='elu', kernel_initializer='glorot_normal',
                                                    dtype=tf.float64),
                custom_tf.layers.DiscontinuityDense(16, activation='elu', kernel_initializer='glorot_normal',
                                                    dtype=tf.float64),
                tf.keras.layers.Dense(128, activation='elu', kernel_initializer='glorot_normal', dtype=tf.float64),
                custom_tf.layers.DenseResidualLayer(128, activation='elu', kernel_initializer='glorot_normal',
                                                    dtype=tf.float64),
                custom_tf.layers.DiscontinuityDense(16, activation='elu', kernel_initializer='glorot_normal',
                                                    dtype=tf.float64),
                tf.keras.layers.Dense(128, activation='elu', kernel_initializer='glorot_normal', dtype=tf.float64),
                tf.keras.layers.Dense(1, activation='linear', kernel_initializer='glorot_normal', dtype=tf.float64)
            ]
        )

    return psimodel

