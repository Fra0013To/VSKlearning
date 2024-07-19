import pickle
import os
import tensorflow as tf
import numpy as np
from custom_tf.rbf_archetypes import GaussianRBF


class VSKFitter(tf.keras.models.Model):
    def __init__(self,
                 model_psi,
                 rbf_centers_coo,  # num_centers-by-domain_dim
                 rbf=GaussianRBF(),
                 shape_param=None
                 ):
        super().__init__()

        num_rbf_centers, domain_dim = rbf_centers_coo.shape

        self._rbf = rbf

        # LIST OF ATTRIBUTES:
        self._domain_dim = domain_dim
        self._num_rbf_centers = num_rbf_centers

        self._rbf_centers_coo = rbf_centers_coo

        self._tf_dtype = self._rbf._tf_dtype
        check_dtypes = (
            self._tf_dtype != model_psi.output.dtype
        )
        if np.any(check_dtypes):
            raise ValueError(
                f"""
                The model outputs must have same dtype ({self._tf_dtype}) of the rbf!
                They are instead: {model_psi.output.dtype}.
                """
            )

        self._model_psi = model_psi

        self._rbf_centers_coo_tf = tf.cast(rbf_centers_coo, dtype=self._tf_dtype)

        if shape_param is None:
            self._shape_param = np.ones((1, self._num_rbf_centers))
        else:
            if shape_param.ndim == 1.:
                self._shape_param = np.expand_dims(shape_param, axis=0)
            elif shape_param.ndim == 2 and shape_param.shape[-1] == 1:
                self._shape_param = np.transpose(shape_param)
            else:
                self._shape_param = shape_param

        self.__coefficients_layer = tf.keras.layers.Dense(self._num_rbf_centers,
                                                          activation='linear',
                                                          # kernel_initializer='ones',
                                                          # bias_initializer='zeros',
                                                          name='coeff_layer',
                                                          dtype=self._tf_dtype
                                                          )

        self.eps = tf.convert_to_tensor(1e-10, dtype=tf.float64)

    def _coefficients(self):

        return self.__coefficients_layer(tf.ones((1, 1)))

    def _psi_extension(self, inputs):
        inputs = tf.cast(inputs, dtype=self._tf_dtype)
        inputs_ext = tf.concat(
            [
                inputs,
                self._model_psi(inputs)
            ],
            axis=-1
        )

        return inputs_ext

    def __rbf_centers_coo_ext(self):

        return self._psi_extension(self._rbf_centers_coo_tf)

    def _kernel_eval(self, inputs, with_psi=True):
        if with_psi:
            inputs_ext = self._psi_extension(inputs)

            rbf_centers_coo_ext = self.__rbf_centers_coo_ext()

            rbfs, _ = self._rbf.evaluate(centers=rbf_centers_coo_ext,
                                         inputs=inputs_ext,
                                         radii=self._shape_param
                                         )
        else:
            rbfs, _ = self._rbf.evaluate(centers=self._rbf_centers_coo,
                                         inputs=inputs,
                                         radii=self._shape_param
                                         )

        return rbfs

    def call(self, inputs, training=None, mask=None):

        rbfs = self._kernel_eval(inputs)

        out_tensor = tf.reduce_sum(rbfs * self._coefficients(), axis=-1)

        return out_tensor

    def _kernel_matrix(self, with_psi=True):
        if with_psi:
            rbf_centers_coo_ext = self.__rbf_centers_coo_ext()
            rbfs, _ = self._rbf.evaluate(centers=rbf_centers_coo_ext,
                                         inputs=rbf_centers_coo_ext,
                                         radii=self._shape_param
                                         )
        else:
            rbfs, _ = self._rbf.evaluate(centers=self._rbf_centers_coo,
                                         inputs=self._rbf_centers_coo,
                                         radii=self._shape_param
                                         )

        return rbfs

    def _export(self, folder_path, model_name):
        if not os.path.exists(f'{folder_path}/{model_name}'):
            os.makedirs(f'{folder_path}/{model_name}')

        self._model_psi.save(f'{folder_path}/{model_name}/model_psi.keras')

        coeff_weights =self.__coefficients_layer.get_weights()
        with open(f'{folder_path}/{model_name}/model_coefflayer_weights.pkl', 'wb') as file:
            pickle.dump(coeff_weights, file)

        with open(f'{folder_path}/{model_name}/rbf_centers_coo.pkl', 'wb') as file:
            pickle.dump(self._rbf_centers_coo, file)

        with open(f'{folder_path}/{model_name}/rbf.pkl', 'wb') as file:
            pickle.dump(self._rbf, file)

        with open(f'{folder_path}/{model_name}/shape_param.pkl', 'wb') as file:
            pickle.dump(self._shape_param, file)


def load_vskmodel(path, custom_objects=None):
    if custom_objects is None:
        custom_objects = {}

    model_psi = tf.keras.models.load_model(f'{path}/model_psi.keras', custom_objects=custom_objects)

    with open(f'{path}/model_coefflayer_weights.pkl', 'rb') as file:
        coeff_weights = pickle.load(file)

    with open(f'{path}/rbf_centers_coo.pkl', 'rb') as file:
        rbf_centers_coo = pickle.load(file)

    with open(f'{path}/rbf.pkl', 'rb') as file:
        rbf = pickle.load(file)

    with open(f'{path}/shape_param.pkl', 'rb') as file:
        shape_param = pickle.load(file)

    model = VSKFitter(model_psi=model_psi, rbf_centers_coo=rbf_centers_coo, rbf=rbf, shape_param=shape_param)

    _ = model._coefficients()
    model._VSKFitter__coefficients_layer.set_weights(coeff_weights)

    return model

