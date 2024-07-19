import tensorflow as tf
from abc import ABC, abstractmethod

rbfs_dict = {}


class RadialBasisFunction(ABC):
    def __init__(self, tf_dtype=tf.float64):
        """
        Initialization of the class.
        :param tf_dtype: dtype for TF tensors (cast the inputs with this type in the abstract methods!)
        """
        self._tf_dtype = tf_dtype
    @abstractmethod
    def evaluate(self, centers, inputs, radii):
        """
        Evaluate the RBF in the inputs, w.r.t. centers and radii.
        :param centers: TF-tensor of shape (1, num_centers, domain_dim)
        :param inputs: TF-tensor of shape (num_inputs, domain_dim)
        :param radii: TF-tensor of shape (num_inputs, num_centers) or (1, num_centers)
        :return rbfs: num_inputs-by-num_centers tensor with evaluation of the RBFs w.r.t. each pair (center, input)
        :return cntrs_minus_inputs: optional. num_inputs-by-num_centers-by-domain_dim tensor s.t. element (i, j, k)
            is the k-th component of the vector obtained by the difference between j-th center and i-th input
        """
        pass

    @abstractmethod
    def gradients(self, centers, inputs, radii):
        """
        Evaluate the gradients of the RBFs at the inputs, w.r.t. centers and radii.
        :param centers: TF-tensor of shape (1, num_centers, domain_dim)
        :param inputs: TF-tensor of shape (num_inputs, domain_dim)
        :param radii: TF-tensor of shape (num_inputs, num_centers) or (1, num_centers)
        :param dtype: dtype for TF tensors
        :return grads: num_inputs-by-num_centers-by-domain_dim tensor where el. (i,j,k) is the derivative of
            rbf centered in j-th center at i-th input, with respect to k-th variable
        """
        pass

    @abstractmethod
    def hessian_diags(self, centers, inputs, radii):
        """
        Evaluate the gradients of the RBFs at the inputs, w.r.t. centers and radii.
        :param centers: TF-tensor of shape (1, num_centers, domain_dim)
        :param inputs: TF-tensor of shape (num_inputs, domain_dim)
        :param radii: TF-tensor of shape (num_inputs, num_centers) or (1, num_centers)
        :param dtype: dtype for TF tensors
        :return hessian_diags: num_inputs-by-num_centers-by-domain_dim tensor where el. (i, j, k) is the 2nd-ord.
            derivative of rbf centered in j-th center at i-th input, with respect to k-th variable
        """
        pass

    def laplacians(self, centers, inputs, radii):
        hessian_diags = self.hessian_diags(centers, inputs, radii)
        laplacians = tf.reduce_sum(hessian_diags, axis=-1)

        return laplacians


class GaussianRBF(RadialBasisFunction):
    def __init__(self, tf_dtype=tf.float64):
        """
        Initialization of the class.
        :param tf_dtype: dtype for TF tensors (cast the inputs with this type in the abstract methods!)
        """
        super().__init__(tf_dtype=tf_dtype)

    def evaluate(self, centers, inputs, radii):
        dtype = self._tf_dtype
        centers_ = tf.expand_dims(tf.cast(centers, dtype=dtype), axis=0)
        inputs_ = tf.expand_dims(tf.cast(inputs, dtype=dtype), axis=1)
        radii_ = tf.cast(radii, dtype=dtype)

        # num_inputs-by-num_centers-by-domain_dim tensor
        cntrs_minus_inputs = centers_ - inputs_

        rbfs = tf.math.exp(
            -((radii_ ** 2) *
              tf.reduce_sum(cntrs_minus_inputs ** 2, axis=-1)
              )
        )

        return rbfs, cntrs_minus_inputs

    def gradients(self, centers, inputs, radii):
        rbfs, cntrs_minus_inputs = self.evaluate(centers, inputs, radii)

        grads = 2 * tf.expand_dims(rbfs * (radii ** 2), axis=-1) * cntrs_minus_inputs

        return grads

    def hessian_diags(self, centers, inputs, radii):
        rbfs, cntrs_minus_inputs = self.evaluate(centers, inputs, radii)

        # COMPUTATION BASED ON self.gradients
        # grads = self.gradients(centers, inputs, radii)
        # hessian_diags = 2 * tf.expand_dims((radii ** 2), axis=-1) * cntrs_minus_inputs * grads
        # hessian_diags = hessian_diags + (-2 * tf.expand_dims(rbfs * (radii ** 2), axis=-1))

        # "DIRECT" COMPUTATION
        hessian_diags = (4 * tf.expand_dims((radii ** 4), axis=-1) * (cntrs_minus_inputs ** 2) -
                         2 * tf.expand_dims((radii ** 2), axis=-1)) * tf.expand_dims(rbfs, axis=-1)

        return hessian_diags


rbfs_dict['gaussian'] = GaussianRBF


class MaternC2RBF(RadialBasisFunction):
    def __init__(self, tf_dtype=tf.float64):
        """
        Initialization of the class.
        :param tf_dtype: dtype for TF tensors (cast the inputs with this type in the abstract methods!)
        """
        super().__init__(tf_dtype=tf_dtype)

    def __evaluation_variables(self, centers, inputs, radii):
        dtype = self._tf_dtype
        centers_ = tf.expand_dims(tf.cast(centers, dtype=dtype), axis=0)
        inputs_ = tf.expand_dims(tf.cast(inputs, dtype=dtype), axis=1)
        radii_ = tf.cast(radii, dtype=dtype)

        # num_inputs-by-num_centers-by-domain_dim tensor
        cntrs_minus_inputs = centers_ - inputs_

        cntrs_minus_inputs_norms = tf.math.sqrt(tf.reduce_sum(cntrs_minus_inputs ** 2, axis=-1))

        return centers_, inputs_, radii_, cntrs_minus_inputs, cntrs_minus_inputs_norms

    def evaluate(self, centers, inputs, radii):

        centers_, inputs_, radii_, cntrs_minus_inputs, cntrs_minus_inputs_norms = self.__evaluation_variables(
            centers=centers,
            inputs=inputs,
            radii=radii
        )

        rbfs = (1 + radii_ * cntrs_minus_inputs_norms) * tf.math.exp(
            -(radii_ *
              cntrs_minus_inputs_norms
              )
        )

        return rbfs, cntrs_minus_inputs

    def gradients(self, centers, inputs, radii):
        centers_, inputs_, radii_, cntrs_minus_inputs, cntrs_minus_inputs_norms = self.__evaluation_variables(
            centers=centers,
            inputs=inputs,
            radii=radii
        )

        grads = radii_ * tf.math.exp(
            -(radii_ *
              cntrs_minus_inputs_norms
              )
        )

        grads = -cntrs_minus_inputs * grads

        return grads

    def hessian_diags(self, centers, inputs, radii):
        # TODO (maybe using autodiiff of TensorFlow)

        return None


rbfs_dict['matern_c2'] = MaternC2RBF





