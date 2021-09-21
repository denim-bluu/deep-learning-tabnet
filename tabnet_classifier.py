"""
Tensorflow 2.x Implementation of TabNet.

Code Reference:
    - Sparsemax: https://github.com/tensorflow/addons/blob/master/tensorflow_addons/activations/sparsemax.py
    - TabNet Tensorflow 1.0 : https://github.com/google-research/google-research/blob/master/tabnet/tabnet_model.py
"""

from typing import Union

import tensorflow as tf

# Python Tensorflow Implementation
''' Sparsemax code reference: https://github.com/tensorflow/addons/blob/master/tensorflow_addons/activations/sparsemax.py
'''


def register_keras_custom_object(cls):
    tf.keras.utils.get_custom_objects()[cls.__name__] = cls
    return cls


@register_keras_custom_object
@tf.function
def sparsemax(logits, axis):
    logits = tf.convert_to_tensor(logits, name="logits")

    # We need its original shape for shape inference.
    shape = logits.get_shape()
    rank = shape.rank
    is_last_axis = (axis == -1) or (axis == rank - 1)

    if is_last_axis:
        output = _compute_2d_sparsemax(logits)
        output.set_shape(shape)
        return output

    # If dim is not the last dimension, we have to do a transpose so that we can
    # still perform softmax on its last dimension.

    # Swap logits' dimension of dim and its last dimension.
    rank_op = tf.rank(logits)
    axis_norm = axis % rank
    logits = _swap_axis(logits, axis_norm, tf.math.subtract(rank_op, 1))

    # Do the actual softmax on its last dimension.
    output = _compute_2d_sparsemax(logits)
    output = _swap_axis(output, axis_norm, tf.math.subtract(rank_op, 1))

    # Make shape inference work since transpose may erase its static shape.
    output.set_shape(shape)
    return output


def _swap_axis(logits, dim_index, last_index, **kwargs):
    return tf.transpose(
        logits,
        tf.concat(
            [
                tf.range(dim_index),
                [last_index],
                tf.range(dim_index + 1, last_index),
                [dim_index],
            ],
            0,
        ),
        **kwargs,
    )


def _compute_2d_sparsemax(logits):
    """Performs the sparsemax operation when axis=-1."""
    shape_op = tf.shape(logits)
    obs = tf.math.reduce_prod(shape_op[:-1])
    dims = shape_op[-1]

    # In the paper, they call the logits z.
    # The mean(logits) can be substracted from logits to make the algorithm
    # more numerically stable. the instability in this algorithm comes mostly
    # from the z_cumsum. Substacting the mean will cause z_cumsum to be close
    # to zero. However, in practise the numerical instability issues are very
    # minor and substacting the mean causes extra issues with inf and nan
    # input.
    # Reshape to [obs, dims] as it is almost free and means the remanining
    # code doesn't need to worry about the rank.
    z = tf.reshape(logits, [obs, dims])

    # sort z
    z_sorted, _ = tf.nn.top_k(z, k=dims)

    # calculate k(z)
    z_cumsum = tf.math.cumsum(z_sorted, axis=-1)
    k = tf.range(1, tf.cast(dims, logits.dtype) + 1, dtype=logits.dtype)
    z_check = 1 + k * z_sorted > z_cumsum
    # because the z_check vector is always [1,1,...1,0,0,...0] finding the
    # (index + 1) of the last `1` is the same as just summing the number of 1.
    k_z = tf.math.reduce_sum(tf.cast(z_check, tf.int32), axis=-1)

    # calculate tau(z)
    # If there are inf values or all values are -inf, the k_z will be zero,
    # this is mathematically invalid and will also cause the gather_nd to fail.
    # Prevent this issue for now by setting k_z = 1 if k_z = 0, this is then
    # fixed later (see p_safe) by returning p = nan. This results in the same
    # behavior as softmax.
    k_z_safe = tf.math.maximum(k_z, 1)
    indices = tf.stack(
        [tf.range(0, obs), tf.reshape(k_z_safe, [-1]) - 1], axis=1)
    tau_sum = tf.gather_nd(z_cumsum, indices)
    tau_z = (tau_sum - 1) / tf.cast(k_z, logits.dtype)

    # calculate p
    p = tf.math.maximum(tf.cast(0, logits.dtype),
                        z - tf.expand_dims(tau_z, -1))
    # If k_z = 0 or if z = nan, then the input is invalid
    p_safe = tf.where(
        tf.expand_dims(
            tf.math.logical_or(tf.math.equal(k_z, 0),
                               tf.math.is_nan(z_cumsum[:, -1])),
            axis=-1,
        ),
        tf.fill([obs, dims], tf.cast(float("nan"), logits.dtype)),
        p,
    )

    # Reshape back to original size
    p_safe = tf.reshape(p_safe, shape_op)
    return p_safe


def glu(x, n_units=None):
    """Generalized linear unit"""
    if n_units is None:
        n_units = tf.shape(x)[-1] // 2

    return x[:, :n_units] * tf.nn.sigmoid(x[:, n_units:])


class TransformBlock(tf.keras.Model):
    def __init__(self,
                 features: int,
                 momentum: float = 0.7,
                 virtual_batch_size: int = None,
                 block_name='',
                 **kwargs):
        """ Transform block with FC + BN
        This is a block for Feature transformer & Attentive transformer blocks.

        Args:
            features (int): N_a, Embedding feature dimention to use.
            momentum (float, optional): Batch normalization, momentum. Defaults to 0.7.
            virtual_batch_size (int, optional): Ghost Batch Size. Defaults to None.
            block_name (str, optional): Block name. Defaults to ''.
        """
        super(TransformBlock, self).__init__(**kwargs)

        self.features = features
        self.momentum = momentum
        self.virtual_batch_size = virtual_batch_size

        # FC layer
        self.fc = tf.keras.layers.Dense(self.features,
                                        use_bias=False,
                                        name=f'tf_block_fc_{block_name}')
        # BN layer
        self.bn = tf.keras.layers.BatchNormalization(
            axis=-1,
            momentum=momentum,
            virtual_batch_size=virtual_batch_size,
            name=f'tf_block_bn_{block_name}')

    def call(self, inputs, training=None):
        x = self.fc(inputs)
        x = self.bn(x, training=training)
        return x


''' TabNet Code reference: https://github.com/google-research/google-research/blob/master/tabnet/tabnet_model.py
'''


class TabNet(tf.keras.Model):
    def __init__(self,
                 feature_columns: Union[list, tuple],
                 feature_dim: int = 64,
                 output_dim: int = 64,
                 num_features: int = None,
                 num_decision_steps: int = 5,
                 relaxation_factor: float = 1.5,
                 sparsity_coefficient: float = 1e-5,
                 batch_momentum: float = 0.7,
                 virtual_batch_size: int = None,
                 epsilon: float = 1e-5,
                 **kwargs):
        """ TabNet Encoder, Tensorflow 2.x version implementation with minor modification for this project.

        Args:
            feature_columns (Union[list, tuple]): Tensorflow feature columns for the dataset.
            feature_dim (int, optional): Embedding feature dimention to use. Defaults to 64.
            output_dim (int, optional): Output dimension. Defaults to 64.
            num_features (int, optional): Number of features. Defaults to None.
            num_decision_steps (int, optional): Total number of steps. Defaults to 5.
            relaxation_factor (float, optional): >1 will allow features to be used more than once.
                Defaults to 1.5.
            sparsity_coefficient (float, optional): >0 Sparsity regularization in the form of entropy.
                Defaults to 1e-5.
            batch_momentum (float, optional): Batch momentum. Defaults to 0.7.
            virtual_batch_size (int, optional): Ghost Batch size. Defaults to None.
            epsilon (float, optional): Required machine epsilon for prevention of 
                0 divison error in entropy calculation. Defaults to 1e-5.

        """

        super(TabNet, self).__init__(**kwargs)

        # Input checks
        if feature_columns is not None:
            if type(feature_columns) not in (list, tuple):
                raise ValueError(
                    "`feature_columns` must be a list or a tuple.")

            if len(feature_columns) == 0:
                raise ValueError(
                    "`feature_columns` must be contain at least 1 tf.feature_column !"
                )

            if num_features is None:
                num_features = len(feature_columns)
            else:
                num_features = int(num_features)

        else:
            if num_features is None:
                raise ValueError(
                    "If `feature_columns` is None, then `num_features` cannot be None."
                )

        if num_decision_steps < 1:
            raise ValueError("Num decision steps must be greater than 0.")

        feature_dim = int(feature_dim) + int(output_dim)
        output_dim = int(output_dim)
        num_decision_steps = int(num_decision_steps)
        relaxation_factor = float(relaxation_factor)
        sparsity_coefficient = float(sparsity_coefficient)
        batch_momentum = float(batch_momentum)
        epsilon = float(epsilon)

        if relaxation_factor < 1.:
            raise ValueError("`relaxation_factor` cannot be negative !")

        if sparsity_coefficient < 0.:
            raise ValueError("`sparsity_coefficient` cannot be negative !")

        if virtual_batch_size is not None:
            virtual_batch_size = int(virtual_batch_size)

        # Class attributes
        self.feature_columns = feature_columns
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.sparsity_coefficient = sparsity_coefficient
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.epsilon = epsilon

        print("\n REPORT: --------------------------------")
        print(f"Embedding feature dimention: {self.feature_dim}")
        print(f"Output dimension: {self.output_dim}")
        print(f"Total number of steps: {self.num_decision_steps}")
        print(f"Relaxation factor: {self.relaxation_factor}")
        print(f"Sparsity coefficient: {self.sparsity_coefficient}")
        print(f"Ghost batch size: {self.virtual_batch_size}")
        print(f"Momentum in ghost batch normalization: {self.batch_momentum}")
        print("---------------------------------------- \n")

        if num_decision_steps > 1:
            features_for_coeff = feature_dim - output_dim
            print(
                f"[TabNet]: {features_for_coeff} features will be used for decision steps."
            )

        if self.feature_columns is not None:
            self.input_features = tf.keras.layers.DenseFeatures(
                feature_columns, trainable=True)

            self.input_bn = tf.keras.layers.BatchNormalization(
                axis=-1, momentum=batch_momentum, name='input_bn')

        else:
            self.input_features = None
            self.input_bn = None

        # Shared across decision steps layers
        self.transform_f1 = TransformBlock(2 * self.feature_dim,
                                           self.batch_momentum,
                                           self.virtual_batch_size,
                                           block_name='f1')

        self.transform_f2 = TransformBlock(2 * self.feature_dim,
                                           self.batch_momentum,
                                           self.virtual_batch_size,
                                           block_name='f2')

        # Decision step dependent layers
        self.transform_f3_list = [
            TransformBlock(2 * self.feature_dim,
                           self.batch_momentum,
                           self.virtual_batch_size,
                           block_name=f'f3_{i}')
            for i in range(self.num_decision_steps)
        ]

        self.transform_f4_list = [
            TransformBlock(2 * self.feature_dim,
                           self.batch_momentum,
                           self.virtual_batch_size,
                           block_name=f'f4_{i}')
            for i in range(self.num_decision_steps)
        ]

        # Attentive transformer block (# block: Number of decision steps -1)
        self.attentivve_block = [
            TransformBlock(self.num_features,
                           self.batch_momentum,
                           self.virtual_batch_size,
                           block_name=f'coef_{i}')
            for i in range(self.num_decision_steps - 1)
        ]

        self._step_feature_selection_masks = None
        self._step_aggregate_feature_selection_mask = None

    def call(self, inputs, training=None):
        if self.input_features is not None:
            features = self.input_features(inputs)
            features = self.input_bn(features, training=training)

        else:
            features = inputs

        batch_size = tf.shape(features)[0]
        self._step_feature_selection_masks = []
        self._step_aggregate_feature_selection_mask = None

        # Initializes decision-step dependent variables.
        output_aggregated = tf.zeros([batch_size, self.output_dim])
        masked_features = features
        mask_values = tf.zeros([batch_size, self.num_features])
        aggregated_mask_values = tf.zeros([batch_size, self.num_features])
        complementary_aggregated_mask_values = tf.ones(
            [batch_size, self.num_features])

        total_entropy = 0.0
        entropy_loss = 0.

        for ni in range(self.num_decision_steps):
            # Feature transformer with two shared layers
            transform_f1 = self.transform_f1(masked_features,
                                             training=training)
            transform_f1 = glu(transform_f1, self.feature_dim)

            transform_f2 = self.transform_f2(transform_f1, training=training)
            transform_f2 = (glu(transform_f2, self.feature_dim) +
                            transform_f1) * tf.math.sqrt(0.5)

            # Feature transformer with two decision step dependent layers
            transform_f3 = self.transform_f3_list[ni](transform_f2,
                                                      training=training)
            transform_f3 = (glu(transform_f3, self.feature_dim) +
                            transform_f2) * tf.math.sqrt(0.5)

            transform_f4 = self.transform_f4_list[ni](transform_f3,
                                                      training=training)
            transform_f4 = (glu(transform_f4, self.feature_dim) +
                            transform_f3) * tf.math.sqrt(0.5)

            if (ni > 0 or self.num_decision_steps == 1):
                decision_out = tf.nn.relu(transform_f4[:, :self.output_dim])

                # Decision aggregation.
                output_aggregated += decision_out

                # Aggregated masks are used for visualization of the feature importance attributes.
                scale_agg = tf.reduce_sum(decision_out, axis=1, keepdims=True)

                if self.num_decision_steps > 1:
                    scale_agg = scale_agg / tf.cast(
                        self.num_decision_steps - 1, tf.float32)

                aggregated_mask_values += mask_values * scale_agg

            features_for_coef = transform_f4[:, self.output_dim:]

            if ni < (self.num_decision_steps - 1):

                # Attentive transformer
                mask_values = self.attentivve_block[ni](features_for_coef,
                                                        training=training)
                mask_values *= complementary_aggregated_mask_values
                mask_values = sparsemax(mask_values, axis=-1)
                # gamma - mask
                complementary_aggregated_mask_values *= (
                        self.relaxation_factor - mask_values)

                # sparsity regularization in the form of entropy (Grandvalet and Bengio 2004)
                total_entropy += tf.reduce_mean(
                    tf.reduce_sum(
                        -mask_values * tf.math.log(mask_values + self.epsilon),
                        axis=1)) / (tf.cast(self.num_decision_steps - 1,
                                            tf.float32))

                # Add entropy loss
                entropy_loss = total_entropy

                # Feature selection.
                masked_features = tf.multiply(mask_values, features)

                mask_at_step_i = tf.expand_dims(tf.expand_dims(mask_values, 0),
                                                3)
                self._step_feature_selection_masks.append(mask_at_step_i)

            else:
                # This branch is needed for correct compilation by tf.autograph
                entropy_loss = 0.

        # Adds the loss automatically
        self.add_loss(self.sparsity_coefficient * entropy_loss)
        agg_mask = tf.expand_dims(tf.expand_dims(aggregated_mask_values, 0), 3)
        self._step_aggregate_feature_selection_mask = agg_mask

        return output_aggregated

    @property  # Feature mask at each step
    def feature_selection_masks(self):
        return self._step_feature_selection_masks

    @property  # Aggregated feature mask
    def aggregate_feature_selection_mask(self):
        return self._step_aggregate_feature_selection_mask


# TabNet Encoder for classification (regression part omitted)
class TabNetClassifier(tf.keras.Model):
    def __init__(self,
                 feature_columns: int,
                 num_classes: int,
                 num_features: int = None,
                 feature_dim: int = 64,
                 output_dim: int = 64,
                 num_decision_steps: int = 5,
                 relaxation_factor: float = 1.5,
                 sparsity_coefficient: float = 1e-5,
                 batch_momentum: float = 0.7,
                 virtual_batch_size: int = None,
                 epsilon: float = 1e-5,
                 **kwargs):
        super(TabNetClassifier, self).__init__(**kwargs)

        self.num_classes = num_classes

        self.tabnet = TabNet(feature_columns=feature_columns,
                             num_features=num_features,
                             feature_dim=feature_dim,
                             output_dim=output_dim,
                             num_decision_steps=num_decision_steps,
                             relaxation_factor=relaxation_factor,
                             sparsity_coefficient=sparsity_coefficient,
                             batch_momentum=batch_momentum,
                             virtual_batch_size=virtual_batch_size,
                             epsilon=epsilon,
                             **kwargs)

        self.clf = tf.keras.layers.Dense(num_classes,
                                         activation='softmax',
                                         use_bias=False,
                                         name='classifier')

    def call(self, inputs, training=None):
        self.activations = self.tabnet(inputs, training=training)
        out = self.clf(self.activations)

        return out

    def summary(self, *super_args, **super_kwargs):
        super().summary(*super_args, **super_kwargs)
        self.tabnet.summary(*super_args, **super_kwargs)
