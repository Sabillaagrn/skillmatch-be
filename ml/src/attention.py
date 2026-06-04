"""Custom additive (Bahdanau-style) attention pooling layer.

Reconstructed to match the trained model's saved weights exactly:

    layers/attention_layer/vars/0 -> W, shape (256, 1)   # feature projection
    layers/attention_layer/vars/1 -> b, shape (512, 1)   # per-timestep bias

The layer collapses the BiLSTM output (batch, timesteps=512, features=256)
into a single context vector (batch, 256) by learning an importance score per
timestep, softmax-normalizing over time, and taking the weighted sum.

This module MUST be importable wherever the .keras model is loaded, and the
class MUST be registered under the name "AttentionLayer" (the registered_name
stored in the model config) so Keras can deserialize it.
"""
from __future__ import annotations

import tensorflow as tf
from keras import layers
from keras.saving import register_keras_serializable


@register_keras_serializable(name="AttentionLayer")
class AttentionLayer(layers.Layer):
    """Additive attention pooling over the time dimension.

    score_t = tanh(x_t · W + b_t)            # (batch, T, 1)
    a_t     = softmax(score over t)          # (batch, T, 1)
    context = Σ_t a_t · x_t                   # (batch, features)
    """

    def build(self, input_shape):
        # input_shape = (batch, timesteps, features) = (None, 512, 256)
        timesteps = int(input_shape[1])
        features = int(input_shape[-1])
        # NOTE: weight creation order matters — it must match the saved file:
        #   vars/0 = W (features, 1), vars/1 = b (timesteps, 1)
        self.W = self.add_weight(
            name="W", shape=(features, 1),
            initializer="glorot_uniform", trainable=True,
        )
        self.b = self.add_weight(
            name="b", shape=(timesteps, 1),
            initializer="zeros", trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, T, features)
        score = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)  # (batch, T, 1)
        weights = tf.nn.softmax(score, axis=1)                  # over timesteps
        context = tf.reduce_sum(weights * inputs, axis=1)       # (batch, features)
        return context

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
