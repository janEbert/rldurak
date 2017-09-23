import sys

import keras.backend as K
from keras.layers import Input, Dense
from keras.models import Model
import tensorflow as tf

if sys.version_info[0] == 2:
    range = xrange


class Actor:
    """An actor model selecting an action for a given state."""

    def __init__(
            self, sess, state_shape, action_shape, load=True, optimizer='adam',
            alpha=0.001, epsilon=1e-8, tau=0.001, n1=100, n2=150):
        """Construct an actor with the given session, learning rate,
        update factor and neurons in the hidden layers.

        If load is true, load the model instead of creating a new one.
        """
        self.sess = sess
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.alpha = alpha
        self.tau = tau
        self.n1 = n1
        self.n2 = n2
        K.set_session(sess)
        self.model, self.inputs, weights = self.create_model()
        self.target_model = self.create_model()[0]
        self.action_gradients = tf.placeholder(tf.float32,
                [None, self.action_shape])
        parameter_gradients = tf.gradients(self.model.output, weights,
                -self.action_gradients)
        gradients = zip(parameter_gradients, weights)
        optimizer = optimizer.lower()
        assert optimizer in ['adam', 'rmsprop']
        if optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(
                    self.alpha, epsilon=epsilon).apply_gradients(gradients)
        else:
            self.optimizer = tf.train.RMSPropOptimizer(
                    self.alpha, epsilon=epsilon).apply_gradients(gradients)
        self.sess.run(tf.global_variables_initializer())
        if load:
            if self.load_weights():
                self.model._make_predict_function()
                self.target_model._make_predict_function()

    def create_model(self):
        """Return a compiled model."""
        inputs = Input(shape=(self.state_shape,))
        x = Dense(self.n1, activation='relu')(inputs)
        x = Dense(self.n2, activation='relu')(x)
        outputs = Dense(self.action_shape)(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model, inputs, model.trainable_weights

    def train(self, states, gradients):
        """Train the model on the given states and the given gradients
        provided by the critic.
        """
        self.sess.run(self.optimizer, feed_dict={self.inputs: states,
                self.action_gradients: gradients})

    def train_target(self):
        """Train the target model."""
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] \
                    + (1 - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)

    def save_weights(self, file_name=None):
        """Save the current model's weights."""
        if file_name is None:
            file_name = 'actor-' + str(self.state_shape) + '-features.h5'
        self.model.save_weights(file_name)

    def load_weights(self, file_name=None):
        """Load the saved weights for the model and target model."""
        if file_name is None:
            file_name = 'actor-' + str(self.state_shape) + '-features.h5'
        try:
            self.model.load_weights(file_name)
            self.target_model.load_weights(file_name)
        except OSError:
            print('Actor weights could not be found. No data was loaded!')
            return False
        return True