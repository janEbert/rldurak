import sys

import keras.backend as K
from keras.layers import Input, Dense
from keras.layers.merge import add
from keras.models import Model
from keras.optimizers import Adam, RMSprop
import tensorflow as tf

if sys.version_info[0] == 2:
    range = xrange


class Critic:
    """A critic model evaluating an action in a given state."""

    def __init__(
            self, sess, state_shape, action_shape, load=True, optimizer='adam',
            alpha=0.001, epsilon=1e-8, tau=0.001, neurons_per_layer=[100, 50]):
        """Initialize a critic with the given session, learning rate,
        update factor and neurons in the hidden layers.

        If load is true, load the model instead of creating a new one.
        """
        self.sess = sess
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.optimizer_choice = optimizer.lower()
        self.alpha = alpha
        self.tau = tau
        if len(neurons_per_layer) < 2:
            if not neurons_per_layer:
                self.neurons_per_layer = [100, 50]
            else:
                self.neurons_per_layer.append(50)
            print('Neurons per layer for the critic have been adjusted')
        else:
            self.neurons_per_layer = neurons_per_layer
        K.set_session(sess)
        self.model, self.state_input, self.action_input = self.create_model(
                epsilon)
        self.target_model = self.create_model(epsilon)[0]
        self.action_gradients = K.gradients(self.model.output,
                self.action_input)
        self.sess.run(tf.global_variables_initializer())
        if load:
            self.load_weights()
        self.model._make_predict_function()
        self.target_model._make_predict_function()

    def create_model(self, epsilon):
        """Return a compiled model and the state and action input
        layers with the given epsilon for numerical stability.
        """
        inputs = Input(shape=(self.state_shape,))
        action_input = Input(shape=(self.action_shape,))
        x1 = Dense(self.neurons_per_layer[0], activation='relu')(inputs)
        x1 = Dense(self.neurons_per_layer[1], activation='relu')(x1)
        x2 = Dense(self.neurons_per_layer[1], activation='relu')(action_input)
        x = add([x1, x2])
        for n in self.neurons_per_layer[2:]:
            x = Dense(n, activation='relu')(x)
        outputs = Dense(self.action_shape)(x)

        model = Model(inputs=[inputs, action_input], outputs=outputs)

        assert self.optimizer_choice in ['adam', 'rmsprop']
        if self.optimizer_choice == 'adam':
            opti = Adam(lr=self.alpha, epsilon=epsilon)
        else:
            opti = RMSprop(lr=self.alpha, epsilon=epsilon)
        model.compile(optimizer=opti, loss='mse')
        return model, inputs, action_input

    def get_gradients(self, states, actions):
        """Return the gradients for the given states and actions."""
        return self.sess.run(self.action_gradients, feed_dict={
                self.state_input: states, self.action_input: actions})[0]

    def train_target(self):
        """Train the target model."""
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = (self.tau * weights[i]
                    + (1 - self.tau) * target_weights[i])
        self.target_model.set_weights(target_weights)

    def save_weights(self, file_name=None):
        """Save the current model's weights."""
        if file_name is None:
            file_name = ('critic-' + self.optimizer_choice + '-'
                    + str(self.state_shape) + '-features.h5')
        self.model.save_weights(file_name)

    def load_weights(self, file_name=None):
        """Load the saved weights for the model and target model."""
        if file_name is None:
            file_name = ('critic-' + self.optimizer_choice + '-'
                    + str(self.state_shape) + '-features.h5')
        try:
            self.model.load_weights(file_name)
            self.target_model.load_weights(file_name)
        except OSError:
            print('Critic weights could not be found. No data was loaded!')