import keras.backend as K
from keras.layers import Input, Dense
from keras.layers.merge import add
from keras.models import Model
from keras.optimizers import Adam, RMSprop, SGD
import tensorflow as tf


class Critic:
    """A critic model evaluating an action in a given state."""

    def __init__(self, sess, alpha=0.001, tau=0.001, n1=100, n2=150):
        """Initialize a critic with the given session, learning rate,
        update factor and neurons in the hidden layers.

        If load is true, load the model instead of creating a new one.
        """
        self.sess = sess
        self.alpha = alpha
        self.tau = tau
        self.n1 = n1
        self.n2 = n2
        self.state_shape = 55
        self.action_shape = 5
        K.set_session(sess)
        self.model, self.state_input, self.action_input = self.create_model()
        self.target_model = self.create_model()
        self.action_gradients = K.gradients(self.model.output,
                self.action_input)
        if load:
            self.load_weights()
        self.sess.run(tf.initialize_all_variables())

    def create_model(self):
        """Return a compiled model."""
        inputs = Input(shape=(self.state_shape,))
        action_input = Input(shape=(self.action_shape,))
        x1 = Dense(self.n1, activation='relu')(inputs)
        x1 = Dense(self.n2, activation='relu')(x1)
        x2 = Dense(self.n2, activation='relu')(action_input)
        x = add([x1, x2])
        outputs = Dense(self.action_shape)(x)

        model = Model(inputs=[inputs, action_input], outputs=outputs)
        model.compile(optimizer=Adam(lr=self.alpha), loss='mse')
        # model.compile(optimizer=RMSprop(lr=0.1), loss='mse')
        # model.compile(optimizer=SGD(lr=0.1), loss='mse')
        return model, inputs, action_input

    def get_gradients(self, states, actions):
        return self.sess.run(self.action_gradients, feed_dict={
                self.state_input: states, self.action_input: actions})[0]

    def train_target(self):
        """Train the target model."""
        weights = self.model.get_weights()
        target_weights = self.model.get_weights()
        target_weights = self.tau * actor_weights
                + (1 - self.tau) * target_weights
        self.target_model.set_weights(target_weights)

    def save_weights(self, file_name='critic.h5'):
        """Save the current model's weights."""
        self.model.save_weights(file_name)

    def load_weights(self, file_name='critic.h5'):
        """Load the saved weights for the model and target model."""
        self.model.load_weights(file_name)
        self.target_model.load_weights(file_name)