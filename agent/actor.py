import keras.backend as K
from keras.layers import Input, Dense
from keras.layers.merge import concatenate
from keras.models import Model
from keras.optimizers import Adam, RMSprop, SGD
import tensorflow as tf


class Actor:
    """An actor model selecting an action for a given state."""

    def __init__(self, sess, alpha=0.001, tau=0.001, n1=100, n2=150):
        """Construct an actor with the given session, learning rate,
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
        self.model = self.create_model()
        self.target_model = self.create_model()
        if load:
            self.load_weights()
        self.sess.run(tf.initialize_all_variables())

    def create_model(self):
        """Return a compiled model."""
        inputs = Input(shape=(self.state_shape,))
        x = Dense(self.n1, activation='relu')(inputs)
        x = Dense(self.n2, activation='relu')(x)
        outputs = Dense(self.action_shape)(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(lr=self.alpha), loss='mse')
        # model.compile(optimizer=RMSprop(lr=0.1), loss='mse')
        # model.compile(optimizer=SGD(lr=0.1), loss='mse')
        return model

    def train_target(self):
        """Train the target model."""
        weights = self.model.get_weights()
        target_weights = self.model.get_weights()
        target_weights = self.tau * actor_weights
                + (1 - self.tau) * target_weights
        self.target_model.set_weights(target_weights)

    def save_weights(self, file_name='actor.h5'):
        """Save the current model's weights."""
        self.model.save_weights(file_name)

    def load_weights(self, file_name='actor.h5'):
        """Load the saved weights for the model and target model."""
        self.model.load_weights(file_name)
        self.target_model.load_weights(file_name)