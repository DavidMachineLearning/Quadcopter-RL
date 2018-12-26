import tensorflow as tf
import numpy as np


class Actor:
    """Actor Model."""
    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        self.model = None
        self.train_fn = None
        self._build_model()

    def _build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (as a placeholder tensor)
        states = tf.keras.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        net = tf.keras.layers.Dense(units=32,kernel_regularizer=tf.keras.regularizers.L1L2(l2=1e-6))(states)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation('relu')(net)
        net = tf.keras.layers.Dense(units=64,kernel_regularizer=tf.keras.regularizers.L1L2(l2=1e-6))(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation('relu')(net)
        net = tf.keras.layers.Dense(units=128,kernel_regularizer=tf.keras.regularizers.L1L2(l2=1e-6))(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation('relu')(net)
        net = tf.keras.layers.Dense(units=128, activation="relu")(net)

        # Add final output layer with sigmoid activation
        raw_actions = tf.keras.layers.Dense(units=self.action_size, activation='sigmoid',
                                            name='raw_actions')(net)

        # Scale [0, 1] output for each action dimension to proper range
        actions = tf.keras.layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
                                         name='actions')(raw_actions)

        # Create Keras model
        self.model = tf.keras.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = tf.keras.layers.Input(shape=(self.action_size,))
        loss = tf.keras.backend.mean(-action_gradients * actions)

        # Define optimizer and training function
        optimizer = tf.keras.optimizers.Adam(lr=6.5e-4)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = tf.keras.backend.function(inputs=[self.model.input, action_gradients,
                                                          tf.keras.backend.learning_phase()],
                                                  outputs=[], updates=updates_op)


class Critic:
    """Critic Model."""
    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.model = None
        self.get_action_gradients = None
        self._build_model()

    def _build_model(self):
        """Build a critic network that maps (state, action) pairs -> Q-values."""
        # Define input layers (as placeholder tensors)
        states = tf.keras.layers.Input(shape=(self.state_size,), name='states')
        actions = tf.keras.layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = tf.keras.layers.Dense(units=32,kernel_regularizer=tf.keras.regularizers.L1L2(l2=1e-6))(states)
        net_states = tf.keras.layers.BatchNormalization()(net_states)
        net_states = tf.keras.layers.Activation('relu')(net_states)
        net_states = tf.keras.layers.Dense(units=64, activation='relu')(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = tf.keras.layers.Dense(units=32,kernel_regularizer=tf.keras.regularizers.L1L2(l2=1e-6))(actions)
        net_actions = tf.keras.layers.BatchNormalization()(net_actions)
        net_actions = tf.keras.layers.Activation('relu')(net_actions)
        net_actions = tf.keras.layers.Dense(units=64, activation='relu')(net_actions)

        # Combine state and action pathways
        net = tf.keras.layers.Add()([net_states, net_actions])
        net = tf.keras.layers.Activation('relu')(net)

        # Add final output layer to prduce action values (Q values)
        q_values = tf.keras.layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = tf.keras.Model(inputs=[states, actions], outputs=q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = tf.keras.optimizers.Adam(lr=1e-4)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = tf.keras.backend.gradients(q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = tf.keras.backend.function(inputs=[*self.model.input,
                                                                      tf.keras.backend.learning_phase()],
                                                              outputs=action_gradients)
