import numpy as np
from tools.functions import str_2_array
from buffer2 import Buffer
import tensorflow as tf # irene

"""
D-COACH implementation
"""


class DCOACH:
    def __init__(self, dim_a, action_upper_limits, action_lower_limits, e, buffer_min_size, buffer_max_size,
                 buffer_sampling_rate, buffer_sampling_size, train_end_episode):
        self.low_dim_state = True

        # Initialize variables
        self.h = None
        self.state_representation = None
        self.policy_action_label = None
        self.e = np.array(str_2_array(e, type_n='float'))
        self.dim_a = dim_a
        self.action_upper_limits = str_2_array(action_upper_limits, type_n='float')
        self.action_lower_limits = str_2_array(action_lower_limits, type_n='float')
        self.count = 0
        self.buffer_sampling_rate = buffer_sampling_rate
        self.buffer_sampling_size = buffer_sampling_size
        self.train_end_episode = train_end_episode

        # Initialize DCOACH buffer
        self.buffer = Buffer(min_size=buffer_min_size, max_size=buffer_max_size)

    def _generate_policy_label(self, action):
        if np.any(self.h):
            error = np.array(self.h * self.e).reshape(1, self.dim_a)
            self.policy_action_label = []

            for i in range(self.dim_a):
                self.policy_action_label.append(np.clip(action[i] / self.action_upper_limits[i] + error[0, i], -1, 1))

            self.policy_action_label = np.array(self.policy_action_label).reshape(1, self.dim_a)
        else:
            self.policy_action_label = np.reshape(action, [1, self.dim_a])

    def _single_update(self, neural_network, state_representation, policy_label):


        # TRAIN policy model
        optimizer_policy_model = tf.keras.optimizers.SGD(learning_rate=0.003)

        with tf.GradientTape() as tape_policy:

            policy_output = self.policy_model([state_representation])

            policy_loss = 0.5 * tf.reduce_mean(tf.square(policy_output - policy_label))
            grads = tape_policy.gradient(policy_loss, self.policy_model.trainable_variables)

        optimizer_policy_model.apply_gradients(zip(grads, self.policy_model.trainable_variables))
        '''
        tf.keras.utils.plot_model(self.policy_model,
                                  to_file='policy.png',
                                  show_shapes=True,
                                  show_layer_names=True)
        '''

    def _batch_update(self, neural_network, transition_model, batch, i_episode, t):
        observation_sequence_batch = [np.array(pair[0]) for pair in batch]  # state(t) sequence
        action_sequence_batch = [np.array(pair[1]) for pair in batch]
        current_observation_batch = [np.array(pair[2]) for pair in batch]  # last
        action_label_batch = [np.array(pair[3]) for pair in batch]

        batch_size = len(observation_sequence_batch)

        lstm_hidden_state_batch = transition_model.get_lstm_hidden_state_batch(neural_network,
                                                                                     observation_sequence_batch,
                                                                                     action_sequence_batch, batch_size)

        state_representation_batch = transition_model.get_state_representation_batch(neural_network, current_observation_batch, lstm_hidden_state_batch, batch_size)


        self._single_update(neural_network, state_representation_batch, action_label_batch)

    
    def _batch_update_LowDim(self, neural_network, transition_model, batch, i_episode, t):
        observation_sequence_batch = [np.array(pair[0]) for pair in batch]  # state(t) sequence
        action_sequence_batch = [np.array(pair[1]) for pair in batch]
        current_observation_batch = [np.array(pair[2]) for pair in batch]  # last
        action_label_batch = [np.array(pair[3]) for pair in batch]

        batch_size = len(observation_sequence_batch)
        obs_dim = 4 # for cart pole
        current_observation_batch_tf = tf.convert_to_tensor(
            np.reshape(current_observation_batch, [batch_size, obs_dim]), dtype=tf.float32)
        
        self._single_update(neural_network, current_observation_batch_tf, action_label_batch)



    def feed_h(self, h):
        self.h = h

    def action(self, neural_network, state_representation, i_episode, t):
        self.count += 1
        self.state_representation = state_representation
        # shape and type of state_representation
        print('state_representation', self.state_representation.shape, " ",  self.state_representation.dtype)

        if i_episode == 0 and t == 0 :
            # self.policy_model = neural_network.policy_model()
            self.policy_model = neural_network.policy_model_low_dim()



        action = self.policy_model([self.state_representation])

        action = action.numpy()



        #action = neural_network.sess.run(neural_network.policy_output,
                                         #feed_dict={'policy/state_representation:0': self.state_representation})
        out_action = []

        for i in range(self.dim_a):
            action[0, i] = np.clip(action[0, i], -1, 1) * self.action_upper_limits[i]
            out_action.append(action[0, i])

        return np.array(out_action)

    def train(self, neural_network, transition_model, action, t, done, i_episode):
        self._generate_policy_label(action)
        #print('train agent')
        # Policy training
        if np.any(self.h):  # if any element is not 0

            self._single_update(neural_network, self.state_representation, self.policy_action_label)
            print('agent single update')
            print("feedback:", self.h)

            if self.low_dim_state:
                self.buffer.add([self.state_representation, action, self.state_representation, self.policy_action_label])
            else:
                # Add last step to memory buffer
                if transition_model.last_step(self.policy_action_label) is not None:
                    print("appended data", transition_model.last_step(self.policy_action_label))
                    self.buffer.add(transition_model.last_step(self.policy_action_label))

            # Train sampling from buffer
            if self.buffer.initialized():
                print('Train sampling from buffer')

                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)  # TODO: probably this config thing should not be here
                if self.low_dim_state:
                    self._batch_update_LowDim(neural_network, transition_model, batch, i_episode, t)
                else:
                    self._batch_update(neural_network, transition_model, batch, i_episode, t)

        # Train policy every k time steps from buffer
        if self.buffer.initialized() and t % self.buffer_sampling_rate == 0 or (self.train_end_episode and done):
            print('Train policy every k time steps from buffer')
            batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
            if self.low_dim_state:
                self._batch_update_LowDim(neural_network, transition_model, batch, i_episode, t)
            else:
                self._batch_update(neural_network, transition_model, batch, i_episode, t)
