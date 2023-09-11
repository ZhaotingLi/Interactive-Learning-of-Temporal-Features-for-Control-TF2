import numpy as np
from tools.functions import str_2_array
from buffer2 import Buffer
import tensorflow as tf # irene

"""
batched D-COACH implementation
"""


class BDCOACH:
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

        self.human_model_included = True
        self.agent_with_hm_learning_rate = 1e-3
        self.human_model_learning_rate = 1e-3
        # self.dim_o = 4 # cart pole
        self.dim_o = 3 # pendulum
        self.h_threshold = 0.1
        self.action_limit = 1

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

    def _single_update(self, state_representation, policy_label):


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
        current_observation_batch_tf = tf.convert_to_tensor(
            np.reshape(current_observation_batch, [batch_size, self.dim_o]), dtype=tf.float32)
        
        self._single_update(current_observation_batch_tf, action_label_batch)



    def Human_single_update(self, observation, action, h_human):
        # TRAIN Human model
        optimizer_Human_model = tf.keras.optimizers.Adam(learning_rate=self.human_model_learning_rate)

        with tf.GradientTape() as tape_policy:

            h_predicted = self.Human_model([observation, action])
            #print("h_human: ", h_human)
            #print("h_predicted: ", h_predicted)
            policy_loss = 0.5 * tf.reduce_mean(tf.square(h_human- h_predicted))
            grads = tape_policy.gradient(policy_loss, self.Human_model.trainable_variables)

        optimizer_Human_model.apply_gradients(zip(grads, self.Human_model.trainable_variables))

        return
    

    def _policy_batch_update_with_HM(self, batch):


        observations_batch = [np.array(pair[0]) for pair in batch]  # state(t) sequence
        observations_reshaped_tensor = tf.convert_to_tensor(np.reshape(observations_batch, [self.buffer_sampling_size, self.dim_o]),
                                                            dtype=tf.float32)

        optimizer_policy_model = tf.keras.optimizers.Adam(learning_rate=self.agent_with_hm_learning_rate)

        with tf.GradientTape() as tape_policy:
            # policy_output = self.policy_model([state_representation])
            actions_batch = self.policy_model([observations_reshaped_tensor])

            # 5. Get bath of h predictions from Human model
            h_predicted_batch = self.Human_model([observations_reshaped_tensor, actions_batch])

            h_predicted_batch = self.discretize_feedback(h_predicted_batch)

            # 6. Get batch of a_target from batch of predicted h (error = h * e --> a_target = a + error)
            #print("actions_batch: ", actions_batch)
            #print("h_predicted_batch: ", h_predicted_batch)
            a_target_batch = self._generate_batch_policy_label(actions_batch, h_predicted_batch)

            # 7. Update policy indirectly from Human model

            #print("a_target_batch", a_target_batch)

            policy_loss = 0.5 * tf.reduce_mean(tf.square(actions_batch - a_target_batch))
            grads = tape_policy.gradient(policy_loss, self.policy_model.trainable_variables)

        optimizer_policy_model.apply_gradients(zip(grads, self.policy_model.trainable_variables))


    def Human_batch_update(self, batch):
        #print('bufferF batch update')

        state_batch = [np.array(pair[0]) for pair in batch]  # state(t) sequence
        action_batch = [np.array(pair[1]) for pair in batch]
        h_human_batch = [np.array(pair[2]) for pair in batch]  # last
        #print("h_human_batch: ",h_human_batch)

        batch_size = len(state_batch)
        # Reshape and transform to tensor so they can be pass to the model:
        observation_reshaped_tensor = tf.convert_to_tensor(np.reshape(state_batch, [batch_size, self.dim_o]), dtype=tf.float32)
        action_reshaped_tensor      = tf.convert_to_tensor(np.reshape(action_batch, [batch_size, self.dim_a]), dtype=tf.float32)
        h_human_reshaped_tensor     = tf.convert_to_tensor(np.reshape(h_human_batch, [batch_size, self.dim_a]), dtype=tf.float32)



        self.Human_single_update(observation_reshaped_tensor, action_reshaped_tensor, h_human_reshaped_tensor)


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
            if self.human_model_included:
                self.Human_model = neural_network.Human_model()



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

            self._single_update(self.state_representation, self.policy_action_label)
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


    def TRAIN_Human_Model_included(self, action, h, observation, t, done):

            if np.any(h):  # if any element is not 0

                # 1. append  (o_t, a_t, h_t) to D
                self.h_to_buffer = tf.convert_to_tensor(np.reshape(h, [1, self.dim_a]), dtype=tf.float32)

                self.buffer.add([observation, action, h])

                # 2. Generate a_target
                self.h = h
                action_label = self._generate_policy_label(action)

                # 3. Update policy with current observation and a_target
                self._single_update(observation, self.policy_action_label)

                # 4. Update Human model with a minibatch sampled from buffer D
                if self.buffer.initialized():

                    batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                    self.Human_batch_update(batch)

                    # 4. Batch update of the policy with the Human Model

                    batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                    self._policy_batch_update_with_HM(batch)

            # Train policy every k time steps from buffer
            if self.buffer.initialized() and t % self.buffer_sampling_rate == 0 or (self.train_end_episode and done):
                #print('Train policy every k time steps from buffer')

                # update Human model
                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                self.Human_batch_update(batch)

                # Batch update of the policy with the Human Model
                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                self._policy_batch_update_with_HM(batch)


    def discretize_feedback(self, h_predicted_batch):

            h_predicted_batch = h_predicted_batch.numpy()
            h_predicted_batch = h_predicted_batch.tolist()

            for i in range(len(h_predicted_batch)):
                for j in range(len(h_predicted_batch[i])):
                    if h_predicted_batch[i][j] > -1 and h_predicted_batch[i][j] < -1 * self.h_threshold:
                        h_predicted_batch[i][j] = -1
                    elif h_predicted_batch[i][j] > self.h_threshold and h_predicted_batch[i][j] < 1:
                        h_predicted_batch[i][j] = 1
                    else:
                        h_predicted_batch[i][j] = 0



            return h_predicted_batch
    
    def _generate_batch_policy_label(self, action_batch, h_predicted_batch):


        #if np.any(h_predicted_batch):


        multi = np.asarray(h_predicted_batch) * self.e

        # print('h_predicted_batch',  h_predicted_batch)
        error = multi.reshape(self.buffer_sampling_size, self.dim_a)


        a_target_batch = []

        # a_target = a + error
        # numpy.clip(a, a_min, a_max)
        for i in range(self.buffer_sampling_size):

            a_target_batch.append(np.clip(action_batch[i] / self.action_limit + error[i], -1, 1))

        a_target_batch = np.array(a_target_batch).reshape(self.buffer_sampling_size, self.dim_a)


        return a_target_batch