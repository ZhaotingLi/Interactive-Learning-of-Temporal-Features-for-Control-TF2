import numpy as np
import time
from main_init2 import neural_network, transition_model, transition_model_type, agent, agent_type, exp_num,count_down, \
    max_num_of_episodes, env, render, max_time_steps_episode, human_feedback, save_results, eval_save_path, \
    render_delay, save_policy, save_transition_model

"""
Main loop of the algorithm described in the paper 'Interactive Learning of Temporal Features for Control' 
"""

# Initialize variables
total_feedback, total_time_steps, trajectories_database, total_reward = [], [], [], []
t_total, h_counter, last_t_counter, omg_c, eval_counter, total_r = 1, 0, 0, 0, 0, 0
human_done, evaluation, random_agent, evaluation_started = False, False, False, False

init_time = time.time()

# Print general information
print('\nExperiment number:', exp_num)
print('Environment:', env)
print('Learning algorithm: ', agent_type)
print('Transition Model:', transition_model_type, '\n')

time.sleep(2)

# Count-down before training if requested
if count_down:
    for i in range(10):
        print(' ' + str(10 - i) + '...')
        time.sleep(1)

low_dimension_state = True  # if true, use low dimensional state representation

# Start training loop
for i_episode in range(max_num_of_episodes):
    print('Starting episode number', i_episode)

    if not evaluation:
        transition_model.new_episode()

    observation = env.reset()  # reset environment at the beginning of the episode

    print(observation)

    past_action, past_observation, episode_trajectory, h_counter, r = None, None, [], 0, 0  # reset variables for new episode


    # Iterate over the episode
    for t in range(int(max_time_steps_episode)):

        print(t)

        if render:
            env.render()  # Make the environment visible
            time.sleep(render_delay)  # Add delay to rendering if necessary

        # Get feedback signal
        h = human_feedback.get_h()
        evaluation = human_feedback.evaluation

        # Feed h to agent
        agent.feed_h(h)



        # Map action from observation
        if(not low_dimension_state):
            state_representation = transition_model.get_state_representation(neural_network, observation,  i_episode, t)
        else:
            # obs_dim = 3 # for pendulum
            obs_dim = 4 # for cart pole
            state_representation = observation.reshape(1,obs_dim)
        
        action = agent.action(neural_network, state_representation, i_episode, t)

        # print action
        print(action)

        # Act
        observation, reward, environment_done, info = env.step(action)


        # Compute done
        done = human_feedback.ask_for_done() or environment_done

        # Compute new hidden state of LSTM
        if(not low_dimension_state):
            transition_model.compute_lstm_hidden_state(neural_network, action)

        # Append transition to database
        if not evaluation:
            if past_action is not None and past_observation is not None:
                if(not low_dimension_state):
                    episode_trajectory.append([past_observation, past_action, transition_model.processed_observation])  # append o, a, o' (not really necessary to store it like this)
                else:
                    episode_trajectory.append([past_observation, past_action, observation])
            if(not low_dimension_state):
                past_observation, past_action = transition_model.processed_observation, action
            else:
                past_observation, past_action = observation, action

            if t % 100 == 0 or done:
                trajectories_database.append(episode_trajectory)  # append episode trajectory to database
                episode_trajectory = []

        if np.any(h):
            h_counter += 1

        # Update weights transition model/policy
        if not evaluation:
            if done:
                t_total = done  # tell the agents that the episode finished

            if(not low_dimension_state):
                transition_model.train(neural_network, t_total, done, trajectories_database, i_episode)

            # agent.train(neural_network, transition_model, action, t_total, done, i_episode)
            agent.TRAIN_Human_Model_included(action, h, state_representation, t_total, done )



            t_total += 1

        # Accumulate reward (not for learning purposes, only to quantify the performance of the agents)
        r += reward

        # End of episode
        if done:
            if evaluation:
                total_r += r

                print('Episode Reward:', '%.3f' % r)
                print('\n', i_episode, 'avg reward:', '%.3f' % (total_r / (i_episode + 1)), '\n')
                print('Percentage of given feedback:', '%.3f' % ((h_counter / (t + 1e-6)) * 100))
                total_reward.append(r)
                total_feedback.append(h_counter/(t + 1e-6))
                total_time_steps.append(t_total)
                if save_results:
                    np.save(eval_save_path + exp_num + '_reward', total_reward)
                    np.save(eval_save_path + exp_num + '_feedback', total_feedback)
                    np.save(eval_save_path + exp_num + '_time', total_time_steps)

            if save_policy:
                neural_network.save_policy()

            if save_transition_model:
                neural_network.save_transition_model()

            if render:
                time.sleep(1)

            print('Total time (s):', '%.3f' % (time.time() - init_time))


            break
