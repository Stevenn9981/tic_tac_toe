import random
import sys

sys.path.append('.')

from codes.part2.src.TicTacToeEnv2 import TicTacToeEnv2
from codes.part2.src.settings import *
from codes.part2.src.utils import *

import os
import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import sequential
from tf_agents.policies import policy_saver
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.utils import common


def show_random_policy():
    """
        Create a video (named named RandomPlay_Part2.mp4 in videos folder) that records how a random policy plays the game
    """
    py_env = TicTacToeEnv2()
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    random_policy = create_random_policy(tf_env)
    create_policy_eval_video(tf_env, random_policy, "RandomPlay_Part2")


def observation_and_action_constraint_splitter(obs):
    """
        A splitter that splits states and action constraints in observations
    """
    return obs['state'], obs['legal_moves']


def create_q_net(train_env):
    """
        Create a Q network used in DQN
        Args:
            train_env: the game environments
        Return:
            q_net: the created Q network
    """
    conv_layer_params = [32, 64, 128]
    action_tensor_spec = tensor_spec.from_spec(train_env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

    # Define a helper function to create Conv layers configured with the right
    # activation and kernel initializer.
    def conv_layer(num_units):
        return tf.keras.layers.Conv2D(
            filters=num_units,
            kernel_size=[3, 3],
            padding="same",
            data_format="channels_last",
            activation=tf.nn.leaky_relu,
            dtype=float)

    # QNetwork consists of a sequence of Conv layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # its output.
    normalization1 = tf.keras.layers.BatchNormalization()
    normalization2 = tf.keras.layers.BatchNormalization()
    conv_layers = [conv_layer(num_units) for num_units in conv_layer_params]
    action_conv = tf.keras.layers.Conv2D(filters=4,
                                         kernel_size=[1, 1], padding="same",
                                         data_format="channels_last")
    flatten = tf.keras.layers.Flatten()
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.2))
    q_net = sequential.Sequential(
        [normalization1] + conv_layers + [action_conv, normalization2, flatten, q_values_layer])

    return q_net


def create_dqn_agent(train_env):
    """
        Create a DQN agent
        Args:
            train_env: the game environments
        Return:
            agent: the created DQN agent
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    q_net = create_q_net(train_env)
    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        n_step_update=n_step_update,
        observation_and_action_constraint_splitter=observation_and_action_constraint_splitter,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=tf.Variable(0))
    agent.initialize()

    return agent


def create_random_policy(train_env):
    """
        Create a random policy for the game
        Args:
            train_env: the game environments
        Return:
            the created random policy
    """
    return random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec(),
                                           observation_and_action_constraint_splitter=observation_and_action_constraint_splitter)


def train_game_agent():
    """
        Train the RL agent. We let the RL agent self-play to collect training data and
        save the trained RL agent in the `pretrained_policy_part1` folder
    """
    train_py_env = TicTacToeEnv2()
    eval_py_env = TicTacToeEnv2()

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    agent = create_dqn_agent(train_env)

    play_policy = agent.policy
    random_policy = create_random_policy(train_env)

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)

    for _ in range(initial_collect_episodes):
        collect_episode(train_env, agent, agent, replay_buffer)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size,
        num_steps=n_step_update + 1).prefetch(3)
    iterator = iter(dataset)

    policy_dir = os.path.join(tempdir, 'part2/pretrained_policy_part2')
    tf_policy_saver = policy_saver.PolicySaver(play_policy)

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step.
    agent.train_step_counter.assign(0)
    win1 = compute_avg_win_battle(eval_env, play_policy, random_policy, num_eval_episodes)[0]
    win2 = compute_avg_win_battle(eval_env, random_policy, play_policy, num_eval_episodes)[1]
    print('Evaluation before training: offense_win = {0}, defense_win = {1}'.format(win1, win2))

    bst = 0  # store the value of best-so-far win rate

    # Reset the environment.
    train_env.reset()

    for idx in range(num_iterations):
        # Collect a few episodes using collect_policy and store the transitions to the replay buffer.
        for _ in range(collect_episodes_per_iteration):
            collect_episode(train_env, agent, agent, replay_buffer)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss1 = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            policy_win_rate1 = compute_avg_win_battle(eval_env, play_policy, random_policy, num_eval_episodes)[0]
            policy_win_rate2 = compute_avg_win_battle(eval_env, random_policy, play_policy, num_eval_episodes)[1]
            print('Evaluation (step = {0}): offense_win = {1}, defense_win = {2}'.format(step,
                                                                                         policy_win_rate1,
                                                                                         policy_win_rate2,
                                                                                         ))
            policy_win_rate = (policy_win_rate1 + policy_win_rate2) / 2
            if policy_win_rate >= bst:
                bst = policy_win_rate
                tf_policy_saver.save(policy_dir)


def test_game_agent():
    """
        Test the trained RL agent. We compared the trained RL agents with itself,
        random policy (as first-player and second-player), respectively. Then calculate
        the win rates of the trained RL agent in each scenario. Besides, we also
        generate videos to record the matches under each scenario to give a more
        intuitive view. All the videos are saved in the `videos` folder.
    """
    eval_py_env = TicTacToeEnv2()
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    random_policy = create_random_policy(eval_env)
    policy_dir = os.path.join(tempdir, 'part2/pretrained_policy_part2')

    play_policy = PlayPolicy(tf.saved_model.load(policy_dir))

    win_rate_1 = compute_avg_win_battle(eval_env, play_policy, random_policy, num_test_episodes)[0]
    win_rate_2 = compute_avg_win_battle(eval_env, random_policy, play_policy, num_test_episodes)[1]
    print(f'Play {num_test_episodes} rounds against the random policy as first and second player, separately:')
    print(f'Win rate as the first player: {win_rate_1 * 100:.1f}%, Win rate as the second player: {win_rate_2 * 100:.1f}%')

    create_policy_battle_video(eval_env, play_policy, random_policy, 'TrainedAgent2-vs-Random')
    create_policy_battle_video(eval_env, random_policy, play_policy, 'Random-vs-TrainedAgent2')
    create_policy_battle_video(eval_env, play_policy, play_policy, 'TrainedAgent2-vs-TrainedAgent2')


if __name__ == '__main__':
    # show_random_policy()
    if len(sys.argv) == 1 or sys.argv[1] != 'use_pretrain':
        train_game_agent()
    test_game_agent()
