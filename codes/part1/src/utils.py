import base64
import imageio
import IPython
import os
import tensorflow as tf
from tf_agents.trajectories import trajectory

from codes.part1.src.PlayPolicy import PlayPolicy


def embed_mp4(filename):
    """
        Embeds an mp4 file in the jupyter notebook.
        Args:
            filename (str): the file name of the mp4 file
    """
    video = open(filename, 'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
    <video width="480" height="480" controls>
      <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>'''.format(b64.decode())

    return IPython.display.HTML(tag)


def create_policy_eval_video(eval_env, policy, filename, fps=2):
    """
        Create an mp4 file that records the policy self-play.
        Args:
            eval_env: the game environment
            policy: the policy that will self-play the game
            filename (str): the file name of the mp4 file that will be created
    """
    py_env = eval_env._envs[0]
    tf_env = eval_env
    filename = 'videos/' + filename + ".mp4"
    if not os.path.exists('videos'):
        os.makedirs('videos')
    with imageio.get_writer(filename, fps=fps) as video:
        time_step = tf_env.reset()
        video.append_data(py_env.render())
        while not time_step.is_last():
            action_step = policy.action(time_step, eval_env)
            time_step = tf_env.step(action_step.action)
            video.append_data(py_env.render())

    return embed_mp4(filename)


def create_policy_battle_video(eval_env, policy1, policy2, filename, fps=2):
    """
        Create an mp4 file that records a match between two policies.
        Args:
            eval_env: the game environment
            policy1: the first player
            policy2: the second player
            filename (str): the file name of the mp4 file that will be created
    """
    py_env = eval_env._envs[0]
    tf_env = eval_env
    filename = 'videos/' + filename + ".mp4"
    if not os.path.exists('videos'):
        os.makedirs('videos')
    with imageio.get_writer(filename, fps=fps) as video:
        time_step = tf_env.reset()
        video.append_data(py_env.render())
        while not time_step.is_last():
            action_step = policy1.action(time_step, eval_env)
            time_step = tf_env.step(action_step.action)
            video.append_data(py_env.render())
            if not time_step.is_last():
                action_step = policy2.action(time_step, eval_env)
                time_step = tf_env.step(action_step.action)
                video.append_data(py_env.render())

    return embed_mp4(filename)


def compute_avg_win_battle(env, policy1, policy2, num_episodes=10):
    """
        Two policies against each other for {num_episodes} matches. Calculate the win rates of the two policies.
        Args:
            env: the game environment
            policy1: the first player
            policy2: the second player
            num_episodes (int): the number of matches
        Return:
            avg_win_1: win rate of policy 1
            avg_win_2: win rate of policy 2
    """
    total_win_1 = 0.0
    total_win_2 = 0.0
    for _ in range(num_episodes):
        time_step = env.reset()
        while not time_step.is_last():
            action_step = policy1.action(time_step, env) if isinstance(policy1, PlayPolicy) else policy1.action(
                time_step)
            time_step = env.step(action_step.action)
            if not time_step.is_last():
                action_step = policy2.action(time_step, env) if isinstance(policy2, PlayPolicy) else policy2.action(
                    time_step)
                time_step = env.step(action_step.action)
        if env._envs[0].get_result() == 1:
            total_win_1 += 1
        elif env._envs[0].get_result() == 2:
            total_win_2 += 1

    avg_win_1 = total_win_1 / num_episodes
    avg_win_2 = total_win_2 / num_episodes
    return [avg_win_1, avg_win_2]


def collect_episode(env, agent1, agent2, replay_buffer):
    """
        Two RL agents against each other for a match.
        Use a replay buffer to record the data of the two agents for training.
        Args:
            env: the game environment
            agent1: the first player
            agent2: the second player
            replay_buffer: the replay_buffer used to record the training data
    """

    time_step = env.reset()
    trajs_1, trajs_2 = [], []

    while not time_step.is_last():
        action_step = agent1.collect_policy.action(time_step)
        next_time_step = env.step(action_step.action)
        traj1 = trajectory.from_transition(time_step, action_step, next_time_step)
        trajs_1.append(traj1)
        time_step = next_time_step

        if not time_step.is_last():
            action_step = agent2.collect_policy.action(time_step)
            next_time_step = env.step(action_step.action)
            traj2 = trajectory.from_transition(time_step, action_step, next_time_step)
            trajs_2.append(traj2)
            time_step = next_time_step

    # Modify the reward of each step according to the opponent's next step
    if len(trajs_1) == len(trajs_2):  # Player 2 won
        for i in range(len(trajs_1) - 1):
            # This is to modify the reward of the previous step. For example, if player 2 got an active_two, the
            # reward of the previous step of player 1 will decrease by REWARD_ACTIVE_TWO. tf.math.round is used
            # to ignore the REWARD_NON_ADJ here.
            trajs_1[i] = trajs_1[i].replace(reward=trajs_1[i].reward - tf.math.round(trajs_2[i].reward))
            trajs_2[i] = trajs_2[i].replace(reward=trajs_2[i].reward - tf.math.round(trajs_1[i + 1].reward))
        trajs_1[-1] = trajs_1[-1].replace(reward=trajs_1[-1].reward - tf.math.round(trajs_2[-1].reward))
    else:  # Player 1 won
        for i in range(len(trajs_1) - 1):
            trajs_1[i] = trajs_1[i].replace(reward=trajs_1[i].reward - tf.math.round(trajs_2[i].reward))
            trajs_2[i] = trajs_2[i].replace(reward=trajs_2[i].reward - tf.math.round(trajs_1[i + 1].reward))

    for i in range(len(trajs_1)):
        replay_buffer.add_batch(trajs_1[i])
    for i in range(len(trajs_2)):
        replay_buffer.add_batch(trajs_2[i])
