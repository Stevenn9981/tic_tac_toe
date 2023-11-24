import base64
import imageio
import IPython
import shutil
from tf_agents.trajectories import trajectory


def embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    video = open(filename, 'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
    <video width="480" height="480" controls>
      <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>'''.format(b64.decode())

    return IPython.display.HTML(tag)


def create_policy_eval_video(eval_env, policy, filename, fps=2):
    py_env = eval_env._envs[0]
    tf_env = eval_env
    filename = filename + ".mp4"
    with imageio.get_writer(filename, fps=fps) as video:
        time_step = tf_env.reset()
        video.append_data(py_env.render())
        while not time_step.is_last():
            action_step = policy.action(time_step, eval_env)
            time_step = tf_env.step(action_step.action)
            video.append_data(py_env.render())

    return embed_mp4(filename)


def create_policy_battle_video(eval_env, policy1, policy2, filename, fps=2):
    py_env = eval_env._envs[0]
    tf_env = eval_env
    filename = filename + ".mp4"
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


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def compute_avg_return_battle(environment, policy1, policy2, num_episodes=10):
    total_return_1 = 0.0
    total_return_2 = 0.0
    for _ in range(num_episodes):
        episode_return_1 = 0.0
        episode_return_2 = 0.0
        time_step = environment.reset()
        while not time_step.is_last():
            action_step = policy1.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return_1 += time_step.reward
            if not time_step.is_last():
                action_step = policy2.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return_2 += time_step.reward
        total_return_1 += episode_return_1
        total_return_2 += episode_return_2

    avg_return_1 = total_return_1 / num_episodes
    avg_return_2 = total_return_2 / num_episodes
    return [avg_return_1.numpy()[0], avg_return_2.numpy()[0]]


def compute_avg_win_battle(environment, policy1, policy2, num_episodes=10):
    total_return_1 = 0.0
    total_return_2 = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        while not time_step.is_last():
            action_step = policy1.action(time_step)
            time_step = environment.step(action_step.action)
            if not time_step.is_last():
                action_step = policy2.action(time_step)
                time_step = environment.step(action_step.action)
        if environment._envs[0].get_result() == 1:
            total_return_1 += 1
        elif environment._envs[0].get_result() == 2:
            total_return_2 += 1

    avg_return_1 = total_return_1 / num_episodes
    avg_return_2 = total_return_2 / num_episodes
    return [avg_return_1, avg_return_2]


def collect_episode(environment, agent1, agent2, replay_buffer):
    time_step = environment.reset()
    trajs_1, trajs_2 = [], []

    while not time_step.is_last():
        action_step = agent1.collect_policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj1 = trajectory.from_transition(time_step, action_step, next_time_step)
        trajs_1.append(traj1)
        time_step = next_time_step

        if not time_step.is_last():
            action_step = agent2.collect_policy.action(time_step)
            next_time_step = environment.step(action_step.action)
            traj2 = trajectory.from_transition(time_step, action_step, next_time_step)
            trajs_2.append(traj2)
            time_step = next_time_step

    change_flag = False
    # Modify the reward of each step according to the opponent's next step
    if len(trajs_1) == len(trajs_2):  # Player 2 won
        for i in range(len(trajs_1) - 1):
            trajs_1[i] = trajs_1[i].replace(reward=trajs_1[i].reward - trajs_2[i].reward)
            trajs_2[i] = trajs_2[i].replace(reward=trajs_2[i].reward - trajs_1[i + 1].reward)
        trajs_1[-1] = trajs_1[-1].replace(reward=trajs_1[-1].reward - trajs_2[-1].reward)
    else:  # Player 1 won
        change_flag = True
        for i in range(len(trajs_1) - 1):
            trajs_1[i] = trajs_1[i].replace(reward=trajs_1[i].reward - trajs_2[i].reward)
            trajs_2[i] = trajs_2[i].replace(reward=trajs_2[i].reward - trajs_1[i + 1].reward)

    for i in range(len(trajs_1)):
        replay_buffer.add_batch(trajs_1[i])
    for i in range(len(trajs_2)):
        replay_buffer.add_batch(trajs_2[i])

    # If the first player wins, then change the order in the next game.
    return change_flag


def create_zip_file(dirname, base_filename):
    return shutil.make_archive(base_filename, 'zip', dirname)
