import copy
import tensorflow as tf

from tf_agents.trajectories import PolicyStep
import numpy as np


class PlayPolicy():
    def __init__(self, policy):
        self.policy = policy

    def action(self, act: int, env=None):
        action_step = self.policy.action(act)
        if env:
            # when the board is empty, choose the center position.
            py_env = env._envs[0]
            if np.sum(py_env.board) == 0:
                return PolicyStep(action=tf.convert_to_tensor(np.array([40])))
            elif np.sum(py_env.board) > 6:
                current_player = py_env.current_player
                opponent = current_player + 1 if current_player == 1 else 1
                occupied_positions = py_env.info['Occupied']
                for act in range(81):
                    if act not in occupied_positions and self.drop_here_will_win(py_env, act, current_player):
                        return PolicyStep(action=tf.convert_to_tensor(np.array([act + (py_env.num_bin - 1) * 81])))
                for act in range(81):
                    if act not in occupied_positions and self.drop_here_will_win(py_env, act, opponent):
                        return PolicyStep(action=tf.convert_to_tensor(np.array([act + (py_env.num_bin - 1) * 81])))
        return action_step

    def drop_here_will_win(self, py_env, action, color):
        # check if four equal stones are aligned (horizontal, vertical or diagonal)
        directions = [[0, 1], [1, 0], [1, 1], [1, -1]]

        current_board = copy.deepcopy(py_env.board)
        r, c, _ = py_env.decode_action(action)
        current_board[r, c] = color

        for direct in directions:
            count = 0
            for offset in range(-3, 4):
                if 0 <= r + offset * direct[0] < 9 and 0 <= c + offset * direct[1] < 9:
                    if current_board[r + offset * direct[0], c + offset * direct[1]] == color:
                        count += 1
                        if count == 4:
                            return True
                    else:
                        count = 0

        return False
