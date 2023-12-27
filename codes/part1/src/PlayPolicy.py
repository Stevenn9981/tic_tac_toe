import copy
import tensorflow as tf

from tf_agents.trajectories import PolicyStep, TimeStep
import numpy as np

from codes.part1.src.settings import BOARD_SIZE


def drop_here_will_win(py_env, action, color):
    # check if four equal stones are aligned (horizontal, vertical or diagonal)
    directions = [[0, 1], [1, 0], [1, 1], [1, -1]]

    current_board = copy.deepcopy(py_env.board)
    r, c = py_env.decode_action(action)
    current_board[r, c] = color

    for direct in directions:
        count = 0
        for offset in range(-3, 4):
            if 0 <= r + offset * direct[0] < BOARD_SIZE and 0 <= c + offset * direct[1] < BOARD_SIZE:
                if current_board[r + offset * direct[0], c + offset * direct[1]] == color:
                    count += 1
                    if count == 4:
                        return True
                else:
                    count = 0

    return False


class PlayPolicy():
    def __init__(self, policy):
        self.policy = policy

    def action(self, time_step: TimeStep, env=None) -> PolicyStep:
        if env:
            py_env = env._envs[0]
            if np.sum(py_env.board) > 2:
                current_player = py_env.current_player
                opponent = current_player + 1 if current_player == 1 else 1
                occupied_positions = py_env.info['Occupied']
                # Find out if there is a position where the current player can win the game
                for act in range(BOARD_SIZE * BOARD_SIZE):
                    if act not in occupied_positions and drop_here_will_win(py_env, act, current_player):
                        return PolicyStep(action=tf.convert_to_tensor(np.array([act])))
                # Find out if there is a position where the opponent player can win the game
                for act in range(BOARD_SIZE * BOARD_SIZE):
                    if act not in occupied_positions and drop_here_will_win(py_env, act, opponent):
                        return PolicyStep(action=tf.convert_to_tensor(np.array([act])))
        return self.policy.action(time_step)
