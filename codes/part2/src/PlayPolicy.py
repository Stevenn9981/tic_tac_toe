import copy
import tensorflow as tf

from tf_agents.trajectories import PolicyStep, TimeStep
import numpy as np

from codes.part2.src.settings import BOARD_SIZE


def drop_here_will_win(py_env, action, color):
    """Check whether the player can win the game if the stone is placed in the given place

    Args:
        py_env: the game environment
        action (int): the action which represents the chosen place
        color (int): indicates which player is going to play
    """

    # check if four equal stones are aligned (horizontal, vertical or diagonal)
    directions = [[0, 1], [1, 0], [1, 1], [1, -1]]

    current_board = copy.deepcopy(py_env.board)
    r, c, _ = py_env.decode_action(action)
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
    """
        This class is an encapsulation of the trained policy.
        It allows us to write some strict rules instead of only relying on the predictions of trained RL policy.
    """

    def __init__(self, policy):
        self.policy = policy

    def action(self, time_step: TimeStep, env=None) -> PolicyStep:
        """
        Choose the next action based on the current time step. We add some strict rules here:
            1. If there is a place where the current player can win the game, use as many energy points as possible
            and place the stone right down there.
            2. Else if there is a place where the opponent player can win, use as many energy points as possible
            and place the stone right down there.
            3. Else, let the trained RL policy decide where to go.
        Args:
            time_step: current TimeStep
            env: the game environment
        """
        if env:
            py_env = env._envs[0]
            if np.sum(py_env.board) > 2:
                current_player = py_env.current_player
                opponent = current_player + 1 if current_player == 1 else 1
                occupied_positions = py_env.info['Occupied']
                # Find out if there is a position where the current player can win the game
                for act in range(BOARD_SIZE * BOARD_SIZE):
                    if act not in occupied_positions and drop_here_will_win(py_env, act, current_player):
                        return PolicyStep(action=tf.convert_to_tensor(
                            np.array([act + (py_env.num_bin - 1) * BOARD_SIZE * BOARD_SIZE])))
                # Find out if there is a position where the opponent player can win the game
                for act in range(BOARD_SIZE * BOARD_SIZE):
                    if act not in occupied_positions and drop_here_will_win(py_env, act, opponent):
                        return PolicyStep(action=tf.convert_to_tensor(
                            np.array([act + (py_env.num_bin - 1) * BOARD_SIZE * BOARD_SIZE])))
        return self.policy.action(time_step)
