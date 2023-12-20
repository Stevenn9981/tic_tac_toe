import random
import os

import pygame as pg
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts, TimeStep

import numpy as np
from tabulate import tabulate

from typing import Tuple, List

from codes.part1.src.settings import *


class TicTacToeEnv1(py_environment.PyEnvironment):
    """
    Implementation of a TicTacToe Environment based on the instructions of Part 1, Question 1.
    """

    def __init__(self, train=False) -> None:
        """This class contains a TicTacToe environment for Part 1

        Args:
            train (bool): whether this is an environment for training.
        """
        self.n_actions = BOARD_SIZE * BOARD_SIZE  # 9 * 9 grids to drop

        self._observation_spec = {
            'state': array_spec.BoundedArraySpec(shape=(BOARD_SIZE, BOARD_SIZE, 5), dtype=np.int_, minimum=0,
                                                 maximum=1),
            'legal_moves': array_spec.ArraySpec(shape=(self.n_actions,), dtype=np.bool_)}

        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int_, minimum=0, maximum=self.n_actions - 1)
        self.colors = [1, 2]
        self.screen = None
        self.fields_per_side = BOARD_SIZE
        self.train = train
        self.reset()

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def get_result(self):
        return self.result

    def if_chess_nearby(self, row, col) -> bool:
        """
        Determine whether there are chess pieces in 2 squares around the given position (row, col)

        Args:
          row (int): row index
          col (int): column index

        Returns:
          res (boolean): true, if there are
        """
        for i in range(-2, 3):
            for j in range(-2, 3):
                if 0 <= row + i < BOARD_SIZE and 0 <= col + j < BOARD_SIZE:
                    if self.board[row + i, col + j] != 0:
                        return True
        return False

    def _reset(self) -> Tuple[np.ndarray, dict]:
        """
        reset the board game and state
        """
        self.board: np.ndarray = np.zeros(
            (self.fields_per_side, self.fields_per_side), dtype=int
        )
        self.current_player = 1
        self.info = {"Occupied": set(), "legal_moves": np.ones((self.n_actions,), dtype=bool)}
        self.latest_action = None

        # 0 means not finished, 1 or 2 means the winner and 3 means draw
        self.result = 0

        # return self.decompose_board_to_state()
        observations_and_legal_moves = {'state': self.decompose_board_to_state(),
                                        'legal_moves': self.info["legal_moves"]}
        return ts.restart(observations_and_legal_moves)

    def decompose_board_to_state(self):
        """
        Our state is a 9x9x5 matrix.
        The first layer is the opponent's play history, 0 means no stone, 1 means stones placed by the opponent.
        The second layer is the current player's history, 0 means no stone, 1 means stones placed by the current player.
        The third layer is the opponent's latest play-out; only one entry is 1 and the others are 0. If the board is empty now, all entries are 0.
        The fourth layer is whether the current player is the first hand. an array that is full of 1 means yes, and 0 means no.
        The fifth layer shows the empty positions whose adjacent positions are not all empty. 1 means there is at least one chess piece in its adjacent positions, and 0 means no.
        """
        opponent = 2 if self.current_player == 1 else 1
        o_plays = (self.board == opponent) * 1
        c_plays = (self.board == self.current_player) * 1
        l_play = np.zeros_like(self.board)
        if self.latest_action:
            r, c = self.decode_action(self.latest_action)
            l_play[r, c] = 1
        if_first = np.full_like(self.board, (self.current_player == 1) * 1)
        adj_play = np.zeros_like(self.board)
        for row in range(adj_play.shape[0]):
            for col in range(adj_play.shape[1]):
                if self.board[row, col] == 0 and self.if_chess_nearby(row, col):
                    adj_play[row, col] = 1
        return np.stack([o_plays, c_plays, l_play, if_first, adj_play], axis=2)

    def _step(self, action: int) -> TimeStep:
        """step function of the tictactoeEnv1

        Args:
          action (int): integer between [0, 80], each representing a field on the board

        Returns:
          state (np.array): state of 2 players' history, 0 means no stone, 1 means stones placed by the corresponding player (shape: 9x9x2).
          reward (int): reward of the current step
          done (boolean): true, if the game is finished
          (dict): empty dict for future game related information
        """
        action = int(action)
        if not (0 <= action < self.n_actions):
            raise ValueError(f"action '{action}' is not in action_space")

        reward = REWARD_ALIVE
        (row, col) = self.decode_action(action)

        # If the agent/player does not choose an empty square, raise the ValueError.
        if self.board[row, col] != 0:
            if len(self.info["Occupied"]) == BOARD_SIZE * BOARD_SIZE:
                raise ValueError('BORAD IS FULL!')
            raise ValueError('ERROR: Not A LEGAL MOVE (NOT EMPTY)')

        # According to the game rules, randomly select an adjacent position with a probability of 1/16.
        # Note that since this will bring some randomness to the training process, we disable this during training.
        if not self.train:
            if random.random() < 0.5:
                row, col = self.choose_adj_pos(row, col)

        # if len(self.info["Occupied"]) != 0 and not self.if_chess_nearby(row, col):
        #     reward += REWARD_NON_ADJ

        win = False
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and self.board[row, col] == 0:
            self.board[row, col] = self.current_player  # drop the piece on the field
            win = self._is_win(row, col)

            cnt_act_two = self.detect_alive_two(row, col)
            cnt_non_act_three, cnt_act_three = self.detect_three(row, col)

            reward += (
                    cnt_act_two * REWARD_ACTIVE_TWO + cnt_non_act_three * REWARD_NONACT_THREE + cnt_act_three * REWARD_ACTIVE_THREE)

            action = row * BOARD_SIZE + col
            self.latest_action = action
            self.info["Occupied"].add(action)
            self.info['legal_moves'][action] = False

        if win:
            self.result = self.current_player
            reward += REWARD_WIN
        elif len(self.info["Occupied"]) == BOARD_SIZE * BOARD_SIZE:  # Draw
            self.result = 3
            reward += REWARD_DRAW

        done = (win or len(self.info["Occupied"]) == BOARD_SIZE * BOARD_SIZE)
        self.current_player = self.current_player + 1 if self.current_player == 1 else 1
        state = self.decompose_board_to_state()

        observations_and_legal_moves = {'state': state, 'legal_moves': self.info['legal_moves']}

        if done:
            return ts.termination(observations_and_legal_moves, reward)
        else:
            return ts.transition(observations_and_legal_moves, reward)

    def detect_alive_two(self, row: int, col: int) -> int:
        """ Detect how many alive_two can obtain by this play out.

        Args:
            row (int): row of the position to be placed
            col (int): column of the position to be placed

        Returns:
            cnt (int): the number of alive_two
        """
        cnt = 0
        adjs = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [0, -1], [1, -1], [1, 0], [1, 1]]
        for adj in adjs:
            p1_r, p1_c = row + 2 * adj[0], col + 2 * adj[1]
            p2_r, p2_c = row - adj[0], col - adj[1]
            if 0 <= p1_r < 9 and 0 <= p1_c < 9 and 0 <= p2_r < 9 and 0 <= p2_c < 9:
                if self.board[row + adj[0], col + adj[1]] == self.current_player and self.board[p1_r, p1_c] == 0 and \
                        self.board[p2_r, p2_c] == 0:
                    cnt += 1
        return cnt

    def detect_three(self, r: int, c: int) -> tuple:
        """ Detect how many non_active_three and active_three can obtain by this move.

        Args:
            r (int): row of the current move
            c (int): column of the current move

        Returns:
            cnt_non_act (int): the number of non_active_three
            cnt_act (int): the number of active_three
        """
        cnt_non_act = 0
        cnt_act = 0
        directions = [[0, 1], [1, 0], [1, 1], [1, -1]]

        for direct in directions:
            count = 0
            for offset in range(-2, 3):
                if 0 <= r + offset * direct[0] < 9 and 0 <= c + offset * direct[1] < 9:
                    if self.board[r + offset * direct[0], c + offset * direct[1]] == self.current_player or offset == 0:
                        count += 1
                        if count == 3:
                            p1_r, p1_c = r + (offset + 1) * direct[0], c + (offset + 1) * direct[1]
                            p2_r, p2_c = r + (offset - 3) * direct[0], c + (offset - 3) * direct[1]

                            p1_is_empty = (0 <= p1_r < 9 and 0 <= p1_c < 9 and self.board[p1_r, p1_c] == 0)
                            p2_is_empty = (0 <= p2_r < 9 and 0 <= p2_c < 9 and self.board[p2_r, p2_c] == 0)

                            if p1_is_empty and p2_is_empty:
                                cnt_act += 1
                            elif p1_is_empty or p2_is_empty:
                                cnt_non_act += 1
                            break
                    else:
                        count = 0

        return cnt_non_act, cnt_act

    def choose_adj_pos(self, row: int, col: int) -> tuple:
        """ Randomly select an adjacent position with equal probabilities.

        Args:
            row (int): row of the current move
            col (int): column of the current move

        Returns:
            row (int): row of the selected adjacent position
            col (int): column of the selected adjacent position
        """

        adjs = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [0, -1], [1, -1], [1, 0], [1, 1]]
        adj = random.choice(adjs)
        row, col = row + adj[0], col + adj[1]
        return row, col

    def _is_win(self, r: int, c: int) -> bool:
        """check if this player results in a winner

        Args:
            r (int): row of the current move
            c (int): column of the current move

        Returns:
            bool: indicating if there is a winner
        """

        # check if four equal stones are aligned (horizontal, vertical or diagonal)
        directions = [[0, 1], [1, 0], [1, 1], [1, -1]]

        for direct in directions:
            count = 0
            for offset in range(-3, 4):
                if 0 <= r + offset * direct[0] < 9 and 0 <= c + offset * direct[1] < 9:
                    if self.board[r + offset * direct[0], c + offset * direct[1]] == self.current_player or offset == 0:
                        count += 1
                        if count == 4:
                            return True
                    else:
                        count = 0

        return False

    def decode_action(self, action: int) -> List[int]:
        """decode the action integer into a colum and row value

        0 = upper left corner
        8 = lower right corner

        Args:
            action (int): action

        Returns:
            List[int, int]: a list with the [row, col] values
        """
        col = action % BOARD_SIZE
        row = action // BOARD_SIZE
        assert 0 <= col < BOARD_SIZE
        return [row, col]

    def render(self, render_mode="rgb_array") -> np.ndarray:
        """Render the board
        Print a string that shows the current board, if render_mode == human,
        Return the RGB array of a figure which shows the current board.
        """
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=str)
        for ii in range(BOARD_SIZE):
            for jj in range(BOARD_SIZE):
                if self.board[ii, jj] == 0:
                    board[ii, jj] = "-"
                elif self.board[ii, jj] == 1:
                    board[ii, jj] = "X"
                elif self.board[ii, jj] == 2:
                    board[ii, jj] = "O"

        if render_mode == "human":
            board = tabulate(board, tablefmt="fancy_grid")
            print(board)
            print("\n")

        width = height = 400

        white = (255, 255, 255)
        line_color = (0, 0, 0)
        red = (255, 0, 0)

        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pg.init()

        # Set up the drawing window
        if self.screen is None:
            self.screen = pg.display.set_mode([width + 16, height + 16])

        self.screen.fill(white)
        # drawing vertical lines
        for i in range(10):
            pg.draw.line(self.screen, line_color, (width / BOARD_SIZE * i, 0), (width / BOARD_SIZE * i, height), 2)

        # drawing horizontal lines
        for i in range(10):
            pg.draw.line(self.screen, line_color, (0, height / BOARD_SIZE * i), (width, height / BOARD_SIZE * i), 2)
        pg.display.flip()

        latest_row, latest_col = -1, -1
        if self.latest_action:
            latest_row, latest_col = self.decode_action(self.latest_action)

        # drawing noughts and crosses
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                color = line_color
                if latest_row == i and latest_col == j:
                    color = red
                if self.board[i, j] == 1:  # Draw crosses
                    pg.draw.lines(self.screen, color, True, [(width / BOARD_SIZE * (j + 0.5) - 10,
                                                              height / BOARD_SIZE * (i + 0.5) - 10),
                                                             (width / BOARD_SIZE * (j + 0.5) + 10,
                                                              height / BOARD_SIZE * (i + 0.5) + 10)], 3)
                    pg.draw.lines(self.screen, color, True, [(width / BOARD_SIZE * (j + 0.5) - 10,
                                                              height / BOARD_SIZE * (i + 0.5) + 10),
                                                             (width / BOARD_SIZE * (j + 0.5) + 10,
                                                              height / BOARD_SIZE * (i + 0.5) - 10)], 3)
                elif self.board[i, j] == 2:  # Draw noughts
                    pg.draw.circle(self.screen, color,
                                   (width / BOARD_SIZE * (j + 0.5), height / BOARD_SIZE * (i + 0.5)), 12, 3)

        board = np.transpose(
            np.array(pg.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )

        return board
