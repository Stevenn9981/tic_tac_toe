import pdb
import unittest

from codes.part1.src.PlayPolicy import drop_here_will_win
from codes.part1.src.TicTacToeEnv1 import *


class TestPart1(unittest.TestCase):

    def setUp(self):
        self.env = TicTacToeEnv1()

    def test_if_chess_nearby(self):
        self.env._reset()
        self.assertFalse(self.env.if_chess_nearby(4, 4))
        self.assertFalse(self.env.if_chess_nearby(5, 5))

        self.env.board = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 2, 0, 0, 0],
                                   [0, 0, 0, 0, 2, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0]])

        self.assertTrue(self.env.if_chess_nearby(3, 4))
        self.assertTrue(self.env.if_chess_nearby(5, 4))
        self.assertTrue(self.env.if_chess_nearby(6, 4))
        self.assertTrue(self.env.if_chess_nearby(7, 4))
        self.assertFalse(self.env.if_chess_nearby(8, 4))
        self.assertFalse(self.env.if_chess_nearby(6, 1))
        self.assertTrue(self.env.if_chess_nearby(6, 2))
        self.assertTrue(self.env.if_chess_nearby(6, 3))
        self.assertTrue(self.env.if_chess_nearby(6, 4))
        self.assertTrue(self.env.if_chess_nearby(6, 5))
        self.assertTrue(self.env.if_chess_nearby(6, 6))
        self.assertTrue(self.env.if_chess_nearby(6, 7))
        self.assertFalse(self.env.if_chess_nearby(6, 8))

    def test_decompose_board_to_state(self):
        self.env._reset()
        self.assertTrue(np.equal(self.env.decompose_board_to_state(), np.stack([
            np.zeros((BOARD_SIZE, BOARD_SIZE)),
            np.zeros((BOARD_SIZE, BOARD_SIZE)),
            np.zeros((BOARD_SIZE, BOARD_SIZE)),
            np.ones((BOARD_SIZE, BOARD_SIZE)),
            np.zeros((BOARD_SIZE, BOARD_SIZE))
        ], axis=2)).all())

        self.env.board = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 2, 0, 0, 0],
                                   [0, 0, 0, 0, 2, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.env.current_player = 1
        self.env.latest_action = 32
        state = self.env.decompose_board_to_state()

        o_plays = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        c_plays = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        l_play = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        if_first = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1]])
        adj_play = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1, 1, 1, 1, 0],
                             [0, 1, 1, 0, 1, 0, 1, 1, 0],
                             [0, 1, 1, 1, 0, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1, 0, 1, 1, 0],
                             [0, 0, 1, 1, 1, 1, 1, 1, 0],
                             [0, 0, 0, 1, 1, 1, 1, 1, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.assertTrue(np.equal(state, np.stack([o_plays, c_plays, l_play, if_first, adj_play], axis=2)).all())

    def test_detect_alive_two(self):
        self.env._reset()
        self.assertEqual(self.env.detect_alive_two(1, 2), 0)
        self.assertEqual(self.env.detect_alive_two(4, 4), 0)
        self.assertEqual(self.env.detect_alive_two(8, 6), 0)

        self.env.board = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 2, 0, 0, 0],
                                   [0, 0, 1, 0, 2, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.env.current_player = 2
        self.assertEqual(self.env.detect_alive_two(4, 3), 0)
        self.assertEqual(self.env.detect_alive_two(5, 3), 0)
        self.assertEqual(self.env.detect_alive_two(5, 4), 0)
        self.assertEqual(self.env.detect_alive_two(5, 5), 0)
        self.assertEqual(self.env.detect_alive_two(2, 4), 1)
        self.assertEqual(self.env.detect_alive_two(2, 5), 1)
        self.assertEqual(self.env.detect_alive_two(3, 4), 1)
        self.assertEqual(self.env.detect_alive_two(4, 5), 2)

        self.env.board = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 2, 0, 0, 0],
                                   [0, 0, 1, 0, 2, 2, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.env.current_player = 1
        self.assertEqual(self.env.detect_alive_two(3, 1), 0)
        self.assertEqual(self.env.detect_alive_two(3, 2), 2)
        self.assertEqual(self.env.detect_alive_two(4, 1), 1)
        self.assertEqual(self.env.detect_alive_two(4, 3), 0)
        self.assertEqual(self.env.detect_alive_two(5, 1), 0)
        self.assertEqual(self.env.detect_alive_two(5, 2), 3)
        self.assertEqual(self.env.detect_alive_two(5, 4), 1)
        self.assertEqual(self.env.detect_alive_two(6, 4), 1)

    def test_detect_three(self):
        self.env._reset()
        cnt_non_act, cnt_act = self.env.detect_three(1, 2)
        self.assertEqual(cnt_non_act, 0)
        self.assertEqual(cnt_act, 0)
        cnt_non_act, cnt_act = self.env.detect_three(4, 4)
        self.assertEqual(cnt_non_act, 0)
        self.assertEqual(cnt_act, 0)
        cnt_non_act, cnt_act = self.env.detect_three(8, 6)
        self.assertEqual(cnt_non_act, 0)
        self.assertEqual(cnt_act, 0)

        self.env.board = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 2, 0, 0, 0],
                                   [0, 0, 1, 0, 2, 0, 2, 0, 0],
                                   [0, 0, 0, 1, 0, 2, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.env.current_player = 2
        cnt_non_act, cnt_act = self.env.detect_three(1, 2)
        self.assertEqual(cnt_non_act, 0)
        self.assertEqual(cnt_act, 0)
        cnt_non_act, cnt_act = self.env.detect_three(2, 4)
        self.assertEqual(cnt_non_act, 0)
        self.assertEqual(cnt_act, 1)
        cnt_non_act, cnt_act = self.env.detect_three(2, 6)
        self.assertEqual(cnt_non_act, 1)
        self.assertEqual(cnt_act, 0)
        cnt_non_act, cnt_act = self.env.detect_three(3, 7)
        self.assertEqual(cnt_non_act, 1)
        self.assertEqual(cnt_act, 0)
        cnt_non_act, cnt_act = self.env.detect_three(4, 5)
        self.assertEqual(cnt_non_act, 0)
        self.assertEqual(cnt_act, 2)
        cnt_non_act, cnt_act = self.env.detect_three(5, 7)
        self.assertEqual(cnt_non_act, 0)
        self.assertEqual(cnt_act, 1)
        cnt_non_act, cnt_act = self.env.detect_three(6, 6)
        self.assertEqual(cnt_non_act, 1)
        self.assertEqual(cnt_act, 0)

        self.env.board = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 0, 0, 0, 0],
                                   [0, 0, 1, 0, 2, 2, 0, 0, 0],
                                   [0, 0, 1, 1, 2, 2, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 2, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.env.current_player = 1
        cnt_non_act, cnt_act = self.env.detect_three(2, 2)
        self.assertEqual(cnt_non_act, 0)
        self.assertEqual(cnt_act, 0)
        cnt_non_act, cnt_act = self.env.detect_three(3, 2)
        self.assertEqual(cnt_non_act, 0)
        self.assertEqual(cnt_act, 2)
        cnt_non_act, cnt_act = self.env.detect_three(3, 5)
        self.assertEqual(cnt_non_act, 0)
        self.assertEqual(cnt_act, 1)
        cnt_non_act, cnt_act = self.env.detect_three(4, 3)
        self.assertEqual(cnt_non_act, 0)
        self.assertEqual(cnt_act, 2)
        cnt_non_act, cnt_act = self.env.detect_three(5, 1)
        self.assertEqual(cnt_non_act, 1)
        self.assertEqual(cnt_act, 1)
        cnt_non_act, cnt_act = self.env.detect_three(6, 2)
        self.assertEqual(cnt_non_act, 0)
        self.assertEqual(cnt_act, 1)
        cnt_non_act, cnt_act = self.env.detect_three(6, 3)
        self.assertEqual(cnt_non_act, 0)
        self.assertEqual(cnt_act, 0)
        cnt_non_act, cnt_act = self.env.detect_three(6, 4)
        self.assertEqual(cnt_non_act, 0)
        self.assertEqual(cnt_act, 1)

    def test_win(self):
        self.env._reset()
        self.env.current_player = 1
        self.assertEqual(self.env._is_win(1, 2), False)
        self.assertEqual(self.env._is_win(4, 4), False)
        self.env.current_player = 2
        self.assertEqual(self.env._is_win(8, 6), False)

        self.env.board = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 2, 0, 0, 0],
                                   [0, 0, 1, 1, 2, 2, 2, 0, 0],
                                   [0, 0, 0, 1, 0, 2, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.env.current_player = 1
        self.assertEqual(self.env._is_win(2, 2), False)
        self.assertEqual(self.env._is_win(2, 3), True)
        self.assertEqual(self.env._is_win(2, 4), False)
        self.assertEqual(self.env._is_win(3, 1), True)
        self.assertEqual(self.env._is_win(3, 2), False)
        self.assertEqual(self.env._is_win(4, 7), False)
        self.assertEqual(self.env._is_win(6, 3), True)
        self.assertEqual(self.env._is_win(6, 5), False)
        self.assertEqual(self.env._is_win(7, 5), True)
        self.env.current_player = 2
        self.assertEqual(self.env._is_win(2, 4), False)
        self.assertEqual(self.env._is_win(2, 5), True)
        self.assertEqual(self.env._is_win(3, 1), False)
        self.assertEqual(self.env._is_win(3, 6), False)
        self.assertEqual(self.env._is_win(4, 7), True)
        self.assertEqual(self.env._is_win(5, 6), False)
        self.assertEqual(self.env._is_win(6, 5), True)

        self.env.board = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 2, 0, 0, 0],
                                   [0, 0, 1, 0, 2, 2, 2, 0, 0],
                                   [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.env.current_player = 1
        self.assertEqual(self.env._is_win(2, 3), False)
        self.assertEqual(self.env._is_win(4, 3), True)
        self.assertEqual(self.env._is_win(4, 7), False)
        self.assertEqual(self.env._is_win(5, 2), False)
        self.env.current_player = 2
        self.assertEqual(self.env._is_win(4, 3), True)
        self.assertEqual(self.env._is_win(3, 4), False)
        self.assertEqual(self.env._is_win(4, 7), True)
        self.assertEqual(self.env._is_win(5, 5), False)

    def test_decode_action(self):
        self.assertEqual(self.env.decode_action(6), [0, 6])
        self.assertEqual(self.env.decode_action(16), [1, 7])
        self.assertEqual(self.env.decode_action(25), [2, 7])
        self.assertEqual(self.env.decode_action(45), [5, 0])
        self.assertEqual(self.env.decode_action(64), [7, 1])
        self.assertEqual(self.env.decode_action(80), [8, 8])

    def test_step(self):
        self.env.board = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 2, 0, 0, 0],
                                   [0, 0, 1, 0, 2, 2, 2, 0, 0],
                                   [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        with self.assertRaises(ValueError):
            self.env.step(0)
        with self.assertRaises(ValueError):
            self.env.step(30)

        board = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 2, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 2, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        equal = True
        for _ in range(10):
            self.env._reset()
            self.env.step(20)
            self.env.step(23)
            self.env.step(47)
            self.env.step(50)
            equal = equal and np.equal(self.env.board, board).all()
        self.assertFalse(equal)

        equal = True
        train_env = TicTacToeEnv1(train=True)
        for _ in range(10):
            train_env._reset()
            train_env.step(20)
            train_env.step(23)
            train_env.step(47)
            train_env.step(50)
            equal = equal and np.equal(train_env.board, board).all()
        self.assertTrue(equal)


    def test_drop_here_will_win(self):
        self.env._reset()
        self.assertEqual(drop_here_will_win(self.env, 11, 1), False)
        self.assertEqual(drop_here_will_win(self.env, 40, 1), False)
        self.assertEqual(drop_here_will_win(self.env, 78, 1), False)

        self.env.board = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 2, 0, 0, 0],
                                   [0, 0, 1, 1, 2, 2, 2, 0, 0],
                                   [0, 0, 0, 1, 0, 2, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.assertEqual(drop_here_will_win(self.env, 20, 1), False)
        self.assertEqual(drop_here_will_win(self.env, 21, 1), True)
        self.assertEqual(drop_here_will_win(self.env, 22, 1), False)
        self.assertEqual(drop_here_will_win(self.env, 28, 1), True)
        self.assertEqual(drop_here_will_win(self.env, 29, 1), False)
        self.assertEqual(drop_here_will_win(self.env, 43, 1), False)
        self.assertEqual(drop_here_will_win(self.env, 57, 1), True)
        self.assertEqual(drop_here_will_win(self.env, 59, 1), False)
        self.assertEqual(drop_here_will_win(self.env, 68, 1), True)

        self.assertEqual(drop_here_will_win(self.env, 22, 2), False)
        self.assertEqual(drop_here_will_win(self.env, 23, 2), True)
        self.assertEqual(drop_here_will_win(self.env, 28, 2), False)
        self.assertEqual(drop_here_will_win(self.env, 33, 2), False)
        self.assertEqual(drop_here_will_win(self.env, 43, 2), True)
        self.assertEqual(drop_here_will_win(self.env, 51, 2), False)
        self.assertEqual(drop_here_will_win(self.env, 59, 2), True)

        self.env.board = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 2, 0, 0, 0],
                                   [0, 0, 1, 0, 2, 2, 2, 0, 0],
                                   [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0]])

        self.assertEqual(drop_here_will_win(self.env, 21, 1), False)
        self.assertEqual(drop_here_will_win(self.env, 39, 1), True)
        self.assertEqual(drop_here_will_win(self.env, 43, 1), False)
        self.assertEqual(drop_here_will_win(self.env, 47, 1), False)

        self.assertEqual(drop_here_will_win(self.env, 39, 2), True)
        self.assertEqual(drop_here_will_win(self.env, 31, 2), False)
        self.assertEqual(drop_here_will_win(self.env, 43, 2), True)
        self.assertEqual(drop_here_will_win(self.env, 50, 2), False)


if __name__ == '__main__':
    unittest.main()
