import pdb
import unittest

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


if __name__ == '__main__':
    unittest.main()
