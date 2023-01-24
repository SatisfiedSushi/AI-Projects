import numpy as np
from random import *

class not_gate:
    def run_game(self):
        input = [randint(0, 1)]
        actual = [0 if input[0] == 1 else 1]

        return input, actual

class no_change():
    def run_game(self):
        input = [randint(0, 1), randint(0, 1)]
        actual = input.copy()
        return input, actual

class tic_tac_toe:
    def __init__(self):
        self.game_board = [0, 0, 0,
                           0, 0, 0,
                           0, 0, 0]

        self.turn = 0

    def restart_game(self):
        self.turn = 1
        self.game_board = [0, 0, 0,
                           0, 0, 0,
                           0, 0, 0]


    def check_for_winner(self):
        winner = 0

        # horizontal
        if len(set([i for i in self.game_board if self.game_board.index(i) <= 2])) == self.game_board[0]:
            winner = self.game_board[0]
        elif len(set([i for i in self.game_board if self.game_board.index(i) >= 3 and self.game_board.index(i) <= 5])) == self.game_board[3]:
            winner = self.game_board[3]
        elif len(set([i for i in self.game_board if self.game_board.index(i) >= 6])) == self.game_board[6]:
            winner = self.game_board[6]

        # vertical
        transposed_game_board = [self.game_board[0], self.game_board[3], self.game_board[6],
                                 self.game_board[1], self.game_board[4], self.game_board[7],
                                 self.game_board[2], self.game_board[5], self.game_board[8]]
        if len(set([i for i in transposed_game_board if transposed_game_board.index(i) <= 2])) == transposed_game_board[0]:
            winner = transposed_game_board[0]
        elif len(set([i for i in transposed_game_board if transposed_game_board.index(i) >= 3 and transposed_game_board.index(i) <= 5])) == transposed_game_board[3]:
            winner = transposed_game_board[3]
        elif len(set([i for i in transposed_game_board if transposed_game_board.index(i) >= 6])) == transposed_game_board[6]:
            winner = transposed_game_board[6]

        # diagonal
        if len(set([self.game_board[0], self.game_board[4], self.game_board[8]])) == self.game_board[4] or len(set([self.game_board[2], self.game_board[4], self.game_board[6]])) == self.game_board[4]:
            winner = self.game_board[4]

        self.restart_game()
        return winner

    def make_move(self, player: int, move: int):
        self.game_board[move - 1] = player
        self.turn += 1

    def move_options(self):
        return [self.game_board.index(i) for i in self.game_board if self.game_board[i] == 0]







