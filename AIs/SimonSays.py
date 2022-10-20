import numpy as np
import random


class SimonSays:
    def __init__(self, epochs):
        self.status = "Stopped"
        self.simon = False
        self.score = 0
        self.current_round = 0
        self.choice = 0
        self.correct_choice = 0
        self.epochs = epochs

    def call_simon(self) -> None:
        self.simon = bool(random.getrandbits(1))

    def random_choice(self) -> None:
        self.correct_choice = random.randint(1, 4)

    def run(self) -> None:
        self.status = "Running"

    def return_status(self) -> str:
        return self.status

    def train(self) -> None:
        pass

    def run_round(self) -> None:
        self.current_round = 0
        self.score = 0

        self.call_simon()
        self.random_choice()

        if self.choice == 0 and not self.simon:
            score = 10
        elif self.choice == self.correct_choice and self.simon:
            score = 10
        elif self.choice == self.correct_choice and not self.simon:
            score = 1
        else:
            score = 0
