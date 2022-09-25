import random
from enum import Enum


class Blackjack:
    class Action(Enum):
        HITS = 1
        STICKS = 2

    class State:
        def __init__(self):
            self.cards = []
            self.dealer_cards = []

    def __init__(self, dealer_policy=None):
        self.dealer_policy = dealer_policy if dealer_policy is not None else self.dealer_policy
        self.deck = [i for i in range(52)]
        self.cursor = 0

    def is_terminal(self, state):
        return 0

    def begin(self):
        # reset deck
        random.shuffle(self.deck)
        self.cursor = 0

        state = Blackjack.State()
        state.cards = [self.deal(), self.deal()]
        state.dealer_cards = [self.deal(), self.deal()]

        return state

    def step(self, state, action):
        pass

    def deal(self):
        card = self.deck[self.cursor]
        self.cursor += 1
        return card

    def default_dealer_policy(self, state):
        pass
