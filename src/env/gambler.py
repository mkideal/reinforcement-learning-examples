'''Gambler's Problem
A gambler has the opportunity to make bets on the outcomes of a sequence of coin flips.
If the coin comes up heads, he wins as many dollars as he has staked on that flip; if
it is tails, he loses his stake. The game ends when the gambler wins by reaching his
goal of `$100`, or loses by running out of money. On each flop, the gambler must decide
what portion of his capital to stake, in integer numbers of dollars. This problem can
formulated as an undiscounted, episodic, finite MDP.
'''


class Gambler:
    COINS = 100

    '''required methods as an Env
    '''

    def __init__(self, ph=0.5):
        self.p_head = ph
        if ph <= 0 or ph >= 1:
            raise Exception("invalid ph")

    def gamma(self):
        return 1

    def num_states(self):
        return self.COINS + 1

    def is_terminal(self, state):
        return state == 0 or state == self.COINS

    def get_actions(self, state):
        if self.is_terminal(state):
            return None
        max_bet = min(state, self.COINS - state)
        actions = []
        for i in range(max_bet):
            bet = i + 1
            actions.append(bet)
        return actions

    def step_enum(self, state, action):
        reward = 0
        return [
            (state + action, self.transition_reward(state + action), self.p_head),
            (state - action, self.transition_reward(state - action), 1 - self.p_head)
        ]

    def default_policy(self):
        policy = {}
        for state in range(self.num_states()):
            policy[state] = {}
            actions = self.get_actions(state)
            if actions is not None:
                prob = 1.0 / len(actions)
                for action in actions:
                    policy[state][action] = prob
        return policy

    '''optional methods
    '''

    def transition_reward(self, next_state):
        return 1 if next_state == self.COINS else 0
