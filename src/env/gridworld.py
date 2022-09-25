from enum import Enum

'''Example 1: 4x4 gridworld shown below:

|   | 1 | 2 | 3 |
|---|---|---|---|
| 4 | 5 | 6 | 7 |
| 8 | 9 | 10| 11|
| 12| 13| 14|   |

1. Rt = 1 on all transitions.
2. The nonterminal states S = {1,2,...,14}.
3. There are four actions A = {up, down, left, right}, which deterministically cause the
   corresponding state transitions, except that actions would take the agent off the grid
   in fact leave the state unchanged.
4. This is an undiscounted, episodic task, and the terminal states are the empty grids.

Values for equiprobable random policy:

| 0 |-14|-20|-22|
|---|---|---|---|
|-14|-18|-20|-20|
|-20|-20|-18|-14|
|-22|-20|-14| 0 |

policy improvment policy:

|   | ← | ← | ← |
|---|---|---|---|
| ↑ | ← | ← | ↓ |
| ↑ | ↑ | ↓ | ↓ |
| ↑ | → | → |   |

policy improvment values:

| 0 | -1| -2| -3|
|---|---|---|---|
| -1| -2| -3| -2|
| -2| -3| -2| -1|
| -3| -2| -1| 0 |

value iteraion policy:

|   | ← | ← | ↓ |
|---|---|---|---|
| ↑ | ↑ | ↑ | ↓ |
| ↑ | ↑ | ↓ | ↓ |
| ↑ | → | → |   |

value iteraion values:

| 0 | -1| -2| -3|
|---|---|---|---|
| -1| -2| -3| -2|
| -2| -3| -2| -1|
| -3| -2| -1| 0 |
'''


class GridWorld:
    class Action(Enum):
        UP = 1
        DOWN = 2
        LEFT = 3
        RIGHT = 4

    def __init__(self, rows, columns=None):
        self.rows = rows
        self.columns = columns if columns is not None else rows

    '''required methods as an Env
    '''

    def gamma(self):
        return 1

    def num_states(self):
        return self.rows * self.columns

    def is_terminal(self, state):
        return state == 0 or state == self.num_states() - 1

    def get_actions(self, state):
        return None if self.is_terminal(state) else [
            self.Action.UP,
            self.Action.DOWN,
            self.Action.LEFT,
            self.Action.RIGHT
        ]

    def step_enum(self, state, action):
        column = int(state % self.columns)
        row = int((state - column) / self.columns)
        reward = -1
        if action == self.Action.LEFT:
            return [(state if column == 0 else self.make_state(row, column-1), reward, 1)]
        elif action == self.Action.RIGHT:
            return [(state if column+1 == self.columns else self.make_state(row, column+1), reward, 1)]
        elif action == self.Action.UP:
            return [(state if row == 0 else self.make_state(row-1, column), reward, 1)]
        elif action == self.Action.DOWN:
            return [(state if row+1 == self.rows else self.make_state(row+1, column), reward, 1)]
        else:
            raise Exception("unexpected action")

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

    def make_state(self, row, column):
        return row * self.columns + column

    def print_values(self, title, values, formatter=None):
        print(title)
        for i in range(self.rows):
            print('|', end='')
            for j in range(self.columns):
                state = self.make_state(i, j)
                if formatter is not None:
                    print("%s|" % formatter(values[state]), end='')
                else:
                    print("%g|" % round(values[state], 2), end='')
            print()
            if (i == 0):
                print('|', end='')
                for j in range(self.columns):
                    print('---|', end='')
                print()
