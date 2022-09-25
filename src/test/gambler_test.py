from lib2to3.pgen2 import grammar
import sys
sys.path.append("../../")

import matplotlib.pyplot as plt
from src.env import gambler
from src.agent import dp


def gambler_value_iteration(ph):
    env = grammar.Gambler(ph)
    agent = dp.DynamicPrograming()
    (policy, values) = agent.value_iteration(env)
    # show values and policy
    print(f"gamber result for {ph}")
    states = [i for i in range(1, env.num_states() - 1)]
    stakes = [0] * (env.num_states() - 2)
    for state in states:
        for action in policy[state]:
            stakes[state-1] = action
            break
    fig, ax1 = plt.subplots()
    ax1.plot(states, values[1: env.num_states() - 1])
    fig, ax2 = plt.subplots()
    ax2.plot(states, stakes)
    plt.show()


if __name__ == "__main__":
    gambler_value_iteration(0.25)
    gambler_value_iteration(0.40)
