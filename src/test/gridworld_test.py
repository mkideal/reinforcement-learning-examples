import sys
sys.path.append("../../")


from src.env import gridworld
from src.agent import dp

ROWS = 4
COLUMNS = ROWS

def gridworld_evaluate():
    env = gridworld.GridWorld(ROWS, COLUMNS)
    agent = dp.DynamicPrograming()
    values = [0] * (env.num_states())
    max_times = 100000
    times = 0
    theta = 0.0001
    delta = 1
    policy = env.default_policy()
    while delta > theta and times < max_times:
        times += 1
        delta = agent.evaluate(env, policy, values)
    env.print_values("evaluate values:", values)

    delta = 1
    times = 0
    qvalues = []
    for state in range(env.num_states()):
        qvalues.append({action: 0 for action in policy[state]})
    while delta > theta and times < max_times:
        times += 1
        delta = agent.qevaluate(env, policy, qvalues)
    print("evaluate qvalues: delta=%.6f, times=%d, q(11,down)=%g, q(7,down)=%g" % (
        delta,
        times,
        round(qvalues[11][env.Action.DOWN], 2),
        round(qvalues[7][env.Action.DOWN], 2)
    ))


def format_policy(policy):
    for action in policy:
        if action == gridworld.GridWorld.Action.LEFT:
            return "←"
        elif action == gridworld.GridWorld.Action.RIGHT:
            return "→"
        elif action == gridworld.GridWorld.Action.UP:
            return "↑"
        elif action == gridworld.GridWorld.Action.DOWN:
            return "↓"
        else:
            return ""
    return ""


def gridworld_policy_improvment():
    env = gridworld.GridWorld(ROWS, COLUMNS)
    agent = dp.DynamicPrograming()
    (policy, values) = agent.policy_iteration(env)
    env.print_values("policy improvment policy:", policy, format_policy)
    env.print_values("policy improvment values:", values)


def gridworld_value_iteration():
    env = gridworld.GridWorld(ROWS, COLUMNS)
    agent = dp.DynamicPrograming()
    (policy, values) = agent.value_iteration(env)
    env.print_values("value iteraion policy:", policy, format_policy)
    env.print_values("value iteraion values:", values)


if __name__ == "__main__":
    gridworld_evaluate()
    gridworld_policy_improvment()
    gridworld_value_iteration()
