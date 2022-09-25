import random


class DynamicPrograming:
    def compute_value(self, env, policy, values, state):
        value = 0
        gamma = env.gamma()
        for action, pi in policy[state].items():
            for (next_state, reward, prob) in env.step_enum(state=state, action=action):
                value += pi * prob * (reward + gamma * values[next_state])
        return value

    def compute_value_from_qvalues(self, env, policy, qvalues, state):
        value = 0
        for action, pi in policy[state].items():
            value += pi * qvalues[state][action]
        return value

    def compute_qvalue(self, env, policy, qvalues, state, action):
        value = 0
        gamma = env.gamma()
        for (next_state, reward, prob) in env.step_enum(state=state, action=action):
            value += prob * reward
            for next_action, pi in policy[next_state].items():
                value += prob * gamma * pi * qvalues[next_state][next_action]
        return value

    def compute_qvalue_from_values(self, env, values, state, action):
        value = 0
        gamma = env.gamma()
        for (next_state, reward, prob) in env.step_enum(state=state, action=action):
            value += prob * (reward + gamma * values[next_state])
        return value

    def compute_max_value(self, env, values, state):
        max_value = None
        max_action = None
        actions = env.get_actions(state)
        if actions is None:
            return (None, None)
        for action in actions:
            value = self.compute_qvalue_from_values(env, values, state, action)
            if max_value is None or value > max_value:
                max_value = value
                max_action = action
        return (max_value, max_action)

    def make_decision(self, policy, state):
        sum = 0
        actions = []
        for action, pi in policy[state].items():
            sum += pi
            actions.append((action, sum))
        sample = random.random() * sum
        for (action, acc) in actions:
            if sample < acc:
                return action
        return None

    def evaluate(self, env, policy, values):
        max_delta = 0
        for state in range(len(values)):
            new_value = self.compute_value(env, policy, values, state)
            delta = abs(new_value - values[state])
            max_delta = max(delta, max_delta)
            values[state] = new_value
        return max_delta

    def qevaluate(self, env, policy, qvalues):
        max_delta = 0
        for state in range(len(qvalues)):
            for action in qvalues[state]:
                new_value = self.compute_qvalue(
                    env, policy, qvalues, state, action)
                delta = abs(new_value - qvalues[state][action])
                max_delta = max(delta, max_delta)
                qvalues[state][action] = new_value
        return max_delta

    def policy_improvment(self, env, policy, values):
        stable = True
        for state in range(len(values)):
            action = self.make_decision(policy, state)
            if action is None:
                continue
            max_value = None
            max_action = action
            for action, prob in policy[state].items():
                value = self.compute_qvalue_from_values(
                    env, values, state, action)
                if max_value is None or value > max_value:
                    max_value = value
                    max_action = action
            policy[state] = {}
            policy[state][max_action] = 1
            if action != max_action:
                stable = False
        return stable

    def policy_iteration(self, env):
        policy = env.default_policy()
        values = [0] * env.num_states()
        theta = 0.001
        stable = False
        times = 0
        while not stable:
            # policy evaluation
            delta = theta+1
            while delta > theta:
                delta = self.evaluate(env, policy, values)
            # policy improvment
            stable = self.policy_improvment(env, policy, values)
            times += 1
        return (policy, values)

    def value_iteration(self, env):
        values = [0] * env.num_states()

        # value iteration
        theta = 0.001
        delta = theta+1
        while delta > theta:
            delta = 0
            for state in range(len(values)):
                value = values[state]
                new_value, max_action = self.compute_max_value(
                    env, values, state)
                if max_action is None:
                    continue
                delta = max(delta, abs(new_value - value))
                values[state] = new_value

        # policy selection
        policy = [None] * env.num_states()
        for state in range(len(values)):
            policy[state] = {}
            max_value, max_action = self.compute_max_value(env, values, state)
            if max_action is not None:
                policy[state][max_action] = 1

        return (policy, values)
