import random


class MonteCarlo:
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

    def evaluate(self, env, begin, iterations, policy, behavior=None):
        off = behavior is not None
        if not off:
            behavior = policy
        gamma = env.gamma()
        values = {}
        visits = {}
        for i in range(iterations):
            state = begin(env)
            seen = {}
            episode = []
            # generate one episode
            while not env.is_terminal(state):
                if state not in seen:
                    seen[str(state)] = len(episode)
                action = self.make_decision(policy)
                next_state, reward = env.step(state, action)
                episode.append((state, reward))
                state = next_state
            # update values for each state in the episode
            returns = 0
            for i in range(len(episode) - 1, -1, -1):
                (state, reward) = episode[i]
                returns = reward + gamma * returns
                state_key = str(state)
                if seen[state_key] == i:
                    old_value = 0 if state_key not in values else values[state_key]
                    old_visit = 0 if state_key not in visits else visits[state_key]
                    values[state_key] = (
                        old_value * old_visit + returns) / (old_visit + 1)
                    visits[state_key] = old_visit + 1
        return values
