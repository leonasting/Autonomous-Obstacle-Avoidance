# 252-Fall-21-Q-learning-Grid

**Problem Description:**
An agent is being trained in a model-free setting to navigate in a grid environment to reach the goal avoiding the bombs which are present at the random location. The task of the agent is to find the optimal policy using Q-Learning and SARSA algorithms.

**Proposed Solution:**
Set up the grid environment by initializing the following parameters:
Grid size, start position, goal position, reward, bomb count, actions.

**Implement:**
Q learning and SARSA algorithms to find the optimal policy from the start position to the end goal

**Observations:**
Q-Learning focuses on exploitation thus reaching a possible optimal path very early. With SARSA the agent explores the same path it starts with and might not come up with
an optimal path.Changing the starting point from fixed to random might improve performance but will require a lot of episodes to learn about the path.

**Conclusion:**
As we can infer from the graphs, the performance of Q learning is much better than SARSA because in Q Learning, the Q values are derived from the best action in the next state whereas in SARSA the Q values are derived from the actual next state and the action. SARSA is taking more episodes to reach the optimal policy when compared to Q Learning.
