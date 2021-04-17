# Observations

## lake

15x15 gridworld

### VI/PI

- run fairly quickly
- both align for max v plot
- high gamma to allow long term learning/"planning for future rewards"
- low epsilon for convergence criterion to wait for change to get extremely small to stop

### QL

- takes a lot of iterations..
    - cutting at 1m for time concerns for others
- time and value increase linearly after 500k
    - even at 5m, it only gets to about 0.8, whereas pi/vi are at around 2.8
- would need an infeasible number of iterations to perform comparably well
- would benefit from reward shaping, i.e., small negative rewards on each step to encourage quick completion
as well as negative rewards for holes, rather than nothing
    - this was not done mainly for time/implementation concerns
- 0.6 seems to be best gamma at 1m iterations
    - would expect it to be higher to deal with the fact that only reward is at the end
    - maybe to deal with the fact that we need to get there more quickly because wandering around can drop us in a hole
- epsilon of 0.9 seems to do best, still not great, but likely just due to iteration limiting

## forest