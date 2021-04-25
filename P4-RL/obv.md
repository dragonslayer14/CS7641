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

- large number of states with default params seems to have PI/VI take the most iterations to find a value
    - not really a great metric, but it at least means that the problem has more complexity like that
- VI/PI do well as expected in a small number of iterations
    - very small for PI with little change over epsilon due to how it optimizes to begin with, so it is already meeting higher cutoffs
    - still fairly small, couple hundred, for VI since it is waiting for value convergence, even if the policy didn't change
- QL takes a million iterations, 8 minutes, to get a score of 47, about a third of VI/PI, with a fraction of a point increase when run for 5m
    - likely an issue with the fact that given so many states, all cannot be visited as often as QL would require for the best policy