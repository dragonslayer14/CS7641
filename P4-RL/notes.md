# P4

## OH 10

- should do small and big within each problem
- care about exploration strategy
- should be discrete, continuous would make your life harder
- should still talk about why they are interesting and why they are different, one grid world one not grid world
- comparing convergence and time behaviors
- exploration strategy is key, compare with VI PI
- if you are not converging, you may not be visiting all states, see why that is the case
- visualize the policies, will make it easier
- ok to do things like reward shaping
  - certain environments may be too difficult
- analysis of convergence plot
  - delta convergence: where optimal is known value and the convergence towards that is tracked

# OH 11

- usefulness is the important metric in practice, relevance is not as big of a deal
  - not really a useful distinction
- PI/VI
  - need reward and transition matrix, so problem must contain it
  - could build up yourself, but takes time and can make mistakes, better to use one in the library that already has it
- can do 2 non gridworld problems
- gridworld problem doesn't have to be difficult
  - can be defined by own preference, a small grid with some obstacles and some stochastic behavior
  - need enough stochasticity/randomness that it become challenging a little bit
- PI/VI know rewards ahead of time, so they can do that maximization, Q-Learning has to find them by exploring
- can use different convergence plots for different algos so long as they all talk about convergence
- PI takes a small number of iterations, but those iterations take a long time
- VI stops when the values "stop improving"/stop improving significantly
  - significantly here is defined by your criteria
- PI/VI how is convergence defined and know that you need to do it
  - it may be built in, you may have more control on top of that as well
- mdptoolbox example forest should be fine
  - example small may be too small and uninteresting
  - openai gym has some discrete examples that can be used as input
- not expecting the same kind of visualization that can be easily done for gridworlds for non-grid world problems
  - may be able to have some kind of matrix view or 1's and 0's or whatever makes sense
- you define the criteria for convergence for Q-Learning
  - convergence wrt the utility is easier than setting it by policy
- may be some states where q learning does not find the optimal policy
  - maybe it is not being visited enough
  - maybe there is some component leading to this behavior in the problem
- may have to do reward shaping for qlearning for something like frozen lake
- if the problem is not that hard, q learning should be able to find the same policy as VI/PI
- may not find best policy for large/complex problem
  - still need to show doing your best to make them converge
- the goal is to converge to the optimal policy, but if you can't then that is something you need to analyze
- VI focuses on convergence based on values, but the change may be so minor as to not matter and the policy is not changing
  - downside of VI vs PI that you will see in this assignment
  - part of the analysis, what convergence are you going for? which one is easier? and what are you missing by going for that convergence?
  - if you are marking convergence by value rather than policy, probably going to run for longer
  - you set the convergence criteria, just understand the differences
- For checking q learning convergence, would plot q value
  - would probably want some threshold plus certain number of iterations to determine convergence
- should, but not must, explore different sizes for each problem, since it gives more material for analysis
  - complex problem should so some complexities and differences in the behaviors of these methods
    - if they find the optimal policy without any problems, but q learning just takes a bit longer, that is not enough
- forest is not a grid world, because you are stochastically moving through time, not free movement
- definitely want to be able to tune epsilon for q learning for exploit/explore strategy
  - as well as decay, start exploring more, then work towards more exploitation
  - hiive mdptoolbox seems to have the ability to pass in more params than the original
- block dude is not a gridworld since you can place blocks
- toy text is not grid world
- BURLAP is, by far, the best option for this
- plotting in java, probably easiest to dump to csv then plot in something else
- it's ok if q learning doesn't converge after a lot of iterations, it's just part of the analysis

## OH 12

- don't need to describe how the algos work, but it will come up anyway as part of the comparison, since it is based heavily in how they are defined
  - pi looking at the policy changes as opposed to vi looking at the utility of each state
- ideally looking at multiple sizes for each problem to compare easy, medium, challenging setups
  - comparing between two completely different problems is hard because so much changes
  - two sizes should be sufficient
- can use q values to track convergence
  - not sure what this means or if this is in the context of rolling your own
- for PI, if you are looking at more than the policy no longer changing, you are throwing away the advantage of policy iteration
  - policy iteration looks at the point where the policy stops changing, rather than the utility of each state, which may not converge for a while, even without the policy changing
- should have at least some visualization for the HP tuning, but it isn't necessary to show it for all sizes, if you do multiple
- need something larger than 8x8 for frozen lake to call it a large problem