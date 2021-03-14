# Notes

## Lectures

- Hill climbing, annealing, and genetic algos are amnesic, since they end up with the about same amount of information as they start with, just a point or small collection of points, just in a better place in the space

### Hill climbing

- simple
- can get caught in local optima
  - can be improve with random restart hill climbing to try to avoid this

### Simulated Annealing

- uses temperature as a medium for balancing exploring and exploiting the space
- high temperature is more explore
- low temperature is more exploit
- temperature lowered slowly over time
- high temperature helps flatten the space to get past valleys and out of local optima

### Genetic Algorithms

- use and combine a population to create offspring, trying to get better over generations
- crossover merges multiple individuals into a new learner or a set of new learners

### MIMIC

- start with picking uniformly over all points and eventually get better and better until we just have optima
- similar to genetic algorithms in that the "fittest" move forward
- pseudo code
  - generate samples from p^theta_t(x)
  - set theta_(t+1) to top nth percentile of samples
  - keep samples such that f(x) >= theta_t+1
  - estimate new p^theta_t+1(x)
  - repeat
- does well when optima are based on structure without regard for values
  - e.g., x is different from x+-1, regardless of x
- tends to take (orders of magnitude) fewer iterations, but those iterations take longer
  - getting more information per iteration
  - works well when cost to evaluate fitness function is high, since we are calling it much less

## project

- "first problem should highlight advantages of your genetic algorithm, the second of simulated annealing, and the third of MIMIC"
  - "They can be simple. For example, the 4-peaks and k-color problems are rather straightforward, but illustrate relative strengths rather neatly."
- will probably need ABAGAIL and jython to get the necessary algos, seems sklearn does not have them and better to be consistent than a little from each
  - or mlrose

## OH 4

- example problems include 4 peaks and k color, as well as traveling salesman, knapsack, flip flop
  - RHC and annealing are instance based, they don't look at the structure
  - genetic and mimic use distribution of structure
    - relate the structure of the problem to these methods
    - how mimic builds up its distribution may be mimicked in the structure of the problem
 - explain why x problem out performs the others
   - do tuning
   - if you had more time, what steps would you take?
   - graphs
   - convergence properties
     - for ann, compared to backprop
     - still want iteration curve
     - compare with randomized optimization algo to compare to gradient descent
     - compare with A1 in terms of time and accuracy
   - optimization problems
     - look at fitness vs iterations, function evaluations, varying size of problem (how it affects fitness), convergence within timeframe
       - different difficulties of the problem is important to have one stand out
         - if you only have one, you can't be quite sure
           - one may perform well on simpler version of the problem, but worse on more complicated

## OH 5

- there should be some variation in problem size
  - there should be some point where one starts to outperform and why that is the case
  - typically as the problem is harder, solutions that align with the structure of the problem will emerge as the favorite
- mimic paper gives an example of MIMIC outperforming the others
  - maybe k-coloring?
- outperform in value of fitness function, but can also include time
- mlrose implementation of MIMIC should outperform on k-colors, but the max attempts have to be up pretty high
- make sure algorithms are converging
- setting max iterations can misrepresent some algorithms
- only need to tune for the size that is going to be used in the report, not for each size
  - the other curve is just to show that tuned performance over various problem sizes
  - point it out in report
- don't worry about reporting iterations needed, work on a convergence criterion and justify it in the report
- simulated annealing
  - decay strategies, and for all, tables and charts for hyper parameter tuning are useful
  - convergence plots could also be useful to see for hyper parameter tuning
- 1 or 2 paragraphs going over structure, performance, issues, etc. for ANN
- make sure to compare all backprop replacements, including against back prop
  - converge faster, slower, etc.
- change neural net if needed, possibly re tuning for mlrose
- don't fix number of iterations to be the same for all problem sizes, they should converge
- hyper parameter tuning should also include graphs
- ABAGAIL random state would need to be set for reproducibility, setting seed in random class
  - do multiple random seeds and average to get result to avoid good or bad random seeds
- plots
  - function evals and iterations vs fitness
  - hyperparameter tuning
  - complexity vs fitness and function evals
- Jython stuff for ABAGAIL is not great, doesn't allow native use of something like numpy
- abagail and mlrose should already handle discretizing the jumps for neural nets
- the optimization problems likely will not do better than gradient descent
  - if they do, that can be interesting to analyze, maybe it comes down to not having tuned P1 as well as it could have been
- high bias = crude model, high variance is a very complicated model, fits data too well
  - high variance is going to fit on the noise as well
- choosing problems, library will implement certain problems, knapsack, k queens, etc.
  - pick 3, tune, see what is highlighting which algorithm, if two highlight the same one, swap one out
  - should have some idea about which algorithm will outperform based on how the algorithm works
    - simulated annealing is neighbor to neighbor eval, if there is a complex underlying structure, probably won't perform well
    - genetic algorithm, think "if I have two good samples, if I do cross over, is it more likely to get a better solution?"
    - MIMIC tries to understand dependency between the features
    - if there is a complicated underlying structure, the structure based algos (MIMIC and GA) will probably out perform
- optimization problems, there is no "dataset", it's a function you are trying to maximize
- make sure the problems highlight each of the three algorithms necessary
- talk about the problem, why is it interesting, is it np-hard, how many local optima, how many global optima, structure of problem and how some algorithms will exploit that structure
  - maybe a really simple problem that will highlight something like simulated annealing
- tune hp for each algo for each problem for a fixed problem size
  - size should be picked as to highlight the particular algorithm
  - explore different problem sizes and report performance
  - there will be room for improvement, don't have to do everything, it is ok to leave notes of "if x and y were done, the performance would likely increase"
  - convergence property, getting to the correct answer in a reasonable amount of time
    - comparing both on fitness value and time it took to converge
  - think about advantages and disadvantages of back prop vs optimization problems
    - these are discrete problems back prop is for smooth continuous spaces
- plots are not analysis
  - they are the start of analysis, compare and contrast

## OH 6

- for RHC, generate multiple random seeds, run for each and average results for plot
  - maybe just for all of them, they are all randomized optimization
- grid search is not analysis, there is no dataset, but you need to explain why things were tuned
  - why are you tuning it? ask questions of your results and answer those questions
- tune for a middle sized problem, not so simple it is not good, not so complex it takes forever to run, there is already a time crunch
- don't need to look into neighbors for this project
  - RHC is not really meant to perform well, it's a baseline
  - understand its limitations and explain it
  - don't spend too much time on it, use what is defined for you
  - not highlighted in the problems, so it is kind of wasted effort
- only really need to include HPs that are included in the analysis
- backprop is working on continuous values, which is not something the is present in these problems
  - if RHC and SA are performing about as well, the underlying problem is probably simple
  - NN is also probably not the absolute best version of the network that could exist for the problem
  - performance can be defined as both the performance function, as well as time
- GA and MIMIC are looking for underlying structure, so if there is not, it can take much longer and waste efforts
- Time vs problem size could be interesting
  - table could be useful instead of plot
- during dev, may be good to choose 2 or 3 problem sizes, tune for each, then pick one for the analysis
- don't let the whole thing go once you have a result, most likely not going to be optimal, still important to talk about what more you could do differently
- with so many problems and algorithms, may just look at one HP per algorithm, just chose the most interesting one
- mlrose outputs a csv, can plot in excel, just include the code that makes the csv
- vary problem length for all algos, not just the one being highlighted

## OH 7

- GA in NN, instead of a fitness value, the loss function, which will determine what a good population is in terms of GA
- convergence is just getting to a steady value, not necessarily the best one
- set a seed, but also average runs
- not doing CV for this assignment, will get performance vs iteration curve that shows convergence
- still need to care about training and testing performance and bias variance tradeoff
  - BV tradeoff is not as important for the grading here, assumes P1 showed its role
- get to a good performance, but in how much time?
- DOn't need to show the bias variance trade off, but it is assumed to have been done
  - I'm assuming this is if the network is being re-tuned in the new system, since the architecture is frozen when replacing algo
  so no changes to how it is called should be made to ensure the same network is running the same way except for algo substitution
- need a loss curve to show convergence behavior
- fitness curve does not care about sample size, it is for convergence behavior, i.e., fitness over time/iterations
- tuning of RO NN is necessary to find where HP like temperature, restarts, population, etc. are best
  - needing to tune makes sense, default parameters and just changing algos doesn't make sense... should have known
- NN plot for dataset should show the "more interesting one" showing differences
  - SA is wavey as it explores, then improves as it exploits an area, and RHC ends up just under it, shows some difference
  - don't worry too much about a strict guideline as long as you're able to do the analysis
- fitness vs iterations, curve would be expected to go up and hopefully converge at some point
- there is a feedback loop in backprop which is its utility