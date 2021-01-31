# OH

## data sets

- one could have many features
  - curse of dimensionality
  - useful for later dim red
- maybe noisy data
- categorical in one set and numerical in the other
  - combination
- anything that may yield interesting observations on the algorithms
- a lot of training samples for one versus not enough for the other
  - expecting overfitting, etc.
- spend reasonable time to choose, but don't overthing it
- interesting => significant difference between algorithms or tuning of hyper params
  - decision tree works poorly but ANN works well
  - linear kernel for SVM is poor but another one works well

## Hyperparameter tuning

- coarse grid search to find neighborhoods of good parameters
  - prevent overfitting
- learning curve analysis
  - performance of learning and validation curves as a function of number of of training points
  - if you have 1000 points, plot performance with 100 in the training and validation, then 200 ...
    - see bias, variance, or if you are doing well
  - hyper parameters are fixed over analysis
  - one way
    - coarse grid search for set of hyper parameters
    - Learning curve analysis
    - see if there is high bias or high variance and try to find a solution
      - need to understand dataset and model
    - model complexity analysis
  - don't touch testing data when tuning, use cross validation
  - look up bias variance tradeoff
  - demonstrate that you know how to analyze these graphs and how specific hyper params affect complexity
  - look into bias variance equations to see how they relate to over/under fitting

## bagging and boosting

- bagging is high variance
  - overfitting on each learner
- boosting is high bias
  - weak learner starts with high bias

## charts

- plots or tables, so long as trend is visible, plot may be easier with less writing
- learning curve, model complexity curve, plus anything additional that ability to answer assignment's question
- talk about bias and variance

## hyper parameter tuning

- don't need every step
- show important steps
- explanation points
- final learning curve

- unbalanced dataset is allowed, but most things work better on balanced ones, so it will be easier on us to have that

- probably more than 4 or 5 features
- if using imbalanced dataset, accuracy may not be best metric (70% accuracy on a set with 70% of one type is just guessing that for everything)
- gower distance can be used for categorical data

- "pre-pruning" limiting depth for decision tree is going to combat overfitting
  - so this could be an intelligent choice to tackle that problem
- table probably better for SVM kernel comparison, as they isn't really a visible "trend", they are fundamentally different
- accuracy may not be the only metric to check
  - wall clock time
  - is a false positive/negative more costly?
    - then it is more important to not make that error


## metrics to plot

- makes sense for problem
  - accuracy is a good place to start
  - check for type 1/type 2 errors
- knn only really has k and distance metric
- ANN has many more
  - looking at hidden layer width and depth
- Boosting may only the number of learners
- DT could be number of leaves per node, min or max leaves per node
  - at least looking at pruning
- key thing is understanding role of hyper parameter for an algorithm, changing it, and analyzing the change in behavior
- tuning hyper parameters for model complexity and learning curve
  - bias variance
  - performance and generalizability of the model
- compare and contrast