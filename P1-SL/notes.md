# Notes

## OH 1

### data sets

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

### Hyperparameter tuning

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

### bagging and boosting

- bagging is high variance
  - overfitting on each learner
- boosting is high bias
  - weak learner starts with high bias

### charts

- plots or tables, so long as trend is visible, plot may be easier with less writing
- learning curve, model complexity curve, plus anything additional that ability to answer assignment's question
- talk about bias and variance

### hyper parameter tuning

- don't need every step
- show important steps
- explanation points
- final learning curve

### Misc

- unbalanced dataset is allowed, but most things work better on balanced ones, so it will be easier on us to have that
- probably more than 4 or 5 features for dataset
- if using imbalanced dataset, accuracy may not be best metric (70% accuracy on a set with 70% of one type is just guessing that for everything)
- gower distance can be used for categorical data
- "pre-pruning" limiting depth for decision tree is going to combat overfitting
  - so this could be an intelligent choice to tackle that problem
- table probably better for SVM kernel comparison, as they isn't really a visible "trend", they are fundamentally different
- accuracy may not be the only metric to check
  - wall clock time
  - is a false positive/negative more costly?
    - then it is more important to not make that error


### metrics to plot

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

## Readings

### decision trees

- inductive bias
    - prefer small trees to larger trees, given correct classification
        - mitchell 3
- can handle noisy data by changing termination to accept hypothesis without perfect fit
  but it generally susceptible to noise (need pruning) and overfitting given small datasets
    - mitchell 3


### ANN

- bias
    - smooth interpolation between data points
        - mitchell 4
    - prefers less "complex" network
        - lecture
- handles noisy data well
    - mitchell 4
- can overfit given too many backpropagation iterations
    - mitchell 4


### Boosting

- can fail given insufficient data, overly complex base classifiers or base classifiers that are too weak
    - boosting paper
- especially susceptible to noise
    - boosting paper
- **check lectures


### KNN

- distance weighted KNN is robust to noisy data
    - weighted average smooths out isolated noisy example
    - mitchell 8
- especially susceptible to curse of dimensionality
    - uses all dimensions for distance
    - can weight attributes differently to "stretch" the euclidian space
    - can also simply zero out some less relevant attributes
    - mitchell 8
- inductive bias
    - assumes items near k are similar to k (based on distance function)
    - assumes all features matter equally (also based on distance function)
    - lectures
- distance function is domain knowledge, as is k
    - lectures
- better giving more data rather than more dimensions
    - curse of dimensionality, need for exponentially more data as dimensions grow in number
    - lecture
    
- **check lectures


### SVM


## misc
- wine data may be too simple, a bunch of dimensions, little data, but only 3 classes

## OH 2

- confusion matrix is useful for seeing different types of errors being made, useful for imbalanced data sets
- f1 score is a combination of precision and recall
- hyper params
  - tune as many as seem relevant in the background, MC curve only needs to show the most interesting
  - dt - depth and pruning
  - svm - kernels
  - anything that modifies architecture of model
  - NN - size and depth, width and depth
  - KNN - k
  - boosting - number of weak learners
- ok to start with default parameters
- should show some learning curve early on (some initial point, not necessarily the first pass) from which to show the improvements from tuning
- high variance - large gap in learning curve between training and testing(validation) sets
- high bias - convergence of training and testing(validation) sets to a number lower than you know you can get
  - in case of DT, this may come from being overaggressive in pruning, can try to be less aggressive or a different pruning method
- learning curve and model complexity curve can plot against error or accuracy or whatever metric works for what you are trying to solve
- sklearn can use either max-depth (pre-pruning) or cost complexity pruning (post pruning)
- model complexity if you are changing the value of a hyper parameter
  - probably just going to finish tuning then run tuned learner over test data
- number of iterations is a hyper parameter for NN
  - prevents overfitting
- learning curve against training size, useful for all algos
- learning curve against iterations, only for iterative algos, which is only NN in this case
- single HP per model complexity plot, play with sizing to get them next to each other to save space
  - table is fine for discrete variables, like the svm kernel, since there is no real comparison between them
- the dataset being too noisy or needing more samples is a valid conclusion, but enough steps need to be taken to show that
- stratified shuffle split could be good for training/testing split to ensure proper distribution in each set
- learning curve is for a particular model, the model is constant through the curve, just a different number of items in the training set
  - e.g., you would make a dt for the entire training set, prune it, then plot the learning curve for that tree
- grouping by dataset may be easier to read through, but grouping doesn't matter
  - can show algos for one dataset, then the other, then have a conclusion comparing them
- the input to all algorithms should be the same for comparison
- with a given dataset, you may not be able to do better
  - e.g., seeing that the learning curve has a trajectory that they would converge to a smaller variance if more data were present
- can use validation curve model from sklearn as model complexity
- getting high accuracy on testing from a dataset with many features
  - can throw out some features to make it harder, that is valid
- stratified samples is making sure the distribution of class labels is the same between training and test sets