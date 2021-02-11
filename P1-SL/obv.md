# Observations

## Dataset 1 (Wine)

### DT

#### basic

- obvious overfitting in training data with 100% accuracy,
but surprisingly high validation fitting around 89%
    - likely due to such a small dataset covering enough examples

#### tuning

- max depth 3 brings learning curve cv score to about 88%, but this seems to be the "best" given the validation curve and grid search
- ccp alpha search over depth 3 to 6 (fully overfit) does show any better progress
- very small dataset, so without more data points, this seems to be the best
- it is unclear why the suboptimal are being given as optimal, but the better one by score, overfit, will be kept
    - this is likely only working because of the very small dataset and the accuracy would drop given more data
- this is about as heavily biased to the data as it can be, but the attempts at generalization showed no improvement
    - will probably run "best" depth 3 against the test data as well, to see if the general rule to generalize a bit applies here
    or the data are simply so similar and in such small number that overfitting matches the "universe" of the problem

### Boosting

#### basic

- does ok, decline start around 45 examples, still to about 90% accuracy in training data, but ends about 77% on cv
    - plumments at 65 examples
- cv peaks around when training decline starts
- likely hitting the limit of the default 50 learners it was allocated
- validation curve shows accuracy on training data constant at about 90% and cv in the 80's
    - across the entire span, something is messed up and isn't running correctly

#### tuning

- seems to get about 35 learners, then stop improving as more are added
- also need to look at the base learner, allowing more of a tree instead of a stump could result in better performance
- over 97% with DT depth 4, 300 estimators, and .1 ccp
- 99% with dt depth 2, 10 estimators, and .1 ccp
    - very small set, so not many needed with minimal pruning
    - probably good enough
    - run with 150 estimators to check for improvement, but no significant change was found so the simpler model is better here

### KNN

#### basic

- training and cv scores staying mostly parallel
- vague upward trend, but very little improvement from 30 examples to 80
- discrepancy is likely due to the default 5 neighbors of uniform weight
    - this makes sense when the data points are sparse, but should improve as more are added
    - likely due to the fact that there is not a lot of data, so there isn't enough room to progress

#### tuning

- validation over k devolves as k grows
    - unsurprising as it then becomes a plurality vote
- changing to distance weighting works to mitigate the issue by only working on items near x
    - training stays at 100%, which is strange
    - perhaps they occupy specific areas within space and all classes are covered early, so it's lucky?
        - there are only 3, evenly distributed, and it starts with 20 or so examples, not unreasonable
- manhattan distance seems to work better than euclidian for an unknown reason
- k = 3 with manhattan distance seems best at 85% for cv

### ANN

#### basic

- CV score does not seem to improve over examples, this may be a training issue
- default has 100 hidden nodes on second layer which may be over complicating the calculations
    - certainly not overfitting, as the training error drops from about 90% to about 70%

### SVM

#### basic

- seems to become parallel around 65 examples, with about 70% accuracy
- this may be due to the rbf kernel or the number of examples, but I don't know anything about the kernel to say
    - would need to try a coarse search for C and gamma to see how the kernel could improve
    - also a search through the other kernel options, likely avoiding a custom kernel

#### tuning

- linear kernel seems far and away the best
linear: training: 1.0 0.0, cv: 0.9454545454545455 0.06680426571226848
poly: training: 0.7142857142857142 0.03610893068595978, cv: 0.7454545454545454 0.07385489458759967
rbf: training: 0.7380952380952379 0.04937247941268508, cv: 0.7 0.11354541815269817
sigmoid: training: 0.2976190476190476 0.12710807442894417, cv: 0.20909090909090908 0.1171281702429557
- grid search c values
     - c 1 -> 2000 seems to be the same value, .945 cv
     - also keeps the model simple
     - small c does not improve
     - balanced class weight does not improve

## Dataset 2 (Wine Quality)

### DT

### basic

- obvious and expected over fitting with very low
cv accuracy (55%)
  - overfit and not scalable to testing set

### Boosting

#### basic

- training and test accuracy consistent, but around 40% with no real change after 100's of estimators
    - something is wrong, but what? 