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

#### tuning

- 85 nodes in one hidden layer seems to be best, no change to learning rate or alpha
    - could run validation curves to be sure, only slight checking
- got to about 95% train, 92% cv with one layer of 85
- search over alpha, learning rate, and momentum did not give anything better

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

- imbalanced to 2 classes, really only made up of 3 with some samples in the others.
- probably possible to work with, but need to graph other than accuracy
    - confusion matrix is helpful, but not so much for tuning, would need a different score for validation curve
- other datasets may be better, especially if balanced
    - adults?
        - maybe, binary and imbalanced, so higher accuracy over all, but maybe not better in terms of ease of tuning
    - wifi?
        - perfectly balanced,but far too simple, basic runs get 90%+ for all algos, no good comparisons, I don't think
    - yeast?
        - multiclass and imbalanced to a few in a similar way, can run basic and see what improvement room there is
    - need something, balanced, non-trivial, maybe stick with same approx number of params?
        - original idea was comparison of number of dimensions across dataset size and distribution, so sticking could 
        work so long as improvements to confusion matrix scoring can be found, but not sure how
            - still shows the problems of a bunch of data defining a small amount of the class space
            as opposed to a small amount of data points in the same dimensionality defining a smaller class space well

### DT

#### basic

- obvious and expected over fitting with very low
cv accuracy (55%)
  - overfit and not scalable to testing set
- fully overfit tree is depth 25
    - look into pre and post pruning

#### tuning

- no real improvement from tuning depth or alpha with a weighted f1 score
    - grid searching reveals some pre pruning with depth 18-21, but with little impact
    - validation curves show very heavy impact from tuning alpha, likely due to regression to populous class
    - Confusion matrix shows that for the 3 major classes, it is getting the correct class most often
    followed by others in descending order of number of samples
- ultimately, probably just need more data for other classes, the enormous skew makes it highly dependent on a few examples
- unsure why the 3 main classes are not doing better, this may require better domain knowledge about feature importance,
as noted in the dataset text calling for feature selection
    - could be a lot of noise in unimportant features to the actual classification 

### Boosting

#### basic

- training and test accuracy consistent, but around 40% with no real change after 100's of estimators
    - something is wrong, but what?


#### tuning

- able to tune in much the same way as in 1, max depth 5, but alpha pruning did not improve anything
- seems to trend slightly up, even after 1000 learners
    - given enough learners it would do well, but lack of time and computing power will cap it to 500
    - still likely to do poorly on the edge cases, since there are not enough samples to build a better understanding
- went from converging around 40% to maybe around 70% if traced out, cv around 60%
- 500 depth 5 trees is probably too much, but it's good enough for this

### svm

#### basic

- untuned with default rbf kernel gives little room between training and testing, converging ~30%

#### tuning

- taking a long time to grid search...
- grid search gave rbf, c 1000, gamma 0.001
- 10k C works better, grid searching with 100k and 1m
    - should be the final svm step, not much more that can be tweaked


linear: training: 0.5432969852469532 0.0025334724385030678, cv: 0.5184615384615384 0.007802627720360503
poly: training: 0.4447722899294419 0.003390526051674793, cv: 0.4297435897435897 0.01802912392948404
rbf: training: 0.44381013470173186 0.003456624396486848, cv: 0.42743589743589744 0.013436484622127074
sigmoid: training: 0.381975625400898 0.03125054901201807, cv: 0.36538461538461536 0.025057461904733406

### KNN

#### Basic

- training just above 60 and cv jsut about 40
- increase of neighbors shows decrease, unsurprising due to uniform weighting

#### Tuning

- grid search shows p=1 and distance weighted
    - not really surprising
- switching to distance weighted samples shows 100% on training with 10 neighbors, 53% on cv
- validation curve shows about 55% for 14 neighbors

### ANN

#### Basic


#### Tuning