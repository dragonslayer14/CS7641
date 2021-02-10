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

### Boosting

#### basic

- does ok, decline start around 45 examples, still to about 90% accuracy in training data, but ends about 77% on cv
- cv peaks around when training decline starts
- likely hitting the limit of the default 50 learners it was allocated
- validation curve shows accuracy on training data constant at about 90% and cv in the 80's
    - across the entire span, something is messed up and isn't running correctly

### KNN

#### basic

- training and cv scores staying mostly parallel
- vague upward trend, but very little improvement from 30 examples to 80
- discrepancy is likely due to the default 5 neighbors of uniform weight
    - this makes sense when the data points are sparse, but should improve as more are added
    - likely due to the fact that there is not a lot of data, so there isn't enough room to progress

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

## Dataset 2 (Wine Quality)

### DT

### basic

- obvious and expected over fitting with very low
cv accuracy (55%)
  - overfit and not scalable to testing set
