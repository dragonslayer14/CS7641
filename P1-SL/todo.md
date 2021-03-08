- [x] pull sklearn examples
- [ ] find datasets
    - maybe done?
- [ ] set up combined file to read in dataset and run all algos on it
    - no tuning yet, just getting everything in a single place so evaluation
  wrapping /graphing can be started
- [ ] get error metric code set up
    - allows it to be called for each algo after classification is done
    - then will be combined for HP tuning to get error model complexity charts


Decision tree: Depth of tree
SVM: Kernel - can be a table
Neural Network: Hidden layer depth and size, learning rate
Boosting: Number of weak learners
k: kNN 


grid search over multiple hyper parameters, then use validation curves around the best values to optimize
    possibly grid search again over some small range with high precision

- [x] run validation curve for ada over estimators to see about reducing variance
- [x] evaluate svm and run grid search over kernels
- [x] evaluate knn and plot validation over distance weighted k
- [ ] evaluate ann and plot validation over hidden layer size

- [ ] repeat evaluate and tune for dataset 2