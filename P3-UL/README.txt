Code Location: https://github.gatech.edu/cphrampus3/CS7641.git
Running code:

- Project code in folder `P3-UL`
- kmeans elbow method code adapted from from https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
    - pca elbow code from https://www.kaggle.com/lonewolf95/classification-tutorial-with-pca-and-gridsearchcv
- Requirements were pip frozen to requirements.txt
- data labels in wine.csv were modified to be 0 based, rather than 1, for the sake of comparison
- `python main.py` // update as needed
    - run all algorithms for all problems
    - creates fitness vs problem size chart, stored in the charts folder, by the name of the problem which
    contains all algos for comparison
    - ANN comparison step generates a chart named ann_compare.png and outputs fit times to the console for the table