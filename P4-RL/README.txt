Code Location: https://github.gatech.edu/cphrampus3/CS7641.git
Running code:

- Project code in folder `P3-UL`
- kmeans elbow method code adapted from from https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
- pca elbow code from https://www.kaggle.com/lonewolf95/classification-tutorial-with-pca-and-gridsearchcv
- EM BIC model selection code from https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py
- Requirements were pip frozen to requirements.txt
- data labels in wine.csv were modified to be 0 based, rather than 1, for the sake of comparison
- tsne used the base call in yellowbrick to avoid needing to manually fiddle with parameters
    - https://www.scikit-yb.org/en/latest/api/text/tsne.html#t-sne-corpus-visualization
- `python main.py`
    - run all algorithms for all problems
    - ANN comparison dumps scores and times to console for making the necessary table