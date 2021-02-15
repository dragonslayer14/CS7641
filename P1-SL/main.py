import warnings

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import neighbors
import matplotlib
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import validation_curve
from datetime import datetime
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
# We use OneVsRestClassifier for multi-label prediction
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import plot_confusion_matrix

# show popup for graphs on mac

matplotlib.use("TKAgg")

import matplotlib.pyplot as plt

DATASET_1 = "data/wine.csv"
DATASET_2 = "data/winequality.csv"
DATASET_1_NAME = "Wine"
DATASET_2_NAME = "Wine Quality"


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        scoring="accuracy", n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    plt.title(title)

    axes[0].set_title("Learning Curve")
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True,
                       scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")


def average_precision_scorer(estimator, X, y):
    # weight score by weight of label in set
    unique, counts = np.unique(y, return_counts=True)
    counts = counts / len(y)
    scale = dict(zip(unique, counts))
    print(estimator)
    y_score = estimator.predict_proba(X)
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(unique)):
        precision[i], recall[i], _ = precision_recall_curve(y[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(y[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y.ravel(),
                                                                    y_score.ravel())
    average_precision["micro"] = average_precision_score(y, y_score,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))

    return average_precision["micro"]


def plot_validation_curve(estimator, title, X, y, x_lab, y_lab, param_name, param_range,
                          scoring="accuracy", ylim=None, cv=None, n_jobs=None):

    plt.figure()
    plt.title(title)

    if ylim is not None:
        plt.ylim(*ylim)

    # param_range = np.logspace(-6, -1, 5)
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        scoring=scoring, n_jobs=n_jobs, cv=cv)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    lw = 2

    # special case for ann passing tuples for hidden layer sizes
    if type(param_range[0]) is list or type(param_range[0]) is tuple:
        param_range = [r[0] for r in param_range]

    plt.plot(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")


def run_dtc_1(fig_name = None, show_plots = False):
    # read in dataset from file
    with open(DATASET_1, 'r') as f:
        data = np.genfromtxt(f, delimiter=',')

    data, labels = data[:,:-1], data[:,-1]

    # split for training and testing
    data_train,data_test, label_train, label_test = train_test_split(
        data, labels, test_size=0.4, random_state=0, stratify=labels
    )

    # define model
    # fix hyperparameters as needed to avoid unneeded grid search
    clf = DecisionTreeClassifier(criterion="entropy", random_state=0)

    # TODO run cost complexity pruning code to find alpha for ccp

    # based off sklearn example for hp tuning
    # https://scikit-learn.org/stable/modules/grid_search.html#

    # define hyper parameter space to check over
    # param_grid = {
    #     "max_depth": range(1,10),
    #     "min_samples_split": [2, 5, 10],
    #     "ccp_alpha": [0,0.1,0.2,0.3,0.4,0.5],
    #     "min_samples_leaf": range(1,10)
    # }
    #
    # basic = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(data_train, label_train)
    #
    # cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    #
    # sh = GridSearchCV(clf, param_grid, cv=5).fit(data_train, label_train)
    # print(sh.best_estimator_)
    # clf = sh.best_estimator_

    # based on sklearn learning curve example
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    # plot learning curve for current model
    title = f"Learning Curves (Decision tree) ({DATASET_1_NAME})"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.

    plot_learning_curve(clf, title, data_train, label_train, ylim=(0, 1.01),
                        cv=5, n_jobs=4)

    if fig_name is not None:
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
        # plt.savefig(f"{fig_name}_learn_{dt_string}")


    # based off sklearn validation curve example
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html

    # plot validation curve
    title = f"Validation Curve with DT ({DATASET_1_NAME})"
    x_lab = "Depth"
    y_lab = "Score"

    plot_validation_curve(clf, title, data_train, label_train, x_lab, y_lab, cv=5,
                          param_name="max_depth", param_range=range(1, 10), ylim=(0.0, 1.1))

    if fig_name is not None:
        plt.savefig(f"{fig_name}_val_{dt_string}")

    if show_plots:
        plt.show()

    

    # run against test, uncomment for final analysis
    # !! don't touch during training/tuning !!
    # print(clf.score(data_test, label_test))

    # track accuracy and variance so later report can pull numbers as needed


def run_dtc_2(fig_name = None, show_plots = False):
    # read in dataset from file
    with open(DATASET_2, 'r') as f:
        data = np.genfromtxt(f, delimiter=',')

    data, labels = data[:,:-1], data[:,-1]

    # split for training and testing
    data_train,data_test, label_train, label_test = train_test_split(
        data, labels, test_size=0.4, random_state=0, stratify=labels
    )

    # define model
    # fix hyperparameters as needed to avoid unneeded grid search
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=18,
                                 random_state=0)

    # TODO run cost complexity pruning code to find alpha for ccp

    # based off sklearn example for hp tuning
    # https://scikit-learn.org/stable/modules/grid_search.html#

    # define hyper parameter space to check over
    # param_grid = {
    #     "max_depth": range(1, 26),
    #     "min_samples_split": [2, 5, 10],
    #     "ccp_alpha": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    #     "min_samples_leaf": range(1, 10),
    #     "criterion": ["gini", "entropy"]
    # }
    #
    # basic = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(data_train, label_train)
    #
    # sh = GridSearchCV(clf, param_grid, cv=3, scoring="f1_weighted").fit(data_train, label_train)
    # print(sh.best_estimator_)
    # clf = sh.best_estimator_

    # based on sklearn learning curve example
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    # plot learning curve for current model
    title = f"Learning Curves (Decision tree) ({DATASET_2_NAME})"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    plot_learning_curve(clf, title, data_train, label_train, ylim=(0.0, 1.01),
                        cv=5, n_jobs=4, scoring="f1_weighted")

    if fig_name is not None:
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
        plt.savefig(f"{fig_name}_learn_{dt_string}")

    # based off sklearn validation curve example
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html

    # plot validation curve
    title = f"Validation Curve with DT ({DATASET_2_NAME})"
    x_lab = "alpha"
    y_lab = "Score"

    plot_validation_curve(clf, title, data_train, label_train, x_lab, y_lab, cv=5, scoring="f1_weighted",
                          param_name="ccp_alpha", param_range=np.linspace(0,1), ylim=(0.0, 1.1))

    if fig_name is not None:
        plt.savefig(f"{fig_name}_val_{dt_string}")

    # plot_confusion_matrix(basic, data_train, label_train)

    # split and retrain to do a validation confusion matrix
    t = DecisionTreeClassifier(random_state=0)
    t_train,t_test,l_train,l_test = train_test_split(data_train, label_train, random_state=0)

    t.fit(t_train, l_train)

    plot_confusion_matrix(t, t_test, l_test)

    if fig_name is not None:
        plt.savefig(f"{fig_name}_conf_matrix_{dt_string}")


    if show_plots:
        plt.show()

    

    # run against test, uncomment for final analysis
    # !! don't touch during training/tuning !!
    # print(clf.score(data_test, label_test))

    # track accuracy and variance so later report can pull numbers as needed


def run_ada_1(fig_name = None, show_plots = False):
    # read in dataset from file
    with open(DATASET_1, 'r') as f:
        data = np.genfromtxt(f, delimiter=',')

    data, labels = data[:,:-1], data[:,-1]

    # split for training and testing
    data_train,data_test, label_train, label_test = train_test_split(
        data, labels, test_size=0.4, random_state=0, stratify=labels
    )

    # define model
    # fix hyperparameters as needed to avoid unneeded grid search
    clf = AdaBoostClassifier(n_estimators=300, random_state=0)
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, ccp_alpha=0.1),n_estimators=10, random_state=0)

    # based off sklearn example for hp tuning
    # https://scikit-learn.org/stable/modules/grid_search.html#

    # define hyper parameter space to check over
    # param_grid = {
    #     # depth of decision classifier? or just optimize number of depth 1 stumps?
    #     # learning rate
    #     "n_estimators": range(10, 300, 10),
    #     # "base_estimator": [DecisionTreeClassifier(max_depth=i) for i in range(1,6)]
    # }
    #
    # basic = AdaBoostClassifier(random_state=0).fit(data_train, label_train)
    #
    # sh = HalvingGridSearchCV(clf, param_grid, cv=5, factor=2).fit(data_train, label_train)
    # print(sh.best_estimator_)
    # clf = sh.best_estimator_

    # based on sklearn learning curve example
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    # plot learning curve for current model
    learn_title = f"Learning Curves (Boosted Decision tree) ({DATASET_1_NAME})"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    plot_learning_curve(clf, learn_title, data_train, label_train, ylim=(0, 1.01),
                        cv=5, n_jobs=4)

    if fig_name is not None:
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
        plt.savefig(f"{fig_name}_learn_{dt_string}")


    # based off sklearn validation curve example
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html

    # plot validation curve
    val_title = f"Validation Curve with Boosted DT ({DATASET_1_NAME})"
    x_lab = "n_estimators"
    y_lab = "Score"

    plot_validation_curve(clf, val_title, data_train, label_train, x_lab, y_lab, cv=5,
                          param_name="n_estimators", param_range=range(5, 20, 1), ylim=(0.0, 1.1))

    if fig_name is not None:
        plt.savefig(f"{fig_name}_val_{dt_string}")

    if show_plots:
        plt.show()

    

    # run against test, uncomment for final analysis
    # !! don't touch during training/tuning !!
    # print(clf.score(data_test, label_test))

    # track accuracy and variance so later report can pull numbers as needed


def run_ada_2(fig_name = None, show_plots = False):
    # read in dataset from file
    with open(DATASET_2, 'r') as f:
        data = np.genfromtxt(f, delimiter=',')

    data, labels = data[:,:-1], data[:,-1]

    # split for training and testing
    data_train,data_test, label_train, label_test = train_test_split(
        data, labels, test_size=0.4, random_state=0, stratify=labels
    )

    # define model
    # fix hyperparameters as needed to avoid unneeded grid search
    clf = AdaBoostClassifier(n_estimators=300, random_state=0)
    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5), n_estimators=500,
        random_state=0)

    # based off sklearn example for hp tuning
    # https://scikit-learn.org/stable/modules/grid_search.html#

    # define hyper parameter space to check over
    # param_grid = {
    #     # depth of decision classifier? or just optimize number of depth 1 stumps?
    #     # learning rate
    #     "n_estimators": range(50,500,50),
    #     "base_estimator": [DecisionTreeClassifier(max_depth=i, ccp_alpha=a) for i in range(1, 6) for a in np.linspace(0,1,10) ]
    # }
    #
    # basic = AdaBoostClassifier(random_state=0).fit(data_train, label_train)
    #
    # sh = GridSearchCV(clf, param_grid,scoring="f1_weighted", cv=5).fit(data_train, label_train)
    # print(sh.best_estimator_)
    # clf = sh.best_estimator_

    # based on sklearn learning curve example
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    # plot learning curve for current model
    title = f"Learning Curves (Boosted Decision tree) ({DATASET_2_NAME})"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    plot_learning_curve(clf, title, data_train, label_train, ylim=(0, 1.01), scoring="f1_weighted",
                        cv=5, n_jobs=4)

    if fig_name is not None:
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
        # plt.savefig(f"{fig_name}_learn_{dt_string}")


    # based off sklearn validation curve example
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html

    # plot validation curve
    title = f"Validation Curve with Boosted DT ({DATASET_2_NAME})"
    x_lab = "estimators"
    y_lab = "Score"

    plot_validation_curve(clf, title, data_train, label_train, x_lab, y_lab, cv=5, scoring="f1_weighted",
                          param_name="n_estimators", param_range=range(10,300,10), ylim=(0.0, 1.1))

    if fig_name is not None:
        # plt.savefig(f"{fig_name}_val_{dt_string}")
        pass

    # split and retrain to do a validation confusion matrix
    t = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5), n_estimators=500,random_state=0)
    t_train, t_test, l_train, l_test = train_test_split(data_train, label_train, random_state=0)

    t.fit(t_train, l_train)

    plot_confusion_matrix(t, t_test, l_test)

    if fig_name is not None:
        plt.savefig(f"{fig_name}_conf_matrix_{dt_string}")

    if show_plots:
        plt.show()

    

    # run against test, uncomment for final analysis
    # !! don't touch during training/tuning !!
    # print(clf.score(data_test, label_test))

    # track accuracy and variance so later report can pull numbers as needed


def run_svm_1(fig_name = None, show_plots = False):
    # read in dataset from file
    with open(DATASET_1, 'r') as f:
        data = np.genfromtxt(f, delimiter=',')

    data, labels = data[:,:-1], data[:,-1]

    # split for training and testing
    data_train,data_test, label_train, label_test = train_test_split(
        data, labels, test_size=0.4, random_state=0, stratify=labels
    )

    # define model
    # fix hyperparameters as needed to avoid unneeded grid search
    clf = SVC(kernel="linear", random_state=0)

    # based off sklearn example for hp tuning
    # https://scikit-learn.org/stable/modules/grid_search.html#

    # define hyper parameter space to check over
    # param_grid = [
    #     {'C': [1, 10, 100, 1000], 'kernel': ["linear", "poly", "sigmoid"]},
    #     {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001, "scale"], 'kernel': ['rbf']},
    # ]

    # basic = SVC(random_state=0).fit(data_train, label_train)
    #
    # sh = HalvingGridSearchCV(clf, param_grid, cv=5, factor=2).fit(data_train, label_train)
    # print(sh.best_estimator_)
    # clf = sh.best_estimator_

    # based on sklearn learning curve example
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    # plot learning curve for current model
    title = f"Learning Curves (SVM) ({DATASET_1_NAME})"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    plot_learning_curve(clf, title, data_train, label_train, ylim=(0, 1.01),
                        cv=cv, n_jobs=4)

    if fig_name is not None:
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
        plt.savefig(f"{fig_name}_learn_{dt_string}")

    # validation curve won't really work here, just use a table
    # for kernel in ["linear", "poly", "rbf", "sigmoid"]:
    #     split = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    #     training = []
    #     val = []
    #     for train_index, test_index in split.split(data_train, label_train):
    #         # train svc with given kernel
    #         test = SVC(kernel=kernel).fit(data_train[train_index], label_train[train_index])
    #         training.append(test.score(data_train[train_index], label_train[train_index]))
    #         val.append(test.score(data_train[test_index], label_train[test_index]))
    #     # cv sets to get average per kernel
    #     # dump results for table
    #     print(f"{kernel}: training: {np.mean(training)} {np.std(training)}, cv: {np.mean(val)} {np.std(val)}")

    if show_plots:
        plt.show()

    

    # run against test, uncomment for final analysis
    # !! don't touch during training/tuning !!
    # print(clf.score(data_test, label_test))

    # track accuracy and variance so later report can pull numbers as needed


def run_svm_2(fig_name = None, show_plots = False):
    # read in dataset from file
    with open(DATASET_2, 'r') as f:
        data = np.genfromtxt(f, delimiter=',')

    data, labels = data[:,:-1], data[:,-1]

    # split for training and testing
    data_train,data_test, label_train, label_test = train_test_split(
        data, labels, test_size=0.4, random_state=0, stratify=labels
    )

    # define model
    # fix hyperparameters as needed to avoid unneeded grid search
    clf = SVC(random_state=0)

    # based off sklearn example for hp tuning
    # https://scikit-learn.org/stable/modules/grid_search.html#

    # define hyper parameter space to check over
    # param_grid = {
    #     # at least kernels
    #     # C?
    #     # gamma?
    # }
    #
    # basic = SVC(random_state=0).fit(data_train, label_train)
    #
    # sh = GridSearchCV(clf, param_grid, cv=5,scoring="f1_weighted").fit(data_train, label_train)
    # print(sh.best_estimator_)
    # clf = sh.best_estimator_

    # based on sklearn learning curve example
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    # plot learning curve for current model
    title = f"Learning Curves (SVM) ({DATASET_2_NAME})"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    plot_learning_curve(clf, title, data_train, label_train, ylim=(0, 1.01),
                        scoring="f1_weighted", cv=cv, n_jobs=4)

    if fig_name is not None:
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
        plt.savefig(f"{fig_name}_learn_{dt_string}")

    # split and retrain to do a validation confusion matrix
    t = DecisionTreeClassifier(random_state=0)
    t_train, t_test, l_train, l_test = train_test_split(data_train, label_train, random_state=0)

    t.fit(t_train, l_train)

    plot_confusion_matrix(t, t_test, l_test)

    if fig_name is not None:
        plt.savefig(f"{fig_name}_conf_matrix_{dt_string}")

    # validation curve won't really work here, just use a table
    # for kernel in ["linear", "poly", "rbf", "sigmoid"]:
    #     split = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    #     training = []
    #     val = []
    #     for train_index, test_index in split.split(data_train, label_train):
    #         # train svc with given kernel
    #         test = SVC(kernel=kernel).fit(data_train[train_index], label_train[train_index])
    #         training.append(test.score(data_train[train_index], label_train[train_index]))
    #         val.append(test.score(data_train[test_index], label_train[test_index]))
    #     # cv sets to get average per kernel
    #     # dump results for table
    #     print(f"{kernel}: training: {np.mean(training)} {np.std(training)}, cv: {np.mean(val)} {np.std(val)}")

    if show_plots:
        plt.show()

    

    # run against test, uncomment for final analysis
    # !! don't touch during training/tuning !!
    # print(clf.score(data_test, label_test))

    # track accuracy and variance so later report can pull numbers as needed


def run_knn_1(fig_name = None, show_plots = False):
    # read in dataset from file
    with open(DATASET_1, 'r') as f:
        data = np.genfromtxt(f, delimiter=',')

    data, labels = data[:,:-1], data[:,-1]

    # split for training and testing
    data_train,data_test, label_train, label_test = train_test_split(
        data, labels, test_size=0.4, random_state=0, stratify=labels
    )

    # define model
    # fix hyperparameters as needed to avoid unneeded grid search
    clf = neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance', p=1)

    # 4 of the 5 folds use to train
    max_neigh = int(len(data_train)*0.8)

    # based off sklearn example for hp tuning
    # https://scikit-learn.org/stable/modules/grid_search.html#

    # define hyper parameter space to check over
    # param_grid = {
    #     # neighbors (k)
    #     "n_neighbors": range(1,max_neigh, 6),
    #     # weights (uniform, distance)
    #     "weights": ["uniform", "distance"],
    #     # p (1 = manhattan, 2 = euclidian)
    #     "p": [1,2]
    # }
    #
    # basic = neighbors.KNeighborsClassifier().fit(data_train, label_train)
    #
    # sh = GridSearchCV(clf, param_grid, cv=5, factor=2).fit(data_train, label_train)
    # print(sh.best_estimator_)
    # clf = sh.best_estimator_

    # based on sklearn learning curve example
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    # plot learning curve for current model
    title = f"Learning Curves (KNN) ({DATASET_1_NAME})"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    plot_learning_curve(clf, title, data_train, label_train, ylim=(0, 1.01),
                        cv=5, n_jobs=4)

    if fig_name is not None:
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
        plt.savefig(f"{fig_name}_learn_{dt_string}")


    # based off sklearn validation curve example
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html

    # plot validation curve
    title = f"Validation Curve with KNN ({DATASET_1_NAME})"
    x_lab = "Neighbors"
    y_lab = "Score"

    plot_validation_curve(clf, title, data_train, label_train, x_lab, y_lab, cv=5,
                          param_name="n_neighbors", param_range=range(1, 10), ylim=(0.0, 1.1))

    if fig_name is not None:
        plt.savefig(f"{fig_name}_val_{dt_string}")

    if show_plots:
        plt.show()

    

    # run against test, uncomment for final analysis
    # !! don't touch during training/tuning !!
    # print(clf.score(data_test, label_test))

    # track accuracy and variance so later report can pull numbers as needed


def run_knn_2(fig_name = None, show_plots = False):
    # read in dataset from file
    with open(DATASET_2, 'r') as f:
        data = np.genfromtxt(f, delimiter=',')

    data, labels = data[:,:-1], data[:,-1]

    # split for training and testing
    data_train,data_test, label_train, label_test = train_test_split(
        data, labels, test_size=0.4, random_state=0, stratify=labels
    )

    # define model
    # fix hyperparameters as needed to avoid unneeded grid search
    clf = neighbors.KNeighborsClassifier()

    # based off sklearn example for hp tuning
    # https://scikit-learn.org/stable/modules/grid_search.html#

    # define hyper parameter space to check over
    # param_grid = {
    #     # neighbors (k)
    #     "n_neighbors": range(1, max_neigh, 5),
    #    # weights (uniform, distance)
    #     "weights": ["uniform", "distance"],
    #     # p (1 = manhattan, 2 = euclidian)
    #     "p": [1, 2]
    # }
    #
    # basic = neighbors.KNeighborsClassifier().fit(data_train, label_train)
    #
    # # 4 of the 5 folds use to train
    max_neigh = int(len(data_train)*0.8)
    #
    # sh = GridSearchCV(clf, param_grid, scoring="f1_weighted", cv=5).fit(data_train, label_train)
    # print(sh.best_estimator_)
    # clf = sh.best_estimator_

    # based on sklearn learning curve example
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    # plot learning curve for current model
    title = f"Learning Curves (KNN) ({DATASET_2_NAME})"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    plot_learning_curve(clf, title, data_train, label_train, ylim=(0, 1.01), scoring="f1_weighted",
                        cv=cv, n_jobs=4)

    if fig_name is not None:
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
        plt.savefig(f"{fig_name}_learn_{dt_string}")


    # based off sklearn validation curve example
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html

    # plot validation curve
    title = f"Validation Curve with KNN ({DATASET_2_NAME})"
    x_lab = "Neighbors"
    y_lab = "Score"

    plot_validation_curve(clf, title, data_train, label_train, x_lab, y_lab, scoring="f1_weighted",
                          param_name="n_neighbors", param_range=range(1, max_neigh,5), ylim=(0.0, 1.1))

    if fig_name is not None:
        plt.savefig(f"{fig_name}_val_{dt_string}")

    # split and retrain to do a validation confusion matrix
    t = DecisionTreeClassifier(random_state=0)
    t_train, t_test, l_train, l_test = train_test_split(data_train, label_train, random_state=0)

    t.fit(t_train, l_train)

    plot_confusion_matrix(t, t_test, l_test)

    if fig_name is not None:
        plt.savefig(f"{fig_name}_conf_matrix_{dt_string}")

    if show_plots:
        plt.show()

    

    # run against test, uncomment for final analysis
    # !! don't touch during training/tuning !!
    # print(clf.score(data_test, label_test))

    # track accuracy and variance so later report can pull numbers as needed


def run_ann_1(fig_name = None, show_plots = False):
    # read in dataset from file
    with open(DATASET_1, 'r') as f:
        data = np.genfromtxt(f, delimiter=',')

    data, labels = data[:,:-1], data[:,-1]

    # split for training and testing
    data_train,data_test, label_train, label_test = train_test_split(
        data, labels, test_size=0.4, random_state=0, stratify=labels
    )

    # define model
    # fix hyperparameters as needed to avoid unneeded grid search
    # clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
    #                     solver='sgd', verbose=10, random_state=0,
    #                     learning_rate_init=.1)
    clf = MLPClassifier(hidden_layer_sizes=(85,), random_state=0)

    # based off sklearn example for hp tuning
    # https://scikit-learn.org/stable/modules/grid_search.html#

    # define hyper parameter space to check over
    # param_grid = {
    #     # hidden layers?
    #     # "hidden_layer_sizes": [(i,) for i in range(5,50,5)],
    #     # alpha
    #     "alpha": [1e-3,1e-4,1e-5],
    #     # learning rate
    #     "learning_rate_init": [1e-2,1e-3,1e-4],
    #     # momentum
    #     "momentum": np.linspace(.1,1.0,10)
    #     # solver
    # }
    #
    # basic = MLPClassifier().fit(data_train, label_train)
    #
    # with warnings.catch_warnings():
    #     warnings.filterwarnings("ignore", category=ConvergenceWarning,
    #                             module="sklearn")
    #     sh = HalvingGridSearchCV(clf, param_grid, cv=5, factor=2).fit(data_train, label_train)
    # print(sh.best_estimator_)
    # clf = sh.best_estimator_

    # based on sklearn learning curve example
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    # plot learning curve for current model
    title = f"Learning Curves (ANN) ({DATASET_1_NAME})"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                module="sklearn")
        plot_learning_curve(clf, title, data_train, label_train, ylim=(0, 1.01),
                        cv=5, n_jobs=4)

    if fig_name is not None:
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
        plt.savefig(f"{fig_name}_learn_{dt_string}")


    # based off sklearn validation curve example
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html

    # plot validation curve
    title = f"Validation Curve with ANN ({DATASET_1_NAME})"
    x_lab = "hidden_layer_sizes"
    y_lab = "Score"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                module="sklearn")
        plot_validation_curve(clf, title, data_train, label_train, x_lab, y_lab, cv=5,
                          param_name="hidden_layer_sizes", param_range=[(i,) for i in range(5,100,5)], ylim=(0.0, 1.1))

    if fig_name is not None:
        plt.savefig(f"{fig_name}_val_{dt_string}")

    if show_plots:
        plt.show()

    

    # run against test, uncomment for final analysis
    # !! don't touch during training/tuning !!
    # print(clf.score(data_test, label_test))

    # track accuracy and variance so later report can pull numbers as needed


def run_ann_2(fig_name = None, show_plots = False):
    # read in dataset from file
    with open(DATASET_2, 'r') as f:
        data = np.genfromtxt(f, delimiter=',')

    data, labels = data[:,:-1], data[:,-1]

    # split for training and testing
    data_train,data_test, label_train, label_test = train_test_split(
        data, labels, test_size=0.4, random_state=0, stratify=labels
    )

    # define model
    # fix hyperparameters as needed to avoid unneeded grid search
    clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                        solver='sgd', verbose=10, random_state=0,
                        learning_rate_init=.1)
    clf = MLPClassifier(random_state=0)

    # pulled from sklearn plot mnist example
    # this example won't converge because of CI's time constraints, so we catch the
    # warning and are ignore it here
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                module="sklearn")

    # based off sklearn example for hp tuning
    # https://scikit-learn.org/stable/modules/grid_search.html#

    # define hyper parameter space to check over
    # param_grid = {
    #     # hidden layers?
    #     "hidden_layer_sizes": [(i,) for i in range(5,50,5)]
    #     # alpha
    #     # learning rate
    #     # momentum
    #     # solver
    # }
    #
    # basic = MLPClassifier().fit(data_train, label_train)
    #
    # sh = HalvingGridSearchCV(clf, param_grid, scoring="f1_weighted", cv=5).fit(data_train, label_train)
    # print(sh.best_estimator_)
    # clf = sh.best_estimator_

    # based on sklearn learning curve example
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    # plot learning curve for current model
    title = f"Learning Curves (ANN) ({DATASET_2_NAME})"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    plot_learning_curve(clf, title, data_train, label_train, ylim=(0, 1.01),scoring="f1_weighted",
                        cv=cv, n_jobs=4)

    if fig_name is not None:
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
        plt.savefig(f"{fig_name}_learn_{dt_string}")


    # based off sklearn validation curve example
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html

    # plot validation curve
    title = f"Validation Curve with ANN ({DATASET_2_NAME})"
    x_lab = "hidden_layer_sizes"
    y_lab = "Score"

    plot_validation_curve(clf, title, data_train, label_train, x_lab, y_lab, scoring="f1_weighted",
                          param_name="hidden_layer_sizes", param_range=[(i,) for i in range(5,50,5)], ylim=(0.0, 1.1))

    if fig_name is not None:
        plt.savefig(f"{fig_name}_val_{dt_string}")

    # split and retrain to do a validation confusion matrix
    t = DecisionTreeClassifier(random_state=0)
    t_train, t_test, l_train, l_test = train_test_split(data_train, label_train, random_state=0)

    t.fit(t_train, l_train)

    plot_confusion_matrix(t, t_test, l_test)

    if fig_name is not None:
        plt.savefig(f"{fig_name}_conf_matrix_{dt_string}")

    if show_plots:
        plt.show()


def score_1(clf):
    # read in dataset from file
    with open(DATASET_1, 'r') as f:
        data = np.genfromtxt(f, delimiter=',')

    data, labels = data[:, :-1], data[:, -1]

    # random state makes this the same split
    # split for training and testing
    data_train, data_test, label_train, label_test = train_test_split(
        data, labels, test_size=0.4, random_state=0, stratify=labels
    )
    # train on all of training set, no cv or curves here
    clf.fit(data_train, label_train)

    # just score test set
    clf.score(data_test, label_test)


def score_2(clf):
    # read in dataset from file
    with open(DATASET_2, 'r') as f:
        data = np.genfromtxt(f, delimiter=',')

    data, labels = data[:, :-1], data[:, -1]

    # random state makes this the same split
    # split for training and testing
    data_train, data_test, label_train, label_test = train_test_split(
        data, labels, test_size=0.4, random_state=0, stratify=labels
    )

    # train on all of training set, no cv or curves here
    clf.fit(data_train, label_train)

    # just score test set
    clf.score(data_test, label_test)


if __name__ == '__main__':
    print()
    # dataset 1
    # run_dtc_1("charts/dtc/dtc_1_tunedepth", True)
    # run_dtc_1(show_plots=True)
    # run_ada_1("charts/ada/ada_1_tuneestimators", True)
    # run_ada_1(show_plots=True)
    # run_knn_1("charts/knn/knn_1_tune_k", show_plots=True)
    # run_knn_1(show_plots=True)
    # run_svm_1("charts/svm/svm_1_notune", show_plots=True)
    # run_svm_1(show_plots=True)

    # pulled from sklearn plot mnist example
    # this example won't converge because of CI's time constraints, so we catch the
    # warning and are ignore it here
    # with warnings.catch_warnings():
    #     warnings.filterwarnings("ignore", category=ConvergenceWarning,
    #                             module="sklearn")
        # run_ann_1("charts/ann/ann_1_notune", show_plots=True)
    # run_ann_1("charts/ann/ann_1_tune_layers", show_plots=True)

    # dataset 2
    # run_dtc_2("charts/dtc/dtc_wifi_notune", show_plots=True)
    # run_dtc_2(show_plots=True)
    # run_dtc_2(show_plots=True)
    # run_ada_2("charts/ada/ada_2_", show_plots=True)
    # run_ada_2(show_plots=True)
    run_svm_2("charts/svm/svm_2_notune", show_plots=True)
    # run_svm_2(show_plots=True)
    run_knn_2("charts/knn/knn_2_notune", show_plots=True)
    # run_knn_2(show_plots=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                module="sklearn")
        # run_ann_2("charts/ann/ann_2_notune", show_plots=True)
        run_ann_2(show_plots=True)

    print()
    plt.close('all')

    # print("scoring")
    # print(DATASET_1_NAME)
    # score_1(DecisionTreeClassifier())
    # score_1(AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, ccp_alpha=0.1),n_estimators=10, random_state=0))
    # score_1(SVC(kernel="linear", random_state=0))
    # score_1(neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance', p=1))
    # score_1(MLPClassifier(hidden_layer_sizes=(85,), random_state=0))
    # print()
    # print(DATASET_2_NAME)
    # score_2(DecisionTreeClassifier(criterion="entropy", max_depth=18, random_state=0))
    # score_2(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5), n_estimators=500, random_state=0))
    # score_2(SVC(gamma=0.001, C=1000, random_state=0))
    # score_2(neighbors.KNeighborsClassifier())
    # score_2(MLPClassifier())

