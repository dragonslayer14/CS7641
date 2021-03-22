import warnings

import matplotlib
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.manifold import TSNE
from sklearn.metrics import plot_confusion_matrix, rand_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.model_selection import validation_curve
from datetime import datetime
import numpy as np
import itertools

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

import seaborn as sns
# show popup for graphs on mac

matplotlib.use("TKAgg")

import matplotlib.pyplot as plt

DATASET_1 = "data/wine.csv"
DATASET_2 = "data/winequality.csv"
DATASET_1_NAME = "Wine"
DATASET_2_NAME = "Wine Quality"


# these plots are only really necessary for ann tuning
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


# taken from https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
def plot_elbow(X, k_range = range(1, 10)):
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}

    for k in k_range:
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k, random_state=0).fit(X)
        kmeanModel.fit(X)

        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                            'euclidean'), axis=1)) / X.shape[0])
        inertias.append(kmeanModel.inertia_)

        mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                       'euclidean'), axis=1)) / X.shape[0]
        mapping2[k] = kmeanModel.inertia_

    for key, val in mapping1.items():
        print(f'{key} : {val}')

    plt.figure()
    plt.plot(k_range, distortions, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method using Distortion')
    plt.show()

    for key, val in mapping2.items():
        print(f'{key} : {val}')

    plt.figure()
    plt.plot(k_range, inertias, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method using Inertia')
    plt.show()


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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                module="sklearn")
        # plot_learning_curve(clf, title, data_train, label_train, ylim=(0, 1.01),
        #                 cv=5, n_jobs=4)

    if fig_name is not None:
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
        # plt.savefig(f"{fig_name}_learn_{dt_string}")


    # based off sklearn validation curve example
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html

    # plot validation curve
    title = f"Validation Curve with ANN ({DATASET_1_NAME})"
    x_lab = "Iterations"
    y_lab = "Score"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                module="sklearn")
        plot_validation_curve(clf, title, data_train, label_train, x_lab, y_lab, cv=5,
                          param_name="max_iter", param_range=range(10,210,10), ylim=(0.0, 1.1))

    if fig_name is not None:
        plt.savefig(f"{fig_name}_val_{dt_string}")

    if show_plots:
        plt.show()


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
    clf = MLPClassifier(hidden_layer_sizes=(290,), max_iter=110, random_state=0)

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
    #     "hidden_layer_sizes": [(i,j,) for i in range(10,210,10) for j in range(10,210,10)],
    #     # alpha
    #     # "alpha": [1e-3,1e-4,1e-5],
    #     # learning rate
    #     # "learning_rate_init": [1e-2,1e-3,1e-4],
    #     # momentum
    #     # "momentum": np.linspace(.1,1.0,10)
    #     # solver
    # }
    #
    basic = MLPClassifier().fit(data_train, label_train)
    #
    # sh = GridSearchCV(clf, param_grid, scoring="f1_weighted", cv=5, verbose=3).fit(data_train, label_train)
    # print(sh.best_estimator_)
    # clf = sh.best_estimator_

    # based on sklearn learning curve example
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    # plot learning curve for current model
    title = f"Learning Curves (ANN) ({DATASET_2_NAME})"

    plot_learning_curve(basic, title, data_train, label_train, ylim=(0, 1.01),scoring="f1_weighted",
                        cv=5, n_jobs=4)
    plot_learning_curve(clf, title, data_train, label_train, ylim=(0, 1.01),scoring="f1_weighted",
                        cv=5, n_jobs=4)

    if fig_name is not None:
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
        plt.savefig(f"{fig_name}_learn_{dt_string}")


    # based off sklearn validation curve example
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html

    # plot validation curve
    title = f"Validation Curve with ANN ({DATASET_2_NAME})"
    x_lab = "Iterations"
    y_lab = "Score"

    plot_validation_curve(clf, title, data_train, label_train, x_lab, y_lab, scoring="f1_weighted", cv=5,
                          param_name="max_iter", param_range=range(10,160,10), ylim=(0.0, 1.1))

    if fig_name is not None:
        plt.savefig(f"{fig_name}_val_{dt_string}")

    # split and retrain to do a validation confusion matrix
    t = MLPClassifier(hidden_layer_sizes=(290,), random_state=0)
    t_train, t_test, l_train, l_test = train_test_split(data_train, label_train, random_state=0)

    t.fit(t_train, l_train)

    plot_confusion_matrix(t, t_test, l_test)

    if fig_name is not None:
        plt.savefig(f"{fig_name}_conf_matrix_{dt_string}")

    if show_plots:
        plt.show()


def run_k_means_1():
    # plot "score" over range of k
    #   sum of squared distances
    # treat like MCC to narrow down optimal cluster number
    with open(DATASET_1, 'r') as f:
        data = np.genfromtxt(f, delimiter=',')

    data, labels = data[:,:-1], data[:,-1]

    # split for training and testing
    data_train, data_test, label_train, label_test = train_test_split(
        data, labels, test_size=0.4, random_state=0, stratify=labels
    )

    # plot_elbow(data_train, range(1,10))

    # with tuned number of clusters
    kmeans = KMeans(n_clusters=3, random_state=0).fit(data_train)

    label_pred = kmeans.predict(data_train)

    # check score to get an idea how well the clusters capture true values
    print(rand_score(label_train, label_pred))
    labels = kmeans.labels_
    print(davies_bouldin_score(data_train, labels))

    tsne = TSNE()
    X_embedded = tsne.fit_transform(data_train)

    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=label_train, legend='full',
                    palette=sns.color_palette("bright", 3))
    print()

    # TODO feed into ANN


def run_k_means_2():
    # plot "score" over range of k
    #   sum of squared distances
    # treat like MCC to narrow down optimal cluster number
    with open(DATASET_2, 'r') as f:
        data = np.genfromtxt(f, delimiter=',')

    data, labels = data[:,:-1], data[:,-1]

    # split for training and testing
    data_train, data_test, label_train, label_test = train_test_split(
        data, labels, test_size=0.4, random_state=0, stratify=labels
    )

    # plot_elbow(data_train, range(1,10))

    # with tuned number of clusters
    label_pred = KMeans(n_clusters=5, random_state=0).fit_predict(data_train)

    # check score to get an idea how well the clusters capture true values
    print(rand_score(label_train, label_pred))

    # TODO feed into ANN


def run_em_1():

    with open(DATASET_1, 'r') as f:
        data = np.genfromtxt(f, delimiter=',')

    data, labels = data[:,:-1], data[:,-1]

    # split for training and testing
    data_train, data_test, label_train, label_test = train_test_split(
        data, labels, test_size=0.4, random_state=0, stratify=labels
    )

    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 21)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(n_components=n_components, covariance_type=cv_type, random_state=0)
            gmm.fit(data_train)
            bic.append(gmm.bic(data_train))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                  'darkorange'])
    clf = best_gmm
    bars = []

    # Plot the BIC scores
    plt.figure()
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + \
           .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    plt.xlabel('Number of components')
    plt.legend([b[0] for b in bars], cv_types)

    plt.title(f'Selected GMM: {best_gmm.covariance_type} model, '
              f'{best_gmm.n_components} components')
    plt.show()

    # TODO feed into ANN


def run_em_2():

    with open(DATASET_2, 'r') as f:
        data = np.genfromtxt(f, delimiter=',')

    data, labels = data[:,:-1], data[:,-1]

    # split for training and testing
    data_train, data_test, label_train, label_test = train_test_split(
        data, labels, test_size=0.4, random_state=0, stratify=labels
    )

    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 21)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(n_components=n_components, covariance_type=cv_type, random_state=0)
            gmm.fit(data_train)
            bic.append(gmm.bic(data_train))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                  'darkorange'])
    clf = best_gmm
    bars = []

    # Plot the BIC scores
    plt.figure()
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + \
           .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    plt.xlabel('Number of components')
    plt.legend([b[0] for b in bars], cv_types)

    plt.title(f'Selected GMM: {best_gmm.covariance_type} model, '
              f'{best_gmm.n_components} components')
    plt.show()

    # TODO feed into ANN

if __name__ == '__main__':

    # clustering
    # run_k_means_1()
    # run_k_means_2()
    run_em_1()
    # run_em_2()

    print()

    # dimensionality reduction
    # run_pca_1()
    # run_pca_2()
    # run_ica_1()
    # run_ica_2()
    # run_rca_1()
    # run_rca_2()
    # run_lda_1()
    # run_lda_2()

    # ANN work can just be done in the calls
    # no sense making another place to do the work with one more step

    # combination experiments, DR + clustering

    # PCA
    # run_pca_kmeans_1()
    # run_pca_kmeans_2()
    # run_pca_em_1()
    # run_pca_em_2()

    # ICA
    # run_ica_kmeans_1()
    # run_ica_kmeans_2()
    # run_ica_em_1()
    # run_ica_em_2()

    # RCA
    # run_rca_kmeans_1()
    # run_rca_kmeans_2()
    # run_rca_em_1()
    # run_rca_em_2()

    # LDA
    # run_lda_kmeans_1()
    # run_lda_kmeans_2()
    # run_lda_em_1()
    # run_lda_em_2()


    # dataset 1
    run_ann_1("charts/ann_1_final", show_plots=False)

    plt.close('all')

    # dataset 2
    run_ann_2("charts/ann_2_final", show_plots=False)

    plt.close('all')

    if False:
        print("scoring")
        print(DATASET_1_NAME)
        # read in dataset from file
        with open(DATASET_1, 'r') as f:
            data = np.genfromtxt(f, delimiter=',')

        data, labels = data[:, :-1], data[:, -1]

        # random state makes this the same split
        # split for training and testing
        data_train, data_test, label_train, label_test = train_test_split(
            data, labels, test_size=0.4, random_state=0, stratify=labels
        )

        for clf in [
            DecisionTreeClassifier(criterion="entropy", random_state=0),
            DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0),
            AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, ccp_alpha=0.1),n_estimators=10, random_state=0),
            SVC(kernel="linear", random_state=0),
            neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance', p=1),
            MLPClassifier(hidden_layer_sizes=(85,), random_state=0)
        ]:
            start = time.time()
            # train on all of training set, no cv or curves here
            clf.fit(data_train, label_train)

            # just score test set
            print(f"{clf}\t{clf.score(data_test, label_test):.2f}\t{time.time() - start:.2f}")

        print()
        print(DATASET_2_NAME)

        with open(DATASET_2, 'r') as f:
            data = np.genfromtxt(f, delimiter=',')

        data, labels = data[:, :-1], data[:, -1]

        # random state makes this the same split
        # split for training and testing
        data_train, data_test, label_train, label_test = train_test_split(
            data, labels, test_size=0.4, random_state=0, stratify=labels
        )

        for clf in [
            DecisionTreeClassifier(criterion="entropy", max_depth=18, random_state=0),
            AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5), n_estimators=500, random_state=0),
            SVC(gamma=0.001, C=10000, random_state=0),
            neighbors.KNeighborsClassifier(leaf_size=10, n_neighbors=14, p=1, weights='distance'),
            MLPClassifier(hidden_layer_sizes=(290,), max_iter=110, random_state=0)
        ]:
            start = time.time()
            # train on all of training set, no cv or curves here
            clf.fit(data_train, label_train)

            # just score test set
            print(f"{clf}\t{clf.score(data_test, label_test):.2f}\t{time.time() - start:.2f}")

