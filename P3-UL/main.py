import time
import warnings

import matplotlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from scipy.stats import kurtosis
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.manifold import TSNE
from sklearn.metrics import plot_confusion_matrix, rand_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import learning_curve, train_test_split, HalvingGridSearchCV
from sklearn.model_selection import validation_curve
from datetime import datetime
import numpy as np
import itertools

from numpy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

import seaborn as sns
# show popup for graphs on mac
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection

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
def plot_elbow(X, k_range = None):
    if k_range is None:
        k_range = range(1, 10)
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

    for key, val in mapping2.items():
        print(f'{key} : {val}')

    plt.figure()
    plt.plot(k_range, inertias, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method using Inertia')


def run_ann(data_train, label_train, data_test, label_test, algo_name, data_name, fig_name = None, show_plots = False,
            plot_learning = False, plot_val=False, val_param="max_iter",val_range=range(10,210,10), test=False,
            val_lab = "Iterations", grid_search=False, **kwargs):

    if grid_search:
        # based off sklearn example for hp tuning
        # https://scikit-learn.org/stable/modules/grid_search.html#

        # define hyper parameter space to check over
        param_grid = {
            # alpha
            "alpha": [1e-3,1e-4,1e-5],
            # learning rate
            "learning_rate_init": [1e-2,1e-3,1e-4]
        }
        clf = MLPClassifier(hidden_layer_sizes=(85,), **kwargs, max_iter=1000, early_stopping=True, random_state=0)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                    module="sklearn")
            sh = HalvingGridSearchCV(clf, param_grid, cv=5, factor=2).fit(data_train, label_train)
        print(sh.best_estimator_)
    clf = MLPClassifier(hidden_layer_sizes=(85,), max_iter=500, random_state=0, early_stopping=True, **kwargs).fit(data_train, label_train)
    # based on sklearn learning curve example
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    if plot_learning:
        # plot learning curve for current model
        title = f"Learning Curves (ANN {algo_name}) ({data_name})"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                    module="sklearn")
            plot_learning_curve(clf, title, data_train, label_train, ylim=(0, 1.01),
                            cv=5, n_jobs=4)

        if fig_name is not None:
            now = datetime.now()
            dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
            plt.savefig(f"{fig_name}_learn_{dt_string}")

    if plot_val:
        # based off sklearn validation curve example
        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html

        # plot validation curve
        title = f"Validation Curve with ANN ({algo_name}) ({data_name})"
        x_lab = val_lab
        y_lab = "Score"

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                    module="sklearn")
            plot_validation_curve(clf, title, data_train, label_train, x_lab, y_lab, cv=5,
                              param_name=val_param, param_range=val_range, ylim=(0.0, 1.1))

        if fig_name is not None:
            now = datetime.now()
            dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
            plt.savefig(f"{fig_name}_val_{dt_string}")

    if show_plots:
        plt.show()

    if test:
        print(f"ANN ({algo_name}) score: {clf.score(data_test, label_test)}")


def run_k_means(data_train, label_train, data_test=None, k_range = None, n_clusters = None):
    # plot "score" over range of k
    #   sum of squared distances
    # treat like MCC to narrow down optimal cluster number
    if n_clusters is None:
        plot_elbow(data_train, k_range)
        return data_train

    # with tuned number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data_train)
    label_pred = kmeans.predict(data_train)

    # check score to get an idea how well the clusters capture true values
    print(f"kmeans score: {rand_score(label_train, label_pred)}")

    train_transform = kmeans.transform(data_train)
    transform_with_pred = np.append(train_transform, [[x] for x in label_pred], axis=1)

    if data_test is None:
        return transform_with_pred
    else:
        test_transform = kmeans.transform(data_test)
        test_predict = kmeans.predict(data_test)
        return transform_with_pred,\
               np.append(test_transform, [[x] for x in test_predict], axis=1)

    # TODO figure out TSNE
    tsne = TSNE()
    X_embedded = tsne.fit_transform(data_train)

    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=label_train, legend='full',
                    palette=sns.color_palette("bright", 3))


def plot_bic_scores(data, n_components_range = None):

    if n_components_range is None:
        n_components_range = range(1, 21)

    lowest_bic = np.infty
    bic = []
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(n_components=n_components, covariance_type=cv_type, random_state=0)
            gmm.fit(data)
            bic.append(gmm.bic(data))
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


def run_em(data_train, label_train, data_test=None, n_components_range = None, n_components=None, covariance_type=None):

    if n_components is None or covariance_type is None:
        plot_bic_scores(data_train, n_components_range)
        return data_train

    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=0).fit(data_train)
    label_pred = gmm.predict(data_train)

    # check score to get an idea how well the clusters capture true values
    print(f"em score: {rand_score(label_train, label_pred)}")

    prediction = np.append(gmm.predict_proba(data_train), [[x] for x in label_pred], axis=1)

    if data_test is None:
        return prediction
    else:
        test_predict_prob = gmm.predict_proba(data_test)
        test_predict = gmm.predict(data_test)

        return prediction, np.append(test_predict_prob, [[x] for x in test_predict], axis=1)


# taken from https://www.kaggle.com/lonewolf95/classification-tutorial-with-pca-and-gridsearchcv
def plot_pca_curve(data):
    scaler = StandardScaler()
    scaler.fit(data)
    x_train_scaler = scaler.transform(data)
    pca = PCA(random_state=0)
    pca.fit(x_train_scaler)
    cumsum = np.cumsum(pca.explained_variance_ratio_) * 100
    d = [n for n in range(len(cumsum))]
    plt.figure(figsize=(10, 10))
    plt.plot(d, cumsum, color='red', label='cumulative explained variance')
    plt.title('Cumulative Explained Variance as a Function of the Number of Components')
    plt.ylabel('Cumulative Explained variance')
    plt.xlabel('Principal components')
    plt.axhline(y=95, color='k', linestyle='--', label='95% Explained Variance')
    plt.legend(loc='best')

    # reconstruction error by components
    recon_errs = []
    sizes = range(1,12)
    for size in sizes:
        pca = PCA(n_components=size, random_state=0)
        transformed_data = pca.fit_transform(x_train_scaler)
        inverse_data = np.linalg.pinv(pca.components_.T)
        reconstructed_data = transformed_data.dot(inverse_data)
        loss = ((x_train_scaler - reconstructed_data) ** 2).mean()
        recon_errs.append(loss)

    plt.figure()
    plt.title('recon error by Number of Components')
    plt.ylabel('recon error')
    plt.xlabel('Principal components')
    plt.plot(sizes, recon_errs)


def run_pca(data_train, data_test, components = None):
    # tuning
    if components is None:
        plot_pca_curve(data_train)
        print()
        return data_train

    else:
        pca = PCA(n_components=components, random_state=0)

        scaler = StandardScaler()
        scaler.fit(data_train)
        x_train_scaler = scaler.transform(data_train)

        # get reconstruction error score
        transformed_data = pca.fit_transform(x_train_scaler)
        inverse_data = np.linalg.pinv(pca.components_.T)
        reconstructed_data = transformed_data.dot(inverse_data)

        loss = ((x_train_scaler - reconstructed_data) ** 2).mean()
        print(f"PCA loss: {loss}")
        x_test_scaler = scaler.transform(data_test)
        transformed_test = pca.transform(x_test_scaler)

        return transformed_data, transformed_test


def plot_ica_curve(data, n_components_range = None):

    if n_components_range is None:
        n_components_range = range(1, data.shape[1])

    highest_kurtosis = -1*10**4
    kurtosis_vals = []
    plt.figure()

    scaler = StandardScaler()
    scaler.fit(data)
    x_train_scaler = scaler.transform(data)

    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        ica = FastICA(n_components=n_components, max_iter=500, random_state=0)
        transformed = ica.fit_transform(x_train_scaler)
        kurt_values = kurtosis(transformed, axis=0)
        kurtosis_vals.append(np.mean(kurt_values))
        if kurtosis_vals[-1] > highest_kurtosis:
            highest_kurtosis = kurtosis_vals[-1]
            best_gmm = ica

    kurtosis_vals = np.array(kurtosis_vals)

    bars = []

    # this probably doesn't work as is

    # Plot the BIC scores
    plt.plot(n_components_range, kurtosis_vals, label="kurtosis")
    plt.title('average kurtosis per model')
    xpos = np.mod(kurtosis_vals.argmin(), len(n_components_range)) + .65 + \
           .2 * np.floor(kurtosis_vals.argmin() / len(n_components_range))
    plt.text(xpos, 1.03 * kurtosis_vals.max(), '*', fontsize=14)
    plt.xlabel('Number of components')
    plt.xticks(n_components_range)
    plt.legend()

    plt.title(f'Selected ica: {best_gmm.n_components} components')

    kurtosis_vals = kurtosis(best_gmm.fit_transform(x_train_scaler), axis=0)

    plt.figure()
    bars = []
    plt.plot(kurtosis_vals, label="kurtosis")

    plt.xticks(range(n_components_range.stop))
    plt.title('kurtosis per component')
    xpos = np.mod(kurtosis_vals.argmin(), len(n_components_range)) + .65 + \
           .2 * np.floor(kurtosis_vals.argmin() / len(n_components_range))
    plt.text(xpos, 1.03 * kurtosis_vals.max(), '*', fontsize=14)
    plt.xlabel('Component number')
    plt.legend()


def run_ica(data_train, data_test, n_components = None, n_components_range = None):

    if n_components is None:
        plot_ica_curve(data_train, n_components_range)
        print()
        return data_train
    else:
        scaler = StandardScaler()
        scaler.fit(data_train)
        x_train_scaler = scaler.transform(data_train)

        ica = FastICA(n_components=n_components, max_iter=500, random_state=0)
        transformed = ica.fit_transform(x_train_scaler)

        # get reconstruction error
        inverse_data = np.linalg.pinv(ica.components_.T)
        reconstructed_data = transformed.dot(inverse_data)

        loss = ((x_train_scaler - reconstructed_data) ** 2).mean()
        print(f"ICA loss: {loss}")

        # pull the identified components out of full data
        # pull top n components with highest kurtosis as the subset
        kurtosis_vals = [(val,i) for val,i in zip(kurtosis(transformed, axis=0),range(0,data.shape[1]))]

        kurtosis_vals.sort(key=lambda x:-x[0])

        # 4 from visual on graph
        top_n = [x[1] for x in kurtosis_vals[:4]]

        print(f"pulling components {top_n}")

        x_test_scaler = scaler.transform(data_test)
        transformed_test = ica.transform(x_test_scaler)

        return transformed[:,top_n], transformed_test[:,top_n]


def plot_rca_curve(data):
    scaler = StandardScaler()
    scaler.fit(data)
    x_train_scaler = scaler.transform(data)

    # reconstruction error by components
    recon_errs = []
    sizes = range(1,12)
    for size in sizes:
        rca = GaussianRandomProjection(n_components=size, random_state=0)
        transformed_data = rca.fit_transform(x_train_scaler)
        inverse_data = np.linalg.pinv(rca.components_.T)
        reconstructed_data = transformed_data.dot(inverse_data)
        loss = ((x_train_scaler - reconstructed_data) ** 2).mean()
        recon_errs.append(loss)

    plt.figure()
    plt.title('recon error by Number of Components')
    plt.ylabel('recon error')
    plt.xlabel('Components')
    plt.plot(sizes, recon_errs)


def run_rca(data_train, data_test, n_components = None):
    if n_components is None:
        plot_rca_curve(data_train)
        print()
        return data_train
    else:
        rca = GaussianRandomProjection(n_components=n_components, random_state=0)

        scaler = StandardScaler()
        scaler.fit(data_train)
        x_train_scaler = scaler.transform(data_train)

        # get reconstruction error score
        transformed_data = rca.fit_transform(x_train_scaler)
        inverse_data = np.linalg.pinv(rca.components_.T)
        reconstructed_data = transformed_data.dot(inverse_data)

        loss = ((x_train_scaler - reconstructed_data) ** 2).mean()
        print(f"RCA loss: {loss}")
        x_test_scaler = scaler.transform(data_test)
        transformed_test = rca.transform(x_test_scaler)

        return transformed_data, transformed_test


def run_lda(data_train, label_train, data_test, **kwargs):
    scaler = StandardScaler()
    scaler.fit(data_train)
    x_train_scaler = scaler.transform(data_train)

    x_test_scaler = scaler.transform(data_test)

    if len(kwargs) == 0:
        plot_validation_curve(LinearDiscriminantAnalysis(solver="svd"), "LDA tolerance", x_train_scaler, label_train,
                              param_name="tol", param_range=np.linspace(1.0e-6,1.0e-3),
                              x_lab="tolerance", y_lab="accuracy")
        plot_learning_curve(LinearDiscriminantAnalysis(solver="svd"), "LDA learning curve", x_train_scaler, label_train)
        return x_train_scaler, x_test_scaler

    lda = LinearDiscriminantAnalysis(**kwargs)
    transformed_data = lda.fit_transform(x_train_scaler, label_train)

    print(f"LDA score: {lda.score(x_train_scaler, label_train)}")

    transformed_test = lda.transform(x_test_scaler)

    return transformed_data, transformed_test


def plot_first_2_dim(data, true_labels, algo_name):
    plt.figure()
    lw = 2
    label_space = [int(x) for x in set(true_labels)]
    colors = ['navy', 'turquoise', 'darkorange', 'red', 'green', 'purple', 'gold', 'cyan']
    for color, i in zip(colors, label_space):
        plt.scatter(data[true_labels == i, 0], data[true_labels == i, 1], alpha=.8, lw=lw,
                    label=i)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title(algo_name)


if __name__ == '__main__':
    # TODO plots for description

    # todo change ann to take in data and tuning params

    start = time.time()

    with open(DATASET_1, 'r') as f:
        data = np.genfromtxt(f, delimiter=',')

    data, labels = data[:,:-1], data[:,-1]

    # split for training and testing
    data_train_1, data_test_1, label_train_1, label_test_1 = train_test_split(
        data, labels, test_size=0.4, random_state=0, stratify=labels
    )

    with open(DATASET_2, 'r') as f:
        data = np.genfromtxt(f, delimiter=',')

    data, labels = data[:,:-1], data[:,-1]

    # split for training and testing
    data_train_2, data_test_2, label_train_2, label_test_2 = train_test_split(
        data, labels, test_size=0.4, random_state=0, stratify=labels
    )

    # clustering
    # k_means_1_train, k_means_1_test = run_k_means(data_train_1, label_train_1, data_test_1, n_clusters=3)
    # k_means_2_train, k_means_2_test = run_k_means(data_train_2, label_train_2, data_test_2, n_clusters=5)
    # em_1_train, em_1_test = run_em(data_train_1, label_train_1, data_test_1, n_components=3, covariance_type="diag")
    # em_2_train, em_2_test = run_em(data_train_2, label_train_2, data_test_2, n_components=12, covariance_type="full")

    # dimensionality reduction
    pca_1_train, pca_1_test = run_pca(data_train_1, data_test_1, components=6)
    pca_2_train, pca_2_test = run_pca(data_train_2, data_test_2, components=6)
    ica_1_train, ica_1_test = run_ica(data_train_1, data_test_1, n_components=12)
    ica_2_train, ica_2_test = run_ica(data_train_2, data_test_2, n_components=10)
    rca_1_train, rca_1_test = run_rca(data_train_1, data_test_1, n_components=10)
    rca_2_train, rca_2_test = run_rca(data_train_2, data_test_2, n_components=9)
    lda_1_train, lda_1_test = run_lda(data_train_1, label_train_1, data_test_1, solver="svd")
    lda_2_train, lda_2_test = run_lda(data_train_2, label_train_2, data_test_2, solver="svd")

    plot_first_2_dim(pca_1_train, label_train_1, "pca 1")
    plt.savefig("charts/pca_1_first_2_dim")
    plot_first_2_dim(pca_2_train, label_train_2, "pca 2")
    plt.savefig("charts/pca_2_first_2_dim")
    plot_first_2_dim(ica_1_train, label_train_1, "ica 1")
    plt.savefig("charts/ica_1_first_2_dim")
    plot_first_2_dim(ica_2_train, label_train_2, "ica 2")
    plt.savefig("charts/ica_2_first_2_dim")
    plot_first_2_dim(rca_1_train, label_train_1, "rca 1")
    plt.savefig("charts/rca_1_first_2_dim")
    plot_first_2_dim(rca_2_train, label_train_2, "rca 2")
    plt.savefig("charts/rca_2_first_2_dim")
    plot_first_2_dim(lda_1_train, label_train_1, "lda 1")
    plt.savefig("charts/lda_1_first_2_dim")
    plot_first_2_dim(lda_2_train, label_train_2, "lda 2")
    plt.savefig("charts/lda_2_first_2_dim")
    # combination experiments, DR + clustering

    # PCA
    # pca_kmeans_1 = run_k_means(pca_1_train, label_train_1, n_clusters=3)
    # pca_kmeans_2 = run_k_means(pca_2_train, label_train_2, n_clusters=5)
    # pca_em_1 = run_em(pca_1_train, label_train_1, n_components=14, covariance_type="spherical")
    # pca_em_2 = run_em(pca_2_train, label_train_2, n_components=12, covariance_type="full")

    # ICA
    # ica_kmeans_1 = run_k_means(ica_1_train, label_train_1, n_clusters=10)
    # ica_kmeans_2 = run_k_means(ica_2_train, label_train_2, n_clusters=13)
    # ica_em_1 = run_em(ica_1_train, label_train_1, n_components=5, covariance_type="spherical")
    # ica_em_2 = run_em(ica_2_train, label_train_2, n_components=9, covariance_type="diag")

    # RCA
    # rca_kmeans_1 = run_k_means(rca_1_train, label_train_1, n_clusters=3)
    # rca_kmeans_2 = run_k_means(rca_2_train, label_train_2, n_clusters=4)
    # rca_em_1 = run_em(rca_1_train, label_train_1, n_components=20, covariance_type="full")
    # rca_em_2 = run_em(rca_2_train, label_train_2, n_components=10, covariance_type="full")

    # LDA
    # lda_kmeans_1 = run_k_means(lda_1_train, label_train_1, n_clusters=3)
    # lda_kmeans_2 = run_k_means(lda_2_train, label_train_2, n_clusters=7)
    # lda_em_1 = run_em(lda_1_train, label_train_1, covariance_type="tied", n_components=6)
    # lda_em_2 = run_em(lda_2_train, label_train_2, covariance_type="full", n_components=7)

    # one hot encode clusters for kmeans
    # probability of each cluster for em
    # dr will just be the transformed data and original labels

    # can create table of final model performance

    # dataset 1

    # run_ann(k_means_2_train, label_train_2, k_means_2_test, label_test_2, algo_name="kmeans", data_name=DATASET_2_NAME,
    #         plot_learning=False, fig_name="charts/ann_kmeans_2_alpha", show_plots=False, alpha=1e-05,
    #         learning_rate_init=0.01,
    #         test=True)
    # run_ann(em_2_train, label_train_2, em_2_test, label_test_2, algo_name="em", data_name=DATASET_2_NAME,
    #         plot_learning=False, fig_name="charts/ann_em_2_alpha", show_plots=False, learning_rate_init=0.01,
    #         test=True)
    # run_ann(pca_2_train, label_train_2, pca_2_test, label_test_2, algo_name="pca", data_name=DATASET_2_NAME,
    #         plot_learning=False, fig_name="charts/ann_pca_2_alpha", show_plots=False, alpha=0.001,
    #         learning_rate_init=0.01,
    #         test=True)
    # run_ann(ica_2_train, label_train_2, ica_2_test, label_test_2, algo_name="ica", data_name=DATASET_2_NAME,
    #         plot_learning=False, fig_name="charts/ann_ica_2_alpha", show_plots=False, learning_rate_init=0.01,
    #         test=True)
    # run_ann(rca_2_train, label_train_2, rca_2_test, label_test_2, algo_name="rca", data_name=DATASET_2_NAME,
    #         plot_learning=False, fig_name="charts/ann_rca_2_alpha", show_plots=False, learning_rate_init=0.01,
    #         test=True)
    # run_ann(lda_2_train, label_train_2, lda_2_test, label_test_2, algo_name="lda", data_name=DATASET_2_NAME,
    #         plot_learning=False, fig_name="charts/ann_lda_2_alpha", show_plots=False, learning_rate_init=0.01,
    #         test=True)

    print(f"took {time.time() - start:.2f} seconds")
    plt.show()
    print()
    plt.close('all')
    # TODO set up passing plot names to methods to save

