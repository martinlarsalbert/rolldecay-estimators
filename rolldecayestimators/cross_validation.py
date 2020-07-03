import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def model_filter(group, models):
    return group.name in models

def cross_validates(model, data, features, label='B_e_hat', n_splits=5, itterations=10):
    scores = []
    for i in range(itterations):
        scores_ = cross_validate(model, data, features=features, label=label, n_splits=n_splits)
        scores.append(scores_)

    return np.array(scores)


def cross_validate(model, data, features, label='B_e_hat', n_splits=5):
    groups_model = data.groupby(by='model_number')
    models = data['model_number'].unique()
    np.random.shuffle(models)  # Inplace

    kf = KFold(n_splits=n_splits)
    scores = []
    model_test = clone(model)
    for train_index, test_index in kf.split(models):
        models_train = models[train_index]
        models_test = models[test_index]

        data_train = groups_model.filter(func=model_filter, models=models_train)
        data_test = X_test = groups_model.filter(func=model_filter, models=models_test)

        X_train = data_train[features]
        X_test = data_test[features]

        y_train = data_train[label]
        y_test = data_test[label]

        model_test.fit(X=X_train, y=y_train)
        score = model_test.score(X=X_test, y=y_test)
        scores.append(score)


    return scores


def plot_validate(model, data, features, label='B_e_hat', n_splits=5):
    groups_model = data.groupby(by='model_number')
    models = data['model_number'].unique()
    np.random.shuffle(models)  # Inplace

    kf = KFold(n_splits=n_splits)
    model_test = clone(model)

    nrows=2
    ncols = int(np.ceil(n_splits/nrows))

    fig,axes=plt.subplots(ncols=ncols, nrows=nrows)

    axess=axes.flatten()

    not_used = axess[n_splits:]
    for ax in not_used:
        ax.remove()

    axess=axess[0:n_splits]


    fold=0
    for (train_index, test_index),ax in zip(kf.split(models),axess):
        models_train = models[train_index]
        models_test = models[test_index]

        data_train = groups_model.filter(func=model_filter, models=models_train)
        data_test = X_test = groups_model.filter(func=model_filter, models=models_test)

        X_train = data_train[features]
        X_test = data_test[features]

        y_train = data_train[label]
        y_test = data_test[label]

        model_test.fit(X=X_train, y=y_train)
        ax.plot(y_test, model_test.predict(X=X_test),'.')

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        lim = np.max([xlim[1], ylim[1]])
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title('Test fold %i' % fold)
        fold+=1
        ax.plot([0, lim], [0, lim], 'r-')
        ax.set_aspect('equal', 'box')
        ax.grid(True)

    #axes[0].legend()
    axess[0].set_ylabel('$\hat{B_e}$ (prediction)')
    axess[3].set_ylabel('$\hat{B_e}$ (model test)')
    axess[3].set_xlabel('$\hat{B_e}$ (model test)')
    axess[4].set_xlabel('$\hat{B_e}$ (model test)')

    return fig