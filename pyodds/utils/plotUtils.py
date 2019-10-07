from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
sns.set(style="ticks")


def visualize_distribution(X,prediction,score,path):
    """
    Visualize the original density distribution of the data in 2-dimension space.

    Parameters
    ----------
    X: numpy array of shape (n_test, n_features)
        Test data.
    prediction: numpy array of shape (n_test, )
        The prediction result of the test data.
    score: numpy array of shape (n_test, )
        The outlier score of the test data.
    path: string
        The saving path for result figures.
    """

    sns.set(style="ticks")
    X=X.to_numpy()
    X_embedding = TSNE(n_components=2).fit_transform(X)
    sns_plot=sns.jointplot(X_embedding[:,1],X_embedding[:,0], kind="kde", space=0, color="#4CB391")
    sns_plot.savefig(path+'/distribution.png')


def visualize_distribution_static(X,prediction,score, path):
    """
    Visualize the original distribution of the data in 2-dimension space, which outliers/inliers are colored as differnet scatter plot.

    Parameters
    ----------
    X: numpy array of shape (n_test, n_features)
        Test data.
    prediction: numpy array of shape (n_test, )
        The prediction result of the test data.
    score: umpy array of shape (n_test, )
        The outlier score of the test data.
    path: string
        The saving path for result figures.
    """
    sns.set(style="darkgrid")

    X=X.to_numpy()
    X_embedding = TSNE(n_components=2).fit_transform(X)

    outlier_label=[]
    for i in range(len(X_embedding)):
        if prediction[i]==1:
            outlier_label.append('inlier')
        else:
            outlier_label.append('outlier')
    X_outlier = pd.DataFrame({'x_emb':X_embedding[:,0],'y_emb':X_embedding[:,1],'outlier_label':np.array(outlier_label),'score':np.array(score)})
    new_sns = sns.scatterplot(x="x_emb", y="y_emb",hue = "score", sizes =20, palette = 'BuGn_r',legend = False, data = X_outlier)
    new_sns.get_figure().savefig(path+'/distribution_withoutlier.png')



def visualize_distribution_time_serie(ts,value,path):
    """
    Visualize the time-serie data in each individual dimensions.

    Parameters
    ----------
    ts: numpy array of shape (n_test, n_features)
        The value of the test time serie data.
    value: numpy array of shape (n_test, )
        The outlier score of the test data.
    path: string
        The saving path for result figures.
    """
    sns.set(style="ticks")

    ts = pd.DatetimeIndex(ts)
    value=value.to_numpy()[:,1:]
    data = pd.DataFrame(value,ts)
    data = data.rolling(2).mean()
    sns_plot=sns.lineplot(data=data, palette="BuGn_r", linewidth=0.5)
    sns_plot.figure.savefig(path+'/timeserie.png')
    plt.show()



def visualize_outlierscore(value,label,contamination,path):
    """
    Visualize the predicted outlier score.

    Parameters
    ----------
    value: numpy array of shape (n_test, )
        The outlier score of the test data.
    label: numpy array of shape (n_test, )
        The label of test data produced by the algorithm.
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set,
        i.e. the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.
    path: string
        The saving path for result figures.
    """

    sns.set(style="darkgrid")

    ts = np.arange(len(value))
    outlier_label=[]
    for i in range(len(ts)):
        if label[i]==1:
            outlier_label.append('inlier')
        else:
            outlier_label.append('outlier')
    X_outlier = pd.DataFrame({'ts':ts,'Outlier_score':value,'outlier_label':np.array(outlier_label)})
    pal = dict(inlier="#4CB391", outlier="gray")
    g = sns.FacetGrid(X_outlier, hue="outlier_label", palette=pal, height=5)
    g.map(plt.scatter, "ts", "Outlier_score", s=30, alpha=.7, linewidth=.5, edgecolor="white")

    ranking = np.sort(value)
    threshold = ranking[int((1 - contamination) * len(ranking))]
    plt.hlines(threshold, xmin=0, xmax=len(X_outlier)-1, colors="g", zorder=100, label='Threshold')
    threshold = ranking[int((contamination) * len(ranking))]
    plt.hlines(threshold, xmin=0, xmax=len(X_outlier)-1, colors="g", zorder=100, label='Threshold2')
    plt.savefig(path+'/visualize_outlierscore.png')
    plt.show()



def visualize_outlierresult(X,label,path):
    """
    Visualize the predicted outlier result.

    Parameters
    ----------
    X: numpy array of shape (n_test, n_features)
        The test data.
    label: numpy array of shape (n_test, )
        The label of test data produced by the algorithm.

    """
    X['outlier']=pd.Series(label)
    pal = dict(inlier="#4CB391", outlier="gray")
    g = sns.pairplot(X, hue="outlier", palette=pal)
    plt.savefig(path+'/visualize_outlierresult.png')
    plt.show()
