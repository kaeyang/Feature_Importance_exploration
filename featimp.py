import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost
import shap
import seaborn as sns
import warnings

from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')


def pca_feat_importance(X, n, df):
    model = PCA(n_components=8).fit(X)
    X_pc = model.transform(X)
    n_pcs= model.components_.shape[0]
    most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]
    feature_names = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']].columns
    most_important_names = [feature_names[most_important[i]] for i in range(n_pcs)]
    dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}
    pca_df = pd.DataFrame(dic.items())
    return pca_df

def model_feat_selector(model, X, y):
    embeded_lr_selector = SelectFromModel(model, '1.25*median')
    embeded_lr_selector.fit(X, y)
    embeded_lr_support = embeded_lr_selector.get_support()
    embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()
    return embeded_lr_feature


def create_score_df(model, cv, spearman_x, pca_x, log_x, rf_x, xgb_x, y):
    mean_spearman_scores = np.around(cross_val_score(model, spearman_x, y, scoring='accuracy', cv=cv).mean(), 4)
    mean_pca_scores = np.around(cross_val_score(model, pca_x, y, scoring='accuracy', cv=cv).mean(), 4)
    mean_log_scores = np.around(cross_val_score(model, log_x, y, scoring='accuracy', cv=cv).mean(), 4)
    mean_rf_scores = np.around(cross_val_score(model, rf_x, y, scoring='accuracy', cv=cv).mean(), 4)
    mean_xgb_scores = np.around(cross_val_score(model, xgb_x, y, scoring='accuracy', cv=cv).mean(), 4)

    score_df = pd.DataFrame({
        'Method': ['Spearman Rank Correlation Coefficient', 'PCA', 'Logistic Feature Selector', 'Random Forest Feature Selector', 'XGBoost Feature Selector'],
        'accuracy_score': [mean_spearman_scores, mean_pca_scores, mean_log_scores, mean_rf_scores, mean_xgb_scores]
    })

    return score_df