import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
import joblib

dfcolors = pd.read_csv('colors_df_green_trained.csv',index_col='i')

def train_classifier(dfcolors):
    X, y = dfcolors[['0','1','2']], dfcolors['label']

    grid = {
            'C': np.power(10.0, np.arange(-10, 10)),
            'solver': ['newton-cg','lbfgs','sag']
        }
    clf = LogisticRegression(random_state=0, max_iter=10000, tol=5)
    gs = GridSearchCV(clf, grid, scoring='accuracy', cv=5)
    gs.fit(X, y)
    print ('gs.best_score_:', gs.best_score_)

    joblib.dump(gs, 'clf.pkl')
    return gs

clf = joblib.load('clf.pkl')

from script import *
path = 'photos/h600_zoom17.png'

def calculate_green(path):
    image = load_image(path, show=True)
    km, centroids = image_kcolors(20, image)
    pcts = kcolors_pct(km, centroids, show=True)
    ix = np.where(clf.predict(centroids)==1)
    return np.sum([pcts[i] for i in ix])

calculate_green(path)
