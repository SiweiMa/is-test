import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def is_test(train, test, classifier):
    """
    create a maked up column called is_test for the dataset combining train and test set
    train the classifier on the data before and after shuffle.
    we can shuffle multiple times (10x here) to get the average training metrics
    print the training metrics before shuffle and average training metrics after shuffle
    """
    synthetic_data = pd.concat([train, test], axis=0)
    is_test_col = np.hstack((np.zeros(len(train)), np.ones(len(test))))

    synthetic_data['is_test'] = is_test_col
    PR, f1 = classfify(synthetic_data, classifier)
    print(f"the PR before shuffle is {PR} "
          f"the f1 score before shuffle is {f1}")

    PR_after_shuffle = []
    f1_after_shuffle = []
    for _ in range(10):
        np.random.shuffle(is_test_col)
        synthetic_data['is_test'] = is_test_col
        PR, f1 = classfify(synthetic_data, classifier)
        PR_after_shuffle.append(PR)
        f1_after_shuffle.append(f1)

    print(f"the average PR after shuffle is {np.mean(PR_after_shuffle)} "
          f"the average f1 score after shuffle is {np.mean(f1_after_shuffle)}")


def classfify(synthetic_data, classifier):
    """
    train the classifier on synthetic data and output the training metrics
    """
    X = synthetic_data.drop('is_test', axis=1)
    y = synthetic_data['is_test'].values

    classifier = classifier(n_jobs=-1, max_depth=10, min_samples_leaf=5)
    classifier.fit(X, y)
    y_pred = classifier.predict(X)
    y_pred_proba = classifier.predict_proba(X)[:, 1]
    precision, recall, _ = metrics.precision_recall_curve(y, y_pred_proba, pos_label=1)
    PR = metrics.auc(recall, precision)
    f1 = metrics.accuracy_score(y, y_pred)
    return PR, f1


data = pd.read_csv('https://raw.githubusercontent.com/SiweiMa/concrete_ml_lab/main/concrete_ml_lab_final_project.csv')
train, test = train_test_split(data, random_state=42)
is_test(train, test, RandomForestClassifier)
