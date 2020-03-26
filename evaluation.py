import pickle
import random
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.decomposition import PCA
from sklearn.utils import class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, matthews_corrcoef

random.seed(42)
np.random.seed(42)
random_state = 42

# Global Parameters
DATA_FOLDER = "data_100/"
LABEL_FILENAME = "internet_disruptions.tsv"
CAUSES = ["Natural Disaster",
          "DDoS",
          "Power Outage",
          "BGP Update",
          "Misconfiguration",
          "Cable Cut",
          "none"]
CAUSE_INDEX = 1     # which cause to evaluate
DATA_LIMIT = 75     # how many countries/cities/organizations to use (max 100)
TRIALS = 50         # trials
STEP = 3            # ensures no poisoning of results (DO NOT CHANGE)
PCA_DIM = 40        # projection dimension (should be <= DATA_LIMIT)


def process_cause(x):
    if CAUSES[CAUSE_INDEX].lower() in x.lower():
        return "positive_class"
    return "negative_class"


def preprocess_data(data_folder=DATA_FOLDER,
                    label_filename=LABEL_FILENAME,
                    data_limit=DATA_LIMIT):
    nides = pd.read_csv(label_filename, sep="\t")[["common name", "cause"]].dropna()
    for key in nides["common name"].values:
        nides = nides.append({"common name":f"before_{key}","cause":"none"}, ignore_index=True)
    nides["cause"] = nides.cause.apply(process_cause)
    keep_keys = [key for key,value in Counter(nides.cause.values).items()]
    keep_keys = list(nides[nides.cause.isin(keep_keys)]["common name"].values.ravel())
    nides = nides[nides["common name"].isin(keep_keys)]

    lbl = []
    for key in keep_keys:
        cause = nides[nides["common name"] == key].cause.values[0]
        lbl.append(cause)
    y = np.zeros((len(lbl)),dtype=int)
    for i, cause in enumerate(set(lbl)):
        y[np.where(np.array(lbl)==cause)[0]] = i

    X = []
    Y = []
    for i,key in enumerate(sorted(keep_keys)):
        try:
            tmp = pickle.load(open(f"{data_folder}{key}_100.attr", "rb"))
            [X.append(tmp[j].values.T[:,:data_limit]) for j in range(3)]
            [Y.append(y[i]) for j in range(3)]
        except Exception as e:
            continue

    data_full = np.array(X)
    print(data_full.shape)

    data_labels = np.vstack(Y).ravel()

    data_pos = data_full[data_labels == 1]
    data_neg = data_full[data_labels == 0]
    np.random.shuffle(data_pos)
    np.random.shuffle(data_neg)
    min_len = np.min([len(data_pos), len(data_neg)])
    X = np.vstack([data_pos[:min_len], data_neg[:min_len]])
    y = np.vstack([np.ones((min_len,1)),np.zeros((min_len,1))])
    return X,y

def display_results(acc_test,auc_test,mcc_test):
    print("mean +/- std [min, max]")
    print(f"{np.round(np.mean(acc_test),4)} +/- {np.round(np.std(acc_test),4)} [{np.round(np.min(acc_test),4)}-{np.round(np.max(acc_test),4)}]")
    print(f"{np.round(np.mean(auc_test),4)} +/- {np.round(np.std(auc_test),4)} [{np.round(np.min(auc_test),4)}-{np.round(np.max(auc_test),4)}]")
    print(f"{np.round(np.mean(mcc_test),4)} +/- {np.round(np.std(mcc_test),4)} [{np.round(np.min(mcc_test),4)}-{np.round(np.max(mcc_test),4)}]")

    plt.figure(figsize=(15,5))
    data = np.hstack([np.array(acc_test)[:,None],np.array(auc_test)[:,None],np.array(mcc_test)[:,None]])
    df = pd.DataFrame(data,columns=["Test Accuracy","Test AUC","Test Matthew Correlation Coefficient"])
    sb.boxplot(data=df, orient="h")
    plt.title("Detecting Denial-of-Service Attacks\nDistribution of Metrics on Test Set")
    plt.show()


# Start of Evaluation
X,y = preprocess_data()

auc_test = []
acc_test = []
mcc_test = []

for trial in range(TRIALS):
    index = list(range(0, len(X), STEP))
    indices = list(StratifiedShuffleSplit(n_splits=1, test_size=.15).split(index, y[::STEP]))
    train_idx = indices[0][0]
    test_idx = indices[0][1]
    train_idx = np.array([[index[i], index[i]+1, index[i]+2] for i in train_idx]).ravel()
    test_idx = np.array([[index[i], index[i]+1, index[i]+2] for i in test_idx]).ravel()

    x_train = X[train_idx]
    x_test = X[test_idx]
    y_train = y[train_idx].ravel()
    y_test = y[test_idx].ravel()

    for i in range(len(x_train)):
        x_train[i] = np.divide(x_train[i], np.expand_dims(np.max(x_train[i], axis=-1)+1E-3, axis=-1))
    for i in range(len(x_test)):
        x_test[i] = np.divide(x_test[i], np.expand_dims(np.max(x_test[i], axis=-1)+1E-3, axis=-1))
    x_train = x_train - 0.5
    x_test = x_test - 0.5

    pca = PCA(n_components=PCA_DIM)
    pca.fit(np.squeeze(np.mean(x_train, axis=-1)))
    pca_x_train = pca.transform(np.squeeze(np.mean(x_train, axis=-1)))
    pca_x_test = pca.transform(np.squeeze(np.mean(x_test, axis=-1)))

    print(f"Baseline 1:  {accuracy_score(y_test, np.zeros(len(y_test)))}\n")

    # model = LogisticRegression(solver="lbfgs") # you can switch this in if you want
    model = RandomForestClassifier(n_estimators=1000, max_depth=35)
    model.fit(pca_x_train, y_train)

    print(f"Test Accuracy: {accuracy_score(y_test, model.predict(pca_x_test))}")
    print(f"Test AUC:      {roc_auc_score(y_test, model.predict_proba(pca_x_test)[:,1])}")
    print(f"Test Log-Loss: {log_loss(y_test, model.predict_proba(pca_x_test)[:,1])}")
    print(f"Test MCC:      {matthews_corrcoef(y_test, model.predict(pca_x_test))}\n")

    acc_test.append(accuracy_score(y_test, model.predict(pca_x_test)))
    auc_test.append(roc_auc_score(y_test, model.predict_proba(pca_x_test)[:,1]))
    mcc_test.append(matthews_corrcoef(y_test, model.predict(pca_x_test)))
