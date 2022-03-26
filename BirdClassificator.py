# Classifying audio files of bird sounds to their spiecies using support-vector
# machine and random forest classifiers

import numpy as np
import matplotlib.pyplot as plt
import optuna
import os
# import opensmile
# from pydub import AudioSegment
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, make_scorer, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.model_selection import cross_validate, StratifiedKFold

# ---------------- Preparing and verifying data - preprocessing ---------------
# # 1) Preparing the files list
folder_path = 'dane'
files_list = []
for directory, subdirs, files in os.walk(folder_path):
    files_list.extend(files)

#  -------------------------------- WARNING --------------------------------------
# Following commented lines of code are meant to use to calculate the parameters
# from the source files. If you want to modify the code and use some other data,
# you can use following code.

# # 2) Calculating parameters using opensmile library
#
# # Changing sampling frequency to 44100 Hz
# for i in files_list:
#     sound = AudioSegment.from_mp3('dane/' + i)  # waves
#     sound = sound.set_frame_rate(44100)
#
# parametry = []
# smile = opensmile.Smile(
#     feature_set = opensmile.FeatureSet.eGeMAPSv02,
#     feature_level = opensmile.FeatureLevel.Functionals)
#
# for i in files_list:
#     feats = smile.process_file('dane/' + i)
#     parametry.append(feats)
#
# parametry_array = np.asarray(parametry)
# np.save('ptaki_dane.npy', parametry_array)

# ---------------------------------------------------------------------------------

# 3) Loading the data from file
dane = np.load('ptaki_dane.npy')
dane = dane.reshape(215, -1)
print(dane.shape)

# 4) Preparing labels
y = np.zeros(len(dane))

for count, i in enumerate(files_list):
    if i.startswith('Alauda-arvensis'):
        y[count] = 1
    elif i.startswith('Bubo-bubo'):
        y[count] = 2
    elif i.startswith('Crex-crex'):
        y[count] = 3
    elif i.startswith('Glaucidium-passerinum'):
        y[count] = 4
    else:
        y[count] = 5

# Checking how may objects are there in each class (how many bird recording samples
# are there in each species)
counter_y = Counter(y)
plt.bar(counter_y.keys(), counter_y.values())
Counter(y)

# 5) Splitting data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(dane, y, test_size=0.2, random_state=42, stratify=y)

print(X_train.shape)
print(X_test.shape)

# 6) Data standardization
X = StandardScaler().fit_transform(dane)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# ---------- Classification -------------

# Model 1.1 - SVC with F1 as a scoring parameter

# 1) Preparing the functions
scoring = {'f1_macro': make_scorer(f1_score, average='macro')}
model = SVC


def objective(trial, model, get_space, X, y):
    model_space = get_space(trial)
    mdl = model(**model_space)
    scores = cross_validate(mdl, X, y, scoring=scoring, cv=StratifiedKFold(n_splits=5), return_train_score=True)
    return np.mean(scores['test_f1_macro'])


def get_space(trial):
    space = {"C": trial.suggest_uniform("C", 0, 10),
             'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
             "degree": trial.suggest_int("degree", 2, 8)}
    return space


trials = 1000

# 2) Optimalization
study = optuna.create_study(direction='maximize')
study.optimize(lambda x: objective(x, model, get_space, X_train, y_train), n_trials=trials)

# 3) Displaying optimal parameters
print('params: ', study.best_params)

# 4) Evaluation (accuracy, F1, recall, precision, confusion matrix)
lr = model(**study.best_params)
lr.fit(X_train, y_train)
preds = lr.predict(X_test)

print('Accuracy = ', accuracy_score(y_test, preds))
print('F1 = ', f1_score(y_test, preds, average='weighted'))
print('Recall = ', recall_score(y_test, preds, average='weighted'))
print('Precision = ', precision_score(y_test, preds, average='weighted'))
print('Confusion matrix: \n', confusion_matrix(y_test, preds))

# Model 1.2 - SVC with accuracy as a scoring parameter

# 1) Preparing the functions
scoring = {'accuracy': make_scorer(f1_score, average='macro')}
model = SVC


def objective(trial, model, get_space, X, y):
    model_space = get_space(trial)
    mdl = model(**model_space)
    scores = cross_validate(mdl, X, y, scoring=scoring, cv=StratifiedKFold(n_splits=5), return_train_score=True)
    return np.mean(scores['test_accuracy'])


def get_space(trial):
    space = {"C": trial.suggest_uniform("C", 0, 10),
             'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
             "degree": trial.suggest_int("degree", 2, 8)}
    return space


trials = 1000

# 2) Optimalization
study = optuna.create_study(direction='maximize')
study.optimize(lambda x: objective(x, model, get_space, X_train, y_train), n_trials=trials)

# 3) Displaying optimal parameters
print('params: ', study.best_params)

# 4) Evaluation (accuracy, F1, recall, precision, confusion matrix)
lr = model(**study.best_params)
lr.fit(X_train, y_train)
preds = lr.predict(X_test)

print('Accuracy = ', accuracy_score(y_test, preds))
print('F1 = ', f1_score(y_test, preds, average='weighted'))
print('Recall = ', recall_score(y_test, preds, average='weighted'))
print('Precision = ', precision_score(y_test, preds, average='weighted'))
print('Confusion matrix: \n', confusion_matrix(y_test, preds))

# Model 2 - Random Forest Classifier

# 1) Preparing the functions
scoring = {'f1_macro': make_scorer(f1_score, average='macro')}
model = RandomForestClassifier


def get_space(trial):
    space = {"n_estimators": trial.suggest_int("n_estimators", 30, 150),
             "max_depth": trial.suggest_int("max_depth", 5, 20),
             "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
             "n_jobs": trial.suggest_int("n_jobs", -1, -1)}
    return space


def objective(trial, model, X, y):
    model_space = get_space(trial)
    mdl = model(**model_space)
    scores = cross_validate(mdl, X, y, scoring=scoring, cv=StratifiedKFold(n_splits=5), return_train_score=True)
    return np.mean(scores['test_f1_macro'])


trials = 500

# 2) Optimization
study = optuna.create_study(direction='maximize')
study.optimize(lambda x: objective(x, model, X_train, y_train), n_trials=trials)

# 3) Displaying optimal parameters
print('params: ', study.best_params)

# 4) Evaluation (accuracy, F1, recall, precision, confusion matrix)
lr = model(**study.best_params)
lr.fit(X_train, y_train)
preds = lr.predict(X_test)

print('Accuracy = ', accuracy_score(y_test, preds))
print('F1 = ', f1_score(y_test, preds, average='weighted'))
print('Recall = ', recall_score(y_test, preds, average='weighted'))
print('Precision = ', precision_score(y_test, preds, average='weighted'))
print('Confusion matrix: \n', confusion_matrix(y_test, preds))