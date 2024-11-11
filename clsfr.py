#clsfr

import pandas as pd
from os import listdir, makedirs
from sys import argv
from sklearn.model_selection import train_test_split as tts
from xgboost import XGBClassifier as XGB
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc
from argparse import ArgumentParser
from pickle import dump, load
import matplotlib.pyplot as plt
from random import randint
import seaborn as sns

parser = ArgumentParser(prog="clsfr",
                        usage="clsfr.py [options] file",
                        description="SAMPLE TEXT",
                        epilog="Play Sekiro: Shadows Die Twice instead of using this thing")

group = parser.add_mutually_exclusive_group(required = True)
group.add_argument('-t', '--train', action="store_true", help = "Train by csv file and save the model")
group.add_argument('-p', '--predict', action="store_true", help = "Predict genres in csv with existing model")

parser.add_argument('-m', '--model', default="./model", help = "Model input (if predicting) or model output (if training) path")
parser.add_argument('csv_file', help = "csv input path")

args = parser.parse_args(argv[1:])

model = args.model
model = model.replace('\\', '/').removesuffix('/')
csv_file = args.csv_file

data = pd.read_csv(csv_file, index_col=0)
feature_end = None
columns = data.iloc[:, data.columns.get_loc('MFCC0_STD'):].columns
for i in range(len(columns)):
    if 'MFCC' not in columns[i]:
        feature_end = data.columns.get_loc('MFCC0_STD') + i
        break
X = data.iloc[:,:feature_end]
models = []

if args.train:
    Xmin = X.min()
    Xmax = X.max()
    X = (X - Xmin)/(Xmax - Xmin)
    Ys = data.iloc[:, feature_end:]
    x_train, x_test, ys_train, ys_test = tts(X, Ys, test_size = 0.33, random_state = 42)
    n_genres = data.columns[feature_end:].size
    genres = data.columns[feature_end:]
    color_pallete = sns.color_palette('hls', n_genres)
    for i in range(n_genres):
        models.append(XGB(n_estimators=1000))
        models[i].fit(x_train, ys_train.iloc[:,i])
        preds = models[i].predict(x_test)
        print(genres[i] + ' accuracy:', round(accuracy_score(ys_test.iloc[:,i], preds), 5), '\n')
        print(genres[i] + ' confusion matrix:\n', confusion_matrix(ys_test.iloc[:,i], preds), '\n')
        preds = models[i].predict_proba(x_test)[:,1]
        fpr, tpr, threshold = roc_curve(ys_test.iloc[:,i], preds)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color_pallete[i], label=(genres[i] + ' ROC (area = {:.2%})'.format(roc_auc)))
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.00])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()
    for i in range(n_genres):
        try:
            makedirs(model)
        except FileExistsError:
            pass
    dump([models[i],Xmin,Xmax], open(model + '/' + genres[i], 'wb'))
else:
    genres = listdir(model)
    for genre in genres:
        load_model, Xmin, Xmax = load(open(model + '/' + genre, 'rb'))
        models.append(load_model)
    X = (X - Xmin)/(Xmax - Xmin)
    out = pd.DataFrame()
    for i in range(len(genres)):
        pred = models[i].predict_proba(X)[:,1]
        s = pd.DataFrame(pred, columns = [genres[i]], index = data.index)
        out = pd.concat((out, s), axis = 1)
    argmaxs = out.to_numpy().argmax(axis = 1)
    argmaxcols = []
    for i in range(len(argmaxs)):
        argmaxcols.append(out.columns[argmaxs[i]])
    out = pd.concat((out, pd.DataFrame(argmaxcols, index = out.index, columns=['Max_Column'])), axis = 1)
    out.iloc[:,:-1] = out.iloc[:,:-1].map(lambda x: '{:.2%}'.format(x))
    print(out)
    print(pd.DataFrame([[argmaxcols[i], out[argmaxcols[i]][i]] for i in range(len(argmaxcols))], index = out.index, columns=['Max_Column', 'Value']))