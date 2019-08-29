import pandas as pd
import numpy as np
import os
from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

DATASET_PATH = "C:/Users/agilist/Desktop/Kaggle/SpamAccountTwitter/Dataset"

def ExtraTree(X_train, X_test, y_train, y_test):
    ex = ExtraTreesClassifier(n_estimators=100, random_state=0)
    ex.fit(X_train,y_train)
    predictions = ex.predict(X_test)
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))
    ext_accuracy = accuracy_score(y_test, predictions)
    #print("Neural Network :: ", nn_accuracy*100)
    return ext_accuracy*100

def DecisionTree(X_train, X_test, y_train, y_test):
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train,y_train)
    predictions = dt.predict(X_test)
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))
    dt_accuracy = accuracy_score(y_test, predictions)
    #print("Neural Network :: ", nn_accuracy*100)
    return dt_accuracy*100

def NeuralNetwork(X_train, X_test, y_train, y_test):
    mlp = MLPClassifier(hidden_layer_sizes=(7,7,7,7), random_state=0, max_iter=500, solver="sgd")
    mlp.fit(X_train,y_train)
    predictions = mlp.predict(X_test)
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))
    nn_accuracy = accuracy_score(y_test, predictions)
    #print("Neural Network :: ", nn_accuracy*100)
    return nn_accuracy*100

def RandomForest(X_train, X_test, y_train, y_test, colNames):
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred)
    confusionMatrix = confusion_matrix(y_test, y_pred)
    classificationReport = classification_report(y_test,y_pred)
    print(classificationReport)

    return rf_accuracy*100


def NaiveBayes(X_train, X_test, y_train, y_test):
    lda = LDA(n_components=1)
    X_train = lda.fit_transform(X_train, y_train)
    X_test = lda.transform(X_test)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    gnb_accuracy = accuracy_score(y_test, y_pred)
    confusionMatrix = confusion_matrix(y_test, y_pred)
    classificationReport = classification_report(y_test,y_pred)
    print(classificationReport)
    #print(gnb_accuracy*100)
    return gnb_accuracy*100


def LogisticRegressionModel(X_train, X_test, y_train, y_test):
    lda = LDA(n_components=1)
    X_train = lda.fit_transform(X_train, y_train)
    X_test = lda.transform(X_test)

    ## feature scaling is used because of regularization (default in logistic regression)
    logmodel = LogisticRegression(solver="newton-cg")
    logmodel.fit(X_train, y_train)
    predictions = logmodel.predict(X_test)
    confusionMatrix = confusion_matrix(y_test, predictions)
    classificationReport = classification_report(y_test,predictions)
    print(classificationReport)
    #print(confusionMatrix)
    TP = confusionMatrix[0][0]
    TN = confusionMatrix[1][1]
    FP = confusionMatrix[0][1]
    FN = confusionMatrix[1][0]
    #print(TP, TN, FP, FN)
    accuracy = (TP+TN)/(TP+TN+FN+FP)
    #print(accuracy*100)
    return accuracy*100


def plot_bar_x(X):
    # this is for plotting purpose
    value, count = np.unique(X, return_counts=True)
    print(value.tolist())
    print(count.tolist())
    plt.bar(value.tolist(), count.tolist())
    plt.xlabel('Class', fontsize=5)
    plt.ylabel('Occurence', fontsize=5)
    #plt.show()

def splitdataset(dataframe):
    #X = dataframe.iloc[:, :-1].values
    X= dataframe.iloc[:, :-1].values
    standardScalerX = preprocessing.StandardScaler()
    x = standardScalerX.fit_transform(X)
    y = dataframe.iloc[:, -1].values
    #print(X, y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    return x, y, X_train, X_test, y_train, y_test

def main():
    ## Reading dataset
    for file in os.listdir(DATASET_PATH):
        mydf = pd.read_csv(os.path.join(DATASET_PATH, file))
        sns.lmplot(x = "id", y="no_retweets", data=mydf, fit_reg=False, hue='class', legend=False) 
        plt.legend(loc='upper right')
        plt.show()
        
        #https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba --- IMPORTANT
        
        z = np.abs(stats.zscore(mydf.iloc[:, :-1]))
        threshold = 3
        print(np.where(z > threshold))
        mydf = mydf[(z < 3).all(axis=1)]
        
        sns.lmplot(x = "id", y="no_retweets", data=mydf, fit_reg=False, hue='class', legend=False) 
        plt.legend(loc='upper right')
        plt.show()

        print(mydf.head())
        print(mydf.describe())
        print(mydf.info())
        mydf.fillna(0, inplace=True)
        # Split Dataset
        mydf = mydf.drop(columns=['id'])
        X, y, X_train, X_test, y_train, y_test = splitdataset(mydf)
        columnNames = mydf.columns[:-1]
        print(columnNames)

        # plot y_train and y_test to check distribution of data
        plot_bar_x(y)
        plot_bar_x(y_train)
        plot_bar_x(y_test)

        log_accuracy = LogisticRegressionModel(X_train, X_test, y_train, y_test)
        naive_accuracy = NaiveBayes(X_train, X_test, y_train, y_test)
        rf_accuracy = RandomForest(X_train, X_test, y_train, y_test, columnNames)
        nn_accuracy = NeuralNetwork(X_train, X_test, y_train, y_test)
        dt_accuracy = DecisionTree(X_train, X_test, y_train, y_test)
        ext_accuracy = ExtraTree(X_train, X_test, y_train, y_test)
        
        print("Logistic Regression: ", log_accuracy)
        print("Naive Bayes: ", naive_accuracy)
        print("Random Forest: ", rf_accuracy)
        print("Neural Network: ", nn_accuracy)
        print("Decision Trees: ", dt_accuracy)
        print("Extra Trees Classifier: ", ext_accuracy)





if __name__ == '__main__':
    main()