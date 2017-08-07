# algorithms : list of machine learning classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

def fit_accuracy(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def classification(X_train, y_train, X_test, y_test, scores):
    scores = scores
    clf = LogisticRegression()
    score = fit_accuracy(clf, X_train,y_train, X_test, y_test)
    scores['LogisticRegression'].append(score)
    clf = SVC()
    score = fit_accuracy(clf, X_train,y_train, X_test, y_test)
    scores['SVC'].append(score)
    clf = SGDClassifier()
    score = fit_accuracy(clf, X_train,y_train, X_test, y_test)
    scores['SGDClassifier'].append(score)
    clf = GaussianProcessClassifier()
    score = fit_accuracy(clf, X_train,y_train, X_test, y_test)
    scores['GaussianProcessClassifier'].append(score)
    clf = GaussianNB()
    score = fit_accuracy(clf, X_train,y_train, X_test, y_test)
    scores['GaussianNB'].append(score)
    clf = RandomForestClassifier()
    score = fit_accuracy(clf, X_train,y_train, X_test, y_test)
    scores['RandomForestClassifier'].append(score)
    clf = ExtraTreesClassifier()
    score = fit_accuracy(clf, X_train,y_train, X_test, y_test)
    scores['ExtraTreesClassifier'].append(score)
    clf = AdaBoostClassifier()
    score = fit_accuracy(clf, X_train,y_train, X_test, y_test)
    scores['AdaBoostClassifier'].append(score)
    clf = GradientBoostingClassifier()
    score = fit_accuracy(clf, X_train,y_train, X_test, y_test)
    scores['GradientBoostingClassifier'].append(score)
    clf = DecisionTreeClassifier()
    score = fit_accuracy(clf, X_train,y_train, X_test, y_test)
    scores['DecisionTreeClassifier'].append(score)
    
    return scores