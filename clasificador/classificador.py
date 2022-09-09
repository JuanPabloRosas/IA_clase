import pandas as pd
import numpy as np
import os
import sklearn.pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def root_directory():
   current_path = os.path.abspath(__file__)
   return os.path.abspath(os.path.join(current_path, os.pardir))
def data_directory():
   return os.path.join(root_directory(),"hrv dataset", "data")

def load_train_set():
   in_file = os.path.join(data_directory(),"final", "train.csv")
   return pd.read_csv(in_file)
def load_test_set():
   in_file = os.path.join(data_directory(),"final", "test.csv")
   return pd.read_csv(in_file)

def simple_model_evaluation():
   select = SelectKBest(k='all')
   train =load_train_set()
   test = load_test_set()
   target = 'condition'
   hrv_features = list(train)
   hrv_features = [x for x in hrv_features if x not in [target]]
   X_train= train[hrv_features]
   y_train= train[target]
   X_test = test[hrv_features]
   y_test = test[target]
   classifiers = [
                  #RandomForestClassifier(n_estimators=500, max_features='log2', n_jobs=-1),
                  #SVC(C=20, kernel='rbf'),   
                  KNeighborsClassifier(3),
                  SVC(kernel="linear", C=0.025),
                  SVC(gamma=2, C=1),
                  GaussianProcessClassifier(1.0 * RBF(1.0)),
                  DecisionTreeClassifier(max_depth=5),
                  RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                  MLPClassifier(alpha=1, max_iter=1000),
                  AdaBoostClassifier(),
                  GaussianNB(),
                  QuadraticDiscriminantAnalysis(),
               ]
   for clf in classifiers:
      """
      name = str(clf).split('(')[0]
      if 'svc' == name.lower():
            # Normalize the attribute values to mean=0 and variance=1
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
      clf = RandomForestClassifier()
      steps = [('feature_selection', select),
            ('model', clf)]
      pipeline = sklearn.pipeline.Pipeline(steps)
      pipeline.fit(X_train, y_train)
      y_prediction = pipeline.predict(X_test)
      pipeline.score(X_test,y_test)
      print("----------------------------{0}---------------------------".format(name))
      print(sklearn.metrics.classification_report(y_test, y_prediction))
      print()
      print()
      """
      #--------------------------------------------------
      #ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
      clf.fit(X_train, y_train)
      y_prediction = clf.predict(X_test)
      score = clf.score(X_test, y_prediction)
      print(sklearn.metrics.classification_report(y_test, y_prediction))
      
      #DecisionBoundaryDisplay.from_estimator( clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5  )

      # Plot the training points
      #ax.scatter( X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k" )
      # Plot the testing points
      #ax.scatter( X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors="k",  alpha=0.6,    )

      #ax.set_xlim(x_min, x_max)
      #ax.set_ylim(y_min, y_max)
      #ax.set_xticks(())
      #ax.set_yticks(())
      #if ds_cnt == 0:
      #    ax.set_title(name)
      #ax.text( x_max - 0.3, y_min + 0.3, ("%.2f" % score).lstrip("0"), size=15, horizontalalignment="right", )
      #i += 1
    
     
if __name__ == '__main__':
    simple_model_evaluation()
    
   


   