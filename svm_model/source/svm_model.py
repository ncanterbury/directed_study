import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from ggplot import *
from sklearn import metrics 
from sklearn.metrics import roc_auc_score
from IPython import embed
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from pprint import pprint 
from sklearn import datasets 
import numpy as np
from sklearn.cross_validation import train_test_split 

class SVM_Model():
    def __init__(self, data_frame, grid_search):
        self.data_frame = data_frame 
        self.grid_search = grid_search
        self.X = None 
        self.y = None 
        self.X_train = None 
        self.X_test = None 
        self.y_train = None 
        self.y_test = None 
        self.X_train_std = None 
        self.X_test_std = None 
        self.X_combined_std = None 
        self.y_pred = None 


    def make_test_train(self):
        ''' Build the first X and Y arrays. X will hold 
            the attribute data, Y will hold class labels.
            Will read this data out of self.dataframe. 
            Assign self.X and self.y to resulting arrays
        '''
        # create 2d array of length of rows 
        rows = self.data_frame.shape[0]
        cols = self.data_frame.shape[1]
         
        # X is a 2d array that will hold all attribute 
        # values excluding the sentiment class values 

        # Y is a one dimensional array that holds 
        # sentiment class values 
        X = [[] for x in range(rows)]
        y = []

        # build list of column names to iterate over
        columns_list_all_values = list(self.data_frame.columns.values)
        columns_list = []
        for value in columns_list_all_values:
            if value != 'sentiment':
                columns_list.append(value)

        # Build X 
        for column in columns_list:
            row_count = 0
            for value in self.data_frame[str(column)].values:
                X[row_count].append(value)
                row_count += 1
        
        # Build Y
        for value in self.data_frame['sentiment'].values:
            y.append(value)
        
        self.X = X
        self.y = y
    
    def standardize_data(self):
        sc = StandardScaler()
        sc.fit(self.X_train)
        self.X_train_std = sc.transform(self.X_train)
        self.X_test_std = sc.transform(self.X_test)
        self.X_combined_std = np.vstack((self.X_train_std, self.X_test_std))

    def svm_model(self):
        X = self.X
        y = self.y


        # Build test and train sets 
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        #SVM = svm.SVC(C=1.0, gamma=1.0, kernel='poly')
        SVM = svm.SVC(C=2.82843, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma=0.353553390593,
  kernel='linear', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)
        #SVM.fit(self.X_train, self.y_train)
        #y_pred = SVM.predict(self.X_test)
        #self.y_pred = y_pred 
        
        scores = cross_val_score(SVM, self.X, self.y, cv=10)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    def roc_area(self):
        X = self.X
        y = self.y

        y = label_binarize(y, classes=['pro_obama', 'pro_romney', 'neutral'])
        n_classes = 3

        # Build test and train sets 
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        clf = OneVsRestClassifier(svm.SVC(C=2.82843, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma=0.353553390593,
  kernel='linear', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False))
        y_score = clf.fit(self.X_train, self.y_train).decision_function(self.X_test)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(self.y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot of a ROC curve for a specific class
        for i in range(n_classes):
            plt.figure()
            plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            #plt.show()


    def run_grid_search(self):
        ''' run grid search on SVM with specified parameters '''

        param_grid = [
            {'C': [0.03125, 0.04419, 0.0625, 0.08839, 0.125, 0.17677, 0.25, 0.35355, 0.5, 0.7071, 1, 1.41421, 2, 2.82843, 4, 5.6568, 8, 11.3137, 16, 22.6274, 32, 10, 100], 
                'gamma': [0.000030517578125, 0.0000431583, 0.00006103515625, 0.0000863167457503, 0.0001220703125, 0.0001726334915006, 0.000244140625, 0.0003452669830012, 0.00048828125, 0.0006905339660025, 0.0009765625, 0.001381067932005, 0.001953125, 0.00276213586401, 0.00390625, 0.0055242717280199, 0.0078125, 0.0110485434560398, 0.015625, 0.0220970869120796, 0.03125, 0.0441941738241592, 0.0625, 0.0883883476483184, 0.125, 0.1767766952966369, 0.25, 0.3535533905932738, 0.5, 0.7071067811865475, 1, 1.414213562373095, 2, 2.8284271247461901, 4, 5.6568542494923802, 8, 0.1, 0.01, 0.001, 0.0001], 
                'kernel': ['poly','linear', 'rbf']},
        ]

        svc = svm.SVC()
        clf = GridSearchCV(svc, param_grid, verbose=100)
        clf.fit(self.X, self.y)
        print sorted(clf.cv_results_.keys())
        embed()

    def univariate_feature_selection(self):
        # X is self.X, y is self.y
        X = self.X
        y = self.y
        X_new = SelectKBest(chi2, k=4).fit_transform(X, y)
        '''
        print 'original features'
        print self.X
        print 'new features'
        print X_new 
        '''

        self.X = X_new

    def recursive_feature_selection(self):
        X = self.X
        y = self.y
        y_as_ints = []
        for label in y:
            if label == "pro_obama":
                y_as_ints.append(1)
            else:
                y_as_ints.append(0)

        estimator = svm.SVC(kernel="linear")

        selector = RFE(estimator, 5, step=1)
        selector = selector.fit(X, y_as_ints)
        selected_features = selector.ranking_
        '''
        print 'this is selector rankings'
        print selected_features
        '''

        
        # Rebuild X array
        # create 2d array of length of rows 
        rows = self.data_frame.shape[0]
        cols = self.data_frame.shape[1]
         
        # X is a 2d array that will hold all attribute 
        # values excluding the sentiment class values 

        X_new = [[] for x in range(rows)]

        # build list of column names to iterate over
        columns_list_all_values = list(self.data_frame.columns.values)
        columns_list = []
        for value in columns_list_all_values:
            if value != 'sentiment':
                columns_list.append(value)

        feature_list_columns = []
        # build list of column names using only columns from feature selection
        for index in range(len(selected_features)):
            if selected_features[index] == 1:
                feature_list_columns.append(columns_list[index])
        
        # Build X 
        for column in feature_list_columns:
            row_count = 0
            for value in self.data_frame[str(column)].values:
                X_new[row_count].append(value)
                row_count += 1
        
        self.X = X_new 

    def select_from_model(self):
        X = self.X
        y = self.y

        lsvc = LinearSVC(C=0.01, penalty='l1', dual=False).fit(X, y)
        model = SelectFromModel(lsvc, prefit=True)
        X_new = model.transform(X)
        self.X = X_new
        
def get_data_frame(csv_file):
    data_frame = pd.read_csv(csv_file)
    return data_frame

def main():
    csv_file = "~/directedstudyfall17/arff_generator/arff_files/1128.csv"
    data_frame = get_data_frame(csv_file)
    
    SVM = SVM_Model(data_frame, "False")
    SVM.make_test_train()
    #SVM.standardize_data()
    SVM.svm_model()
    #SVM.roc_area()
    embed()
if __name__ == "__main__":
    main()
