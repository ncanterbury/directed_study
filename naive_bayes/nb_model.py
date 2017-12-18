import pandas as pd
import sklearn.naive_bayes as nb
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from ggplot import *
from sklearn import metrics 
from sklearn.metrics import roc_auc_score
from IPython import embed
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from pprint import pprint 
from sklearn import datasets 
import numpy as np
from sklearn.cross_validation import train_test_split 

class NB_Model():
    def __init__(self, data_frame, feature_selection, grid_search):
        self.data_frame = data_frame 
        self.feature_selection = feature_selection
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

    def nb_model(self):
        X = self.X
        y = self.y
        # Build test and train sets 
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        clf = GaussianNB()
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)

        self.y_pred = y_pred
        
        
        '''  
        NB = svm.SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape=None, degree=3, gamma=0.1, kernel='rbf',
            max_iter=-1, probability=False, random_state=None, shrinking=True,
            tol=0.001, verbose=False)
        NB.fit(self.X_train, self.y_train)
        y_pred = NB.predict(self.X_test)
        self.y_pred = y_pred 
        '''
        ''' 
        random_state = np.random.RandomState(0) 
        cv = ShuffleSplit(n_splits=10, test_size=0.25, random_state=random_state)
        score = cross_val_score(NB, self.X_combined_std, self.y, cv=cv)
        scores2 = cross_val_score(NB, self.X_combined_std, self.y, cv=cv, scoring='f1_macro')

        print 'is score'
        print score
        print'is scores2'
        print scores2
        '''
        print('Misclassified samples: %d' % (self.y_test != y_pred).sum())
        print('Accuracy: %.2f' % accuracy_score(self.y_test, y_pred))


    def roc_area(self):
        roc_y_test = []
        roc_y_pred = []

        for label in self.y_test:
            if label == "pro_obama":
                roc_y_test.append(1)
            else:
                roc_y_test.append(0)

        for label in self.y_pred:
            if label == "pro_obama":
                roc_y_pred.append(1)
            else:
                roc_y_pred.append(0)
        
        roc = roc_auc_score(roc_y_test, roc_y_pred)
        print('ROC Area Under Curve') 
        print roc 
        fpr, tpr, thresholds = metrics.roc_curve(roc_y_test, roc_y_pred)

        df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
        plot = ggplot(df, aes(x='fpr', y='tpr')) +\
            geom_line() +\
            geom_abline(linetype='dashed')
        #print plot
        
def get_data_frame(csv_file):
    data_frame = pd.read_csv(csv_file)
    return data_frame

def main():
    csv_file = "~/directedstudyfall17/arff_generator/arff_files/naive_bayes.csv"
    data_frame = get_data_frame(csv_file)
    
    NB = NB_Model(data_frame, 'yes', 'no')
    NB.make_test_train()
    #NB.standardize_data()
    NB.nb_model()
    #embed()
    #NB.run_grid_search()
    NB.roc_area()
if __name__ == "__main__":
    main()

