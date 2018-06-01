import numpy as np
import math
from crossvalidation import Crossvalidation as clcv

class Knn():
    def __init__(self, X, y, k):
        self.X = X
        self.y = y
        self.k = 3
        self.label_list = list(set(y))
        self.__k_search_result__ = []

    def k_search(self, k_cv, k_list, log=False):
        for i in range(0,len(k_list)):
            k_fold = []
            for j in range(0,len(k_cv.data['y_train'])):
                X_tr = k_cv.data['X_train'][j]
                y_tr = k_cv.data['y_train'][j]
                X_te = k_cv.data['X_test'][j]
                y_te = k_cv.data['y_test'][j]
                self.train(X_tr, y_tr, k_list[i])
                accuracy, confmatrix = self.test(X_te, y_te)
                performance = dict(k=k_list[i], accuracy=accuracy, confmatrix=confmatrix, X_tr=X_tr, y_tr=y_tr, label_list=self.label_list)
                k_fold.append(performance)
            # find best k_fold
            best_accuracy_kfold_idx = 0
            for k in range(1, len(k_fold)):
                if k_fold[k]['accuracy'] > k_fold[best_accuracy_kfold_idx]['accuracy']:
                    best_accuracy_kfold_idx = k

            self.__k_search_result__.append(k_fold[best_accuracy_kfold_idx])
        # find the best K in k nn
        best_k_idx = 0
        for i in range(1, len(self.__k_search_result__)):
            if self.__k_search_result__[i]['accuracy'] > self.__k_search_result__[best_k_idx]['accuracy']:
                best_k_idx = i

        if log == True:            
            print '===============================__k_search_result__==============================='
            for i in self.__k_search_result__:
                print '{ k:', i['k'], ' accuracy:', i['accuracy'], ' confusionMatrix:', i['confmatrix'], ' label_list:', i['label_list'], ' }'
            i = self.__k_search_result__[best_k_idx]
            print '=============================__best_k_search_result__============================'
            print '{ k:', i['k'], ' accuracy:', i['accuracy'], ' confusionMatrix:', i['confmatrix'], ' label_list:', i['label_list'], ' }'
        # train with the best k
        i = self.__k_search_result__[best_k_idx]
        X_tr = i['X_tr']
        y_tr = i['y_tr']
        k = i['k']
        self.train(X_tr, y_tr, k)

    def evaluation(self, y_true, y_pred):
        totalLabel = len(self.label_list)
        label_list = self.label_list
        confmatrix = [[0 for x in range(totalLabel)] for y in range(totalLabel)]
        accuracy = 0
        for i in range(0, len(y_true)):
            confmatrix[label_list.index(y_true[i])][label_list.index(y_pred[i])] += 1
            if y_true[i] == y_pred[i]:
                accuracy += 1
        accuracy = float(accuracy)/float(len(y_true))
        return accuracy, confmatrix

    def voting(self, euclid_data):
        euclid_data = sorted(euclid_data, key=lambda x: x[0])
    
        y_label_counter = { key: 0 for key in self.label_list }

        for y in euclid_data[:self.k]:
            y_label_counter[y[1]] += 1
            euclid_data = euclid_data[:self.k]

        return y_label_counter, euclid_data

              
    def calculateEuclid(self, X):
        output = []
        for i in range(0, len(self.X)):
            z = np.subtract(np.asarray(self.X[i], dtype=float), np.asarray(X, dtype=float))
            z = np.abs(z)**2
            z = np.sqrt(np.sum(z))
            output.append([z, self.y[i]])
        return output

    def train(self, X_train, y_train, k):
        self.X = X_train
        self.y = y_train
        self.k = k

    def test(self, X_test, y_test):
        y_pred = []
        for i in X_test:
            euclid_data = self.calculateEuclid(i)
            vote_result, euclid_data = self.voting(euclid_data)
            y_pred.append(max(vote_result, key=vote_result.get))
        accuracy, confmatrix = self.evaluation(y_test, y_pred)

        return accuracy, confmatrix

    def predict(self, X, log=False):
        euclid_data = self.calculateEuclid(X)
        vote_result, euclid_data = self.voting(euclid_data)
        if log == True:
            print "=====Euclid data====="
            for i in euclid_data:
                print i
            print "=====Vote result====="
            print vote_result
        y_pred = max(vote_result, key=vote_result.get)
        return y_pred
