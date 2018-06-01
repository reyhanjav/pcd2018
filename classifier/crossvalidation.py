import numpy as np 
from math import floor

class Crossvalidation:
    def __init__(self, X, y):
        self.X = X 
        self.y = y
        self.data = dict(X_train=[], y_train=[], X_test=[], y_test=[])
        self.label_list = list(set(y))
        self._perf_ = dict()
    
    def split(self, k, stratified=False):
        if stratified == True:
            split_dict = dict()
            for c in self.label_list:
                split_dict[c] = []
            for i in range(0, len(self.y)):
                split_dict[self.y[i]].append(self.X[i])

            for i in range(0,k):
                temp_y_te = []
                temp_y_tr = []
                temp_X_te = []
                temp_X_tr = []
                for c in self.label_list:
                    te_index = int(floor(float(len(split_dict[c])/k)))
                    tr_index = len(split_dict[c]) - te_index

                    temp_X_te += split_dict[c][i*te_index:(i+1)*te_index]

                    if i == 0:
                        temp_X_tr += split_dict[c][te_index:]
                    elif i > 0 and i < (k-1):
                        temp_X_tr += split_dict[c][:i*te_index]+split_dict[c][(i+1)*te_index:]
                    elif i == (k-1):
                        temp_X_tr += split_dict[c][:i*te_index]

                    temp_y_te += [c for z in range(0, te_index)]
                    temp_y_tr += [c for z in range(0, tr_index)]

                self.data['X_test'].append(temp_X_te)
                self.data['y_test'].append(temp_y_te)
                self.data['X_train'].append(temp_X_tr)
                self.data['y_train'].append(temp_y_tr)
        else:
            te_index = int(floor(float(len(self.y)/k)))
            # train is ceil and test is ground
            for i in range(0,k):
                self.data['X_test'].append(self.X[i*te_index:(i+1)*te_index])
                self.data['y_test'].append(self.y[i*te_index:(i+1)*te_index])
                if i == 0:
                    self.data['X_train'].append(X[te_index:])
                    self.data['y_train'].append(y[te_index:])
                elif i > 0 and i < (k-1):
                    self.data['X_train'].append(self.X[:i*te_index]+X[(i+1)*te_index:])
                    self.data['y_train'].append(self.y[:i*te_index]+y[(i+1)*te_index:])
                elif i == (k-1):
                    self.data['X_train'].append(self.X[:i*te_index])
                    self.data['y_train'].append(self.y[:i*te_index])