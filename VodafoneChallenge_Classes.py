'''
This .py file contains the interfaces we write to make our code the more general we could, so that it could
be applied also to other datasets or in general other problems.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import tree

import graphviz

pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=np.NaN)


'''
Class buildTrain is a class we built to create autonomously the training, validation and test set.
We started from a very simple class (very similar to the functions we built in class) and then we implemented
also the interface to add pca, standardization and in the end also one-hot encoding.
'''

class buildTrain():
    def __init__(self, X, y, perc=(0.7,0.15,0.15), std=False, pca=0, seed=None, one_hot=False, cat_col=None):
        if seed is not None:
            np.random.seed(seed)
        n_data, n_features = X.shape
        if not isinstance(perc, tuple) or np.abs(1-sum(perc))>1e-7:
            raise Exception('Invalid value for perc', perc)
        if not isinstance(X, pd.DataFrame):
            raise Exception('must pass a pandas dataframe')
        assert n_data == len(y)
        
        dopca = pca is None or pca > 0
        origin_shape = X.shape
        
        #check and remove nan values
        X = X.copy()
        X[y.name] = y.copy()
        X.dropna(axis=0, how='any', inplace=True)
        
        if X.shape[0] < n_data:
            print('Warning: missing data found and removed. Old input shape: %d, %d, new input shape: %d, %d'
                  % (origin_shape[0], origin_shape[1]+1, *X.shape))
            y = X[y.name]
        X.drop(y.name, axis=1, inplace=True)
        n_data, _ = X.shape
    
        assert n_data == len(y)
        assert X.shape[1] == origin_shape[1]
        
        #do not perform OneHot if want to do PCA or std
        if dopca:
            one_hot = False
            
        if one_hot:
            if cat_col is None:
                cat_col = X.columns
            non_cat_col = [i for i in X.columns if i not in cat_col]
            for cat in cat_col:
                ret, columns = self._one_hot(X[cat])
                X.drop(cat, axis=1, inplace=True)
                for i in range(len(columns)):
                    X[columns[i]] = ret[:,i]
            self.X_one_hot = X.copy()
        
        perm = np.random.random(n_data)
        n_train = int(perc[0]*n_data)
        n_valid = int(perc[1]*n_data)
        train_mask = perm < perc[0]
        valid_mask = ~ train_mask.copy()
        valid_mask[~train_mask] = perm[~train_mask] < perc[0] + perc[1]
        test_mask = ~ np.logical_or(train_mask, valid_mask)
        
        train_data = X[train_mask]
        train_target = y[train_mask]
        valid_data = X[valid_mask]
        valid_target = y[valid_mask]
        test_data = X[test_mask]
        test_target = y[test_mask] 
        assert (len(train_data)+len(valid_data)+len(test_data)) == n_data
        
        if std:
            if not one_hot:
                mean = train_data.mean(axis=0)
                std = train_data.std(axis=0) + 1e-10
                train_data = (train_data - mean) / std
                valid_data = (valid_data - mean) / std
                test_data = (test_data - mean) / std
                print('Performed standardization')
            else:
                mean = train_data[non_cat_col].mean(axis=0)
                std = train_data[non_cat_col].std(axis=0) + 1e-10
                train_data[non_cat_col] = (train_data[non_cat_col] - mean) / std
                valid_data[non_cat_col] = (valid_data[non_cat_col] - mean) / std
                test_data[non_cat_col] = (test_data[non_cat_col] - mean) / std
                print('Performed standardization only on non-categorical columns')
        
        if dopca:
            my_pca = PCA(n_components=pca)
            my_pca.fit(train_data)
            train_data = my_pca.transform(train_data)
            valid_data = my_pca.transform(valid_data)
            test_data = my_pca.transform(test_data)
            print('performed PCA, number of features: %d, explained variance for component:\n'%(my_pca.n_components_), 
                  ['%.2f'%i for i in my_pca.explained_variance_])
        
        self.Xt = train_data
        self.yt = train_target
        self.Xv = valid_data
        self.yv = valid_target
        self.Xts = test_data
        self.yts = test_target
        
    def _one_hot(self, v):
        cats = np.unique(v)
        map_cat = {cat:i for i, cat in enumerate(cats)}
        map_cat_r = {map_cat[key]:key for key in map_cat}
        n_cat = len(cats)
        n_rows = len(v)
        ret = np.zeros((n_rows,n_cat))
        for i in range(n_rows):
            ret[i, map_cat[v[i]]] = 1
        columns = ['%s_%s' % (v.name, map_cat_r[i]) for i in range(n_cat)]
        return ret, columns
                              
        
    def get_train(self):
        return self.Xt, self.yt
    
    def get_valid(self):
        return self.Xv, self.yv
    
    def get_test(self):
        return self.Xts, self.yts
    
    def get_size(self):
        return self.Xt.shape, self.Xv.shape, self.Xts.shape
    
    def get_one_hot(self):
        return self.X_one_hot

'''
Class logger is related to verbosity of some objects (print or plot, for example)
'''

class logger():
    def __init__(self, verbose = True):
        self.v = verbose
        self.log_ = []
        
    def log_it(self, text):
        #adds to log record
        if not isinstance(text, str):
            raise Exception('must pass text to logger')
        if self.v:
            print(text)
        self.log_.append(text)
        
    def print_out(self, text):
        if not isinstance(text, str):
            raise Exception('must pass text to logger')
        #doesn't add to log record
        if self.v:
            print(text)
        
    def show_img(self, array):
        if not isinstance(array, np.ndarray):
            raise Exception(1)
        if self.v:
            plt.imshow(array)
        
    def get_log(self):
        return "\n".join(self.log_)
    
    
''' 
Function check_clusters can be used to check the results of our clustering: it prints out some statistics, 
it analyzes the categories and in the end does a plot.
''' 

def check_clusters(y, clust_labels, img_threshold=15, v=True):
    #checks input
    if y.ndim != 1: 
        raise Exception(2)
    if len(y) != len(clust_labels):
        raise Exception(4)
    
    #logger setup
    my_log = logger(verbose=v)
        
    #build histogram of categories (how many point for each cat)
    cats = {}
    for i in y:
        cats[i] = cats.get(i, 0) + 1
    n_cats = len(cats)
    
    #build histogram of clusters (how many point in each cluster)
    clusters = {}
    for i in clust_labels:
        clusters[i] = clusters.get(i, 0) + 1
    n_clusters = len(clusters)
        
    #create mapping from categories to index (to easily store data)
    #done because we assume y's values can be different from range(n_categories)
    #cat_list useful to quickly go back (header of result matrix)
    cat_map = {}
    cat_list = []
    for i, cat in enumerate(cats):
        cat_map[cat] = i
        cat_list.append(cat)
    
    #for each cluster, computes proportion of point belonging to each category
    result = np.zeros((n_clusters, n_cats))
    tot_per_clust = np.zeros((n_clusters,1), dtype=int)
    for i, clust in enumerate(clusters):
        labels = y[clust_labels == clust]
        tot_per_clust[i] = clusters[clust]
        for cat in labels:
            result[i,cat_map[cat]] += 1
            
    #to compute percentage of category points
    perc_cat = []
    for clust in range(len(result)):
        i_max = np.argmax(result[clust,:])
        tot = cats[cat_list[i_max]]
        perc_cat.append(result[clust, i_max] / tot * 100)
        
    #express each value as a proportion (normalization)
    result = result / tot_per_clust * 100
    
    #show graphical representation if matrix not too big
    if n_cats < img_threshold and n_clusters < img_threshold:
        my_log.show_img(result)
        
    #for each cluster show the category that fits it best
    for i,value in enumerate(np.argmax(result, axis=1)):
        #frequency of category: number of datapoint of a specific category belonging to that cluster
        #over the number of points in the cluster (variety within cluster)
        #category clustering: number of datapoint of a specific category belonging to that cluster,
        #over the total number of points of that category
        my_log.log_it('\ncluster: %s'% i)
        my_log.log_it('--> top category: %s, frequency of category (variety within cluster): %.2f%%, category clustering: %.2f%%'\
                     % (cat_list[value], result[i, value], perc_cat[i]))
        #print("tot per clust:", int(tot_per_clust[i]), "type:", type(np.float(tot_per_clust[i])), "len(y):", len(y), "type:",type(len(y)))
        temp = len(y)
        to_be_logged = ((int(tot_per_clust[i]))/temp)*100
        my_log.log_it('Cluster size: %.2f%%' % to_be_logged)
        to_be_logged = str({cat_list[i]:col for i,col in enumerate(result[i,:])})
        my_log.log_it('--> histogram: %s' % to_be_logged)
    score = np.sum(np.max(result, axis=1))/n_clusters
    weighted = np.dot(np.max(result, axis=1), np.array(perc_cat))/100
    #maybe it's best to weight the score by the category clustering index (see k-means example below)
    my_log.log_it("Overall score (doesn't consider category clustering): %.2f%%, weighted: %.2f%%"%(score, weighted))
    return weighted, my_log.get_log()

'''
Function standardize performs the standardization of an object
'''
def standardize(df, column):
    if not isinstance(column, (str, int)):
        raise Exception(1)
    #returns a copy of the standardized column
    c = df[column].copy()
    mean = c.mean()
    sd = c.std()
    return (c - mean) / sd

'''
function batch_std perform standardization on a dataset columns.
'''
def batch_std(df, columns):
    if not isinstance(columns, str):
        if len(columns) == 0:
            raise Exception('not enough columns')
    else:
        raise Exception('must be an array or list')
    #returns a new dataframe with standardized columns
    new_df = pd.DataFrame()
    for column in columns:
        temp = standardize(df, column)
        new_df[column] = temp
    return new_df

'''
class test is used for testing purposes ???
'''
class test():
    def __init__(self):
        self.best = [0,0,0]
        self.scores = []
        
    def update(self, score, k):
        score, log = score
        if score > self.best[0]:
            self.best = score, k, log
        self.scores.append((k, score))
        
    def get_result(self):
        best = self.best
        scores = self.scores
        print('best weighted score: %.2f%%, number of clusters: %i' % (best[0], best[1]))
        print('log of best: \n%s' % best[2])
        plt.figure()
        plt.plot(*zip(*scores),'-')
 

'''
class interface builds the environment from which we will inherit later in order to build our supervised learning 
classes. It has the train function, used to train our algorithm, the check_balanced and _unbal_output functions
which find unbalanced dataset, the test function to do some tests and, finally, the predict function, which we mainly
use for imputation reason (to complete our original dataset). 
'''

class interface():
    def __init__(self, seed = None, build_seed = None, data=None):
        if seed is None:
            seed = np.random.randint(666766)
        self.seed = seed
        self.build_seed = build_seed
        self.data = data
            
    def train(self, X, y, percentage=(0.70,0.15,0.15), std=False, pca=0, threshold_unbalanced=0.6, epochs=30, **args):
        np.random.seed(self.seed)
        data = self.data
        if data is None:
            data = buildTrain(X, y, percentage, std, pca, seed=self.build_seed)
        self._check_balanced(data.get_train()[1], threshold_unbalanced, args)
        train_param = (X, y, data, epochs, args)
        self._train(*train_param)
        if self.unbalanced:
            self._unbal_output(data.get_valid())
        self.data = data
        self.train_param = train_param
            
    def _check_balanced(self, y, threshold_unbalanced, args):
        unbalanced = False
        
        #check unbalanced dataset
        d_cat = {}
        clean_y = y[~y.isnull()]
        for i in clean_y:
            d_cat[i] = d_cat.get(i, 0) + 1
        max_cat = 0
        max_num = 0
        for cat in d_cat:
            if d_cat[cat] > max_num:
                max_cat = cat
                max_num = d_cat[cat] 
        if max_num / len(clean_y) > threshold_unbalanced:
            print('Warning: found unbalanced dataset, training using balanced setting for class_weight')
            if 'class_weight' in args and args['class_weight'] is None:
                class_weight = {cat: 1/(d_cat[cat]/len(clean_y)) for cat in d_cat}
                args['class_weight'] = class_weight
                print('Weights used:', {i:float('%.2f'%class_weight[i]) for i in class_weight})
            unbalanced = True
        self.unbalanced = unbalanced
        
    def _train(self, X, y, data, epochs, args):
        raise Exception('not implemented')
            
    def _unbal_output(self, valid):
        Xv, yv = valid
        obj = self.obj
        d_cat = {}
        for i in yv:
            d_cat[i] = d_cat.get(i, 0) + 1
        max_cat = 0
        max_num = 0
        for cat in d_cat:
            if d_cat[cat] > max_num:
                max_cat = cat
                max_num = d_cat[cat]
        mask = yv != max_cat
        if np.sum(mask) == 0:
            raise Exception('No data in smaller part of valid set')
        minority_score = obj.score(Xv[mask], yv[mask])
        majority_score = obj.score(Xv[~mask], yv[~mask])
        print('Score on smaller part (%.2f%%) of validation set (unbalanced case): %.2f' % 
              (np.sum(mask)/len(yv)*100, minority_score))
        print('Score on bigger part (%.2f%%) of validation set (unbalanced case): %.2f' % 
              (np.sum(~mask)/len(yv)*100, majority_score))
        print('Category histogram in validation set:', d_cat)
            
   
        
    def test(self, n=10):
        best = np.zeros(n)
        worse = np.zeros(n)
        for i in range(n):
            np.random.seed(np.random.randint(10001)*i)
            tscores, vscores,_ = self._train(self.train_param)
            best[i] = np.max(vscores)
            worse[i] = np.min(vscores)
        print('average best performance: %.2f%%, standard deviation: %f'%(best.mean(), best.std()))
        plt.figure()
        plt.plot(np.arange(n), worse, color='r', label='worse performances')
        plt.plot(np.arange(n), best, color='g', label='best performances')
        legend = plt.legend(loc='upper center', shadow=True)
        plt.xlabel('samples')
        plt.ylabel('test score')
        plt.show()
        
    def predict(self, X, y, fill_up=False):
        obj = self.obj
        assert y.shape[0] == X.shape[0]
        mask = y.isnull()
        to_be_predicted = X[mask]
        assert to_be_predicted.shape[0] != 0
        prediction = obj.predict(to_be_predicted)
        if fill_up:
            y[mask] = prediction
        return prediction
    
    def get_test(self):
        return self.data.get_test()   


'''
train_perc_warm is a function used for the training of our perceptron with warm start option. It returns
the scores on the training and validation set.
'''
def train_perc_warm(perc, data, X, y, epochs, eta0, f_eta, v):
    
    n_features = data.get_train()[0].shape[1]
    n_classes = len(data.get_train()[1].unique())
    
    if n_classes == 1:
        raise Exception(1)
    if n_classes == 2:
        n_classes = 1
        
    coef = np.random.randn(n_classes, n_features) * 1e-2
    intercept = np.random.randn(n_classes) * 1e-2
    eta = eta0
    
    tscores = []
    vscores = []
    for epoch in range(epochs):
        perc.set_params(eta0=eta)
        perc.fit(*data.get_train(), coef_init = coef, intercept_init = intercept)
        tscore = perc.score(*data.get_train())
        vscore = perc.score(*data.get_valid())
        if v:
            print("run=%i tscore=%g vscore=%g" % (epoch+1, tscore, vscore))
        tscores.append(tscore)
        vscores.append(vscore)
        coef, intercept = perc.coef_, perc.intercept_
        eta = f_eta(eta0, epoch)
    if v:
        plt.figure()
        plt.plot(np.arange(epochs), tscores, np.arange(epochs), vscores)
        
    return tscores, vscores    

'''
On the contrary, train_perc_cold is used for the training of our perceptron without warm start.
'''

def train_perc_cold(perc, data, X, y, max_iter):
    
    perc.set_params(max_iter=max_iter)
    perc.fit(*data.get_train())
    tscore = perc.score(*data.get_train())
    vscore = perc.score(*data.get_valid())
    print("tscore=%g vscore=%g" % (tscore, vscore))
        
    return tscore, vscore


'''
Class perc inherits from parent class interface and implements its own function train, which takes into account 
the differences between the presence or not of warm start. It returns, as usual, training and validation score.
'''
class perc(interface):
    
    def _train(self, X, y, data, epochs, args):
        warm_start = True
        learning_rate = args.get('learning_rate', 'optimal')
        it_interval = args.pop('it_interval', 100)
        power_t = args.get('power_t', 0.5)
        #check learning_rate
        if learning_rate == 'constant':
            f_eta = lambda eta0, epoch: eta0
        elif learning_rate == 'invscaling':
            f_eta = lambda eta0, epoch: eta0 / ((epoch + 1) * it_interval)**power_t
        elif learning_rate == 'optimal':
            warm_start = False
            f_eta = None
        else:
            raise Exception('not valid value')                
            
        #compute max_iter
        max_iter = epochs * it_interval
        args['max_iter'] = it_interval
            
        #create multiple perceptron
        perc_ = SGDClassifier(**args)

        param_warm = (perc_, data, X, y, epochs, args['eta0'], f_eta, True)
        param_cold = (perc_, data, X, y, max_iter)
        
        #perform analysis
        if warm_start:
            tscores, vscores = train_perc_warm(*param_warm)
        else:
            tscores, vscores = train_perc_cold(*param_cold)
            
        self.obj = perc_
        return tscores, vscores
    
    def __str__(self):
        return 'Perceptron interface'
    
    
'''
train_MLP does the training for a multilayer perceptron algorithm.
'''
def train_MLP(ml_perc, data, X, y, max_iter= 30, v=True):
    
    tscores = []
    vscores = []
    for epoch in range(max_iter):
        ml_perc.set_params(max_iter=epoch+1)
        ml_perc.fit(*data.get_train())
        tscore = ml_perc.score(*data.get_train())
        vscore = ml_perc.score(*data.get_valid())
        loss = ml_perc.loss_
        if v:
            print(f"epoch={epoch} loss={loss} tscore={tscore} vscore={vscore}")
        tscores.append(tscore)
        vscores.append(vscore)
        ml_perc.set_params(warm_start=True)
        
    if v:
        plt.figure()
        plt.plot(np.arange(max_iter), tscores, np.arange(max_iter), vscores)
    
    return tscores, vscores

'''
Class MLP inherits from interface and just implements its own very short _train function.
'''

class MLP(interface):
    
    def _train(self, X, y, data, epochs, args):             
            
        #create multiple perceptron
        ml_perc_ = MLPClassifier(**args)
        
        param_warm = (ml_perc_, data, X, y, epochs, True)
        
        tscores, vscores = train_MLP(*param_warm)
        
        #save settings
        self.obj = ml_perc_
        return tscores, vscores
    
    def __str__(self):
        return 'MLP interface'
    
'''
train_LR does the training for a logistic regression.
'''

def train_LR(log_reg, data, X, y, max_iter= 30, v=True):

    tscores = []
    vscores = []
    for epoch in range(max_iter):
        log_reg.set_params(max_iter=epoch+1)
        log_reg.fit(*data.get_train())
        tscore = log_reg.score(*data.get_train())
        vscore = log_reg.score(*data.get_valid())
        if v:
            print(f"epoch={epoch} tscore={tscore} vscore={vscore}")
        tscores.append(tscore)
        vscores.append(vscore)
        log_reg.set_params(warm_start=True)
        
    if v:
        plt.figure()
        plt.plot(np.arange(max_iter), tscores, np.arange(max_iter), vscores)
    
    return tscores, vscores

'''
class LogReg inherits from interface and is used for logistic regression.
'''
class LogReg(interface):
         
    def _train(self, X, y, data, epochs, args):            
        
        #create logistic regression
        log_regr_ = LogisticRegression(**args)
        
        param_warm = (log_regr_, data, X, y, epochs, True)
        
        tscores, vscores = train_LR(*param_warm)
        
        #save settings
        self.obj = log_regr_
        return tscores, vscores
    def __str__(self):
        return 'Logistic Regression interface'
    
    
'''
train_trees does the training for a decision tree (or any algorithm of this family)
'''

def train_trees(my_tree, data, X, y):
    
    my_tree.fit(*data.get_train())
    tscore = my_tree.score(*data.get_train())
    vscore = my_tree.score(*data.get_valid())
    print("tscore=%g vscore=%g" % (tscore, vscore))
        
    return tscore, vscore

'''
class trees inheriths from interface with the possibility of deciding among decision trees, random forests or extreme
random forests. we also implemented a view_tree function, which can be applied just to decision trees,
that shows the graph of our built tree
'''
class trees(interface):
    
    def train(self, X, y, percentage=(0.70,0.15,0.15), std=False, pca=0, threshold_unbalanced=0.6, tree_type='RF', **args):
        np.random.seed(self.seed)
        data = self.data
        if data is None:
            data = buildTrain(X, y, percentage, std, pca, seed=self.build_seed)
        self._check_balanced(data.get_train()[1], threshold_unbalanced, args)
        train_param = (X, y, data, tree_type, args)
        self._train(*train_param)
        if self.unbalanced:
            self._unbal_output(data.get_valid())
        self.data = data
        self.tree_type = tree_type
        self.train_param = train_param
        
    def _train(self, X, y, data, tree_type, args):
        if tree_type == 'RF':
            tree_ = RandomForestClassifier(**args)
        elif tree_type == 'DT':
            tree_ = tree.DecisionTreeClassifier(**args)
        elif tree_type == 'XRF':
            tree_ = ExtraTreesClassifier(**args)
        else:
            raise Exception(1)
            
        param_warm = (tree_, data, X, y)
        tscores, vscores = train_trees(*param_warm)
        
        #save settings
        self.obj = tree_
        return tscores, vscores
    
    def view_tree(self, **args):
        if self.tree_type == 'DT':
            dot_data = tree.export_graphviz(self.obj, out_file=None,
                             filled=True, rounded=True, special_characters=True, **args)
            graph = graphviz.Source(dot_data)  
            graph.view()
        else:
            print("Can't show tree for this model")
        
    def __str__(self):
        tree_type = self.tree_type
        if tree_type == 'RF':
            return 'Random Forest Interface'
        elif tree_type == 'DT':
            return 'Decision Tree Interface'
        elif tree_type == 'XRF':
            return 'Extremely Randomized Trees Interface'
        else:
            raise Exception(1)
            

'''
Function test_sup is the one we used for imputation. It takes in input all the objects we want to compare,
and returns as output the object among those with highest score in the test set 
(and plots a graph comparing the scores)
'''
def test_sup(*objs):
    scores = np.zeros(len(objs))
    c = 0
    plt.figure()
    for obj in objs:
        X, y = obj.data.get_test()
        if not isinstance(obj, interface):
            raise Exception('must pass interface subclass object')
        print(X.shape, y.shape)
        score = obj.obj.score(X, y)
        print(score)
        scores[c] = score
        print(scores)
        c += 1
        print(c)
        plt.plot(c, score, '.', label=str(obj))
    legend = plt.legend(loc=(1.01, 0), shadow=True)
    plt.show()
    i_max = np.argmax(scores)
    print('best is %s with score %.2f' % (str(objs[i_max]), scores[i_max]))
    return objs[i_max]

'''
class GridSearch is used for optimizing weights through a grid search. We implemented this
as a greedy randomized algorithm, due to the amount of time that an exhaustive search would have required.
'''
class GridSearch():
    def __init__(self, seed = None, build_seed = None, **args):
        if seed is None:
            seed = np.random.randint(666766)
        self.seed = seed
        self.build_seed = build_seed
         
    def get_best(self, X, y, obj, percentage=(0.8,0.1,0.1), std=False, pca=False, one_hot=False, cat_col=None, epochs=5, 
                 wmin=0, wmax=1, weights=None, start_config=None, data=None):
        if data is None:
            data = buildTrain(X, y, percentage, std, pca, self.build_seed, one_hot, cat_col)
        
        train = data.get_train()[0]
        n_features = train.shape[1]
           
        if weights is None:
            mid_val = (wmin+wmax)/2
            weights = np.array([wmin, mid_val, wmax])
        if start_config is None:
            best_config = np.ones(n_features)
        else:
            best_config = start_config
            
        scores = [0]
        c = 0
        s_time = time.time()
        for feature in np.random.randint(n_features, size=epochs*n_features):
            best_score = 0
            for weight in weights:
                if c %10000 == 0:
                    print(c)
                c += 1
                config = best_config.copy()
                config[feature] = weight
                temp = np.eye(n_features)*config
                X_train_mod = np.dot(train, temp)
                X_valid_mod = np.dot(data.get_valid()[0], temp)
                obj.fit(X_train_mod, data.get_train()[1])
                score = obj.score(X_valid_mod, data.get_valid()[1])
                if score > best_score:
                    scores.append(score)
                    best_score = score
                    best_config[feature] = weight
        print('elapsed time:', time.time()-s_time)
        plt.figure()
        plt.plot(np.arange(len(scores)), scores)
        return best_config, best_score
    
    def score(self, pred_labels, true_labels):
        return metrics.homogeneity_score(pred_labels, true_labels)       