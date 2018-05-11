import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

import VodafoneChallenge_Classes as VCC
import graphviz
import time

pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=np.NaN)


'''
This .py file is intended to be a general code that can be applied to any dataframe df passed as input into
the following functions. We wrote this using the dataframe we have as test input, but the code below wants
to be the more generic possible to do imputation on an incomplete dataset and then some clustering.
'''


df_backup = pd.read_csv('dataset_challenge_v5.TRAINING.csv')
df_clean = df_backup.copy()



'''
The first part of this code is for the cleaning. This is the most specific part since it involves the cleaning 
suited for some specific columns, but still this can be applied for example to a dataframe of the same type of the 
one we have, just with more datapoints than the 2000 we have. 
Some instructions rely on external datasets, for example while retrieving rurual info.
'''
del df_clean['Unnamed: 0']

c = list(df_clean.columns)
c[0] = 'ID'
df_clean.columns = c

df_clean['ZipCode'] = df_clean['ZipCode'].map(lambda x: '%05i' % x, na_action='ignore')

traffic_columns = ['File-Transfer', 'Games',
       'Instant-Messaging-Applications', 'Mail', 'Music-Streaming',
       'Network-Operation', 'P2P-Applications', 'Security',
       'Streaming-Applications', 'Terminals', 'Unclassified', 'VoIP',
       'Web-Applications']
df_clean[traffic_columns]

cats = df_clean['CustomerAge'].astype('category').cat.categories
d = {cat:(15+10*i)/100 for i,cat in enumerate(cats)}
df_clean['NumericAge'] = df_clean['CustomerAge'].map(lambda x: d[x], na_action='ignore')

d = {}
for elem in df_clean['DeviceOperatingSystem']:
    d[elem] = d.get(elem, 0) + 1
print(d) #some categories have very few values, group them
OS_other = []
for key in d:
    if d[key] < 10:
        OS_other.append(key)
        d[key] = 'other'
    else:
        d[key] = key
df_clean['OS_clean'] = df_clean['DeviceOperatingSystem'].map(lambda x: d[x], na_action='ignore')

#Adding rural/urban information
df_zip_istat = pd.read_csv('databases/database.csv')
df_istat_urb = pd.read_csv('databases/it_postal_codes.csv/Foglio 2-Tabella 1.csv', error_bad_lines=False, sep = ';')
my_urb_dict = {'Basso' : 0, 'Medio' : 1, 'Elevato' : 2}
df_istat_urb['GradoUrbaniz'] = df_istat_urb['GradoUrbaniz'].map(lambda x: my_urb_dict[x], na_action = 'ignore')

#check there are no datapoint for which we don't have zip but we've region
df_clean['ZipCode'].isnull()
df_clean['Region'][df_clean['ZipCode'].isnull()]
len(df_clean['Region'][df_clean['ZipCode'].isnull()]) == np.sum(df_clean['Region'][df_clean['ZipCode'].isnull()].isnull())

#we need to insert x for multiple cap cities
isnan = lambda x: x != x
#nan is unique type not equal to itself, so with this lambda function we get True only when the type is NaN

for i in range(df_zip_istat.shape[0]):
    cap = df_zip_istat.loc[i, 'cap/0']
    cap  = '%05d' % cap
    if not isnan(df_zip_istat.loc[i,'cap/1']):
        if not isnan(df_zip_istat.loc[i,'cap/10']):   
            cap = cap[:-2]+'xx'
        else:
            cap = cap[:-1]+'x'
    df_zip_istat.loc[i, 'cap/0'] = cap

d_zip_istat = df_zip_istat.set_index('cap/0').to_dict()['codice']
d_istat_urb = df_istat_urb.set_index('ISTAT').to_dict()['GradoUrbaniz']

mask = df_clean['ZipCode'].isnull()
urban_col = np.zeros(df_clean.shape[0])
urban_col_masked = urban_col[~ mask]
d_zip_istat.update([('51021', 47023),( '83026', 64121),( '74025', 73007),( '55062', 46007),( '38039', 22217),('50037', 48053)])
d_istat_urb.update([(22250, 0),( 78157, 1)])

c = 0
for i in df_clean['ZipCode'][~ mask]:
    try:
        temp = d_zip_istat[i]
        urban_col_masked[c] = d_istat_urb[int(temp)]
    except KeyError:
        i = '%05d' % int(i)
        if i[:-1]+'x' in d_zip_istat:
            temp = d_zip_istat[i[:-1]+'x']
        elif i[:-2]+'xx' in d_zip_istat:
            temp = d_zip_istat[i[:-2]+'xx']
        else:
            raise()
    c += 1
    
df_clean['Urban'] = df_clean['ZipCode'].copy()
df_clean['Urban'][~ mask] = urban_col_masked


'''
The second part of this code is for imputation of missing values. On our notebook we used the function 
test_sup to find which supervised learning algorithm best performed the prediction. Here we base our 
imputation on those findings and therefore we avoid to run the test_sup function several times. 
The algorithms analyzed by test_sup were perceptron, MLP, logistic regression and trees (decision tree, random forest, extreme
random forest) and in the end we found out that MLP and XRF are the ones that usually gave us the best
prediction scores.
'''

df_filled = df_clean.copy()
build_seed = 456245


#imputation for operating system, done with XRF
percentage_used = (0.70,0.15,0.15)
X = df_filled[traffic_columns]
y = df_filled['OS_clean']

my_forest = VCC.trees(build_seed)
my_forest.train(X, y, percentage=(0.70,0.15,0.15), std=False, pca=0, threshold_unbalanced=0.6, tree_type='XRF',
                n_estimators = 50, max_features = "auto", criterion = "entropy", max_depth = 15,
                min_samples_split = 50, n_jobs = -1)
os_missing = my_forest.predict(X,y, fill_up=True)

#imputation for urbanization, done with MLP
X = df_filled[traffic_columns]
df_filled['Urban'] = df_filled['Urban'].map(lambda x: int(x), na_action = 'ignore')
y = df_filled['Urban']

my_MLP = VCC.MLP(build_seed)
my_MLP.train(X, y, percentage=(0.70,0.15,0.15), std=False, pca=0, threshold_unbalanced=0.6, epochs=90,
             hidden_layer_sizes = (200,), batch_size = 50, learning_rate_init=1e-4, solver = 'adam', 
             learning_rate = 'constant', momentum = 0.5, nesterovs_momentum = False,
             alpha = 0.10, tol = 1e-4)
urbanization = my_MLP.predict(X,y, fill_up=True)

#imputation for Numeric Age, done with Perceptron
dict_numage_to_agecat = {0.85: 2, 0.65: 1, 0.35: 0, 0.75: 1, 0.55: 1, 0.45: 1, 0.25: 0, 0.15: 0}
df_filled["NumericAge"] = df_filled["NumericAge"].map(lambda x: dict_numage_to_agecat[x], na_action = 'ignore')
X = df_filled[traffic_columns]
y = df_filled['NumericAge']


my_perc = VCC.perc(build_seed)
my_perc.train(X, y, percentage=(0.70,0.15,0.15), std=False, pca=0, threshold_unbalanced=0.6, epochs=50,  
              loss='log', penalty='none', alpha=1e-8, power_t=0.7, it_interval=100, learning_rate='constant', 
              eta0=1e-4, class_weight={2.0: 261.4, 1.0: 1.54, 0.0: 4.20})
num_age = my_perc.predict(X,y, fill_up=True)

#imputation for data allowance, done with MLP over a masked dataset of points below 0.5 (since looking at the
#density we noticed that there are just few outliers above 0.5)
mask = df_clean['DataAllowance'] > 0.5
X = df_filled[traffic_columns][~mask]
df_filled['DataAllowance'] = df_filled['DataAllowance'].map(lambda x: '%.1f'%x, na_action = 'ignore')
y = df_filled['DataAllowance'][~mask]

my_MLP = VCC.MLP(build_seed)
my_MLP.train(X, y, percentage=(0.70,0.15,0.15), std=False, pca=0, threshold_unbalanced=0.6, epochs=50,
             hidden_layer_sizes = (400,), batch_size = 100, learning_rate_init=1e-4, solver = 'adam', 
             learning_rate = 'constant', momentum = 0.5, nesterovs_momentum = False,
             alpha = 0.05, tol = 1e-4)
#to predict we go back to our entire column
X = df_filled[traffic_columns]
y = df_filled['DataAllowance']
data_all = my_MLP.predict(X,y, fill_up=True)

#imputation for Monthly Data Traffic, done with XRF
X = df_filled[traffic_columns]
df_filled['MonthlyDataTraffic'] = df_filled['MonthlyDataTraffic'].map(lambda x: '%.1f'%x, na_action = 'ignore')
y = df_filled['MonthlyDataTraffic']

my_forest = VCC.trees(build_seed)
my_forest.train(X, y, percentage=(0.70,0.15,0.15), std=False, pca=0, threshold_unbalanced=0.6, tree_type='XRF',
                n_estimators = 50, max_features = "auto", criterion = "entropy", max_depth = 15,
                min_samples_split = 50, n_jobs = -1)
data_traffic = my_forest.predict(X,y, fill_up=True)

#imputation for data ARPU, done with XRF
X = df_filled[traffic_columns]
df_filled['DataArpu'] = df_filled['DataArpu'].map(lambda x: '%.1f'%x, na_action = 'ignore')
y = df_filled['DataArpu']

my_forest = VCC.trees(build_seed)
my_forest.train(X, y, percentage=(0.70,0.15,0.15), std=False, pca=0, threshold_unbalanced=0.6, tree_type='XRF',
                n_estimators = 50, max_features = "auto", criterion = "entropy", max_depth = 15,
                min_samples_split = 50, n_jobs = -1)
data_arpu = my_forest.predict(X,y, fill_up=True)

#imputation for Monthly Voice TrafficCount, done with XRF
X = df_filled[traffic_columns]
df_filled['MonthlyVoiceTrafficCount'] = df_filled['MonthlyVoiceTrafficCount'].map(lambda x: '%.1f'%x, na_action = 'ignore')
y = df_filled['MonthlyVoiceTrafficCount']

my_forest = VCC.trees(build_seed)
my_forest.train(X, y, percentage=(0.70,0.15,0.15), std=False, pca=0, threshold_unbalanced=0.6, tree_type='XRF',
                n_estimators = 50, max_features = "auto", criterion = "entropy", max_depth = 15,
                min_samples_split = 50, n_jobs = -1)
voice_traffic_c = my_forest.predict(X,y, fill_up=True)


#Once imputation is finished, we build our completed dataset df_good and we print some info. 
#We also have to add to MonthlySMSTrafficCount one unique value (we've a single nan). We choose to do that
#very simply with the mean.
df_good = df_filled.copy()
col_to_del = ['CustomerAge', 'DevicePrice', 'Province', 'Region', 'ZipCode', 'DeviceOperatingSystem']
df_good.drop(col_to_del, axis=1, inplace=True)
df_good['MonthlySmsTrafficCount'][df_good['MonthlySmsTrafficCount'].isnull()] = df_good['MonthlySmsTrafficCount'].mean()
df_good.info()




'''
VOGLIAMO AGGIUNGERE QUI I GRAFICI ???? IO LI AGGIUNGEREI, VEDI TU COSA PREFERISCI.
'''




'''
The third part of this code is for finding the optimal weights, through a greedy grid search, by using KNN. We will
then use these to modify the dataset we'll pass as input to our clustering algorithms (in order to avoid problem 
with the distance)
'''

df = df_good.copy()
col = df.columns[2:]

d_map = {'iOS': 1, 'Android': 2, 'other': 3}
df['OS_clean'] = df['OS_clean'].map(d_map, na_action='ignore')
cat_map = {'V-Bag': 1, 'V-Auto': 2, 'V-Pet': 3, 'V-Camera': 4}
df['Product'] = df['Product'].map(cat_map, na_action='ignore')

cat_col = [i for i in col if i not in traffic_columns]
non_cat_col = [i for i in col if i not in cat_col]
cat_col.pop(cat_col.index('MonthlySmsTrafficCount'))

X = df[col]
y = df['Product']
data = VCC.buildTrain(X, y,  perc=(0.3,0.2,0.5), std=False, pca=0, seed=None, one_hot=True, cat_col=cat_col)

knn1 = KNeighborsClassifier(n_neighbors=4)
weights = np.linspace(0, 10.0, num = 70)

grid = VCC.GridSearch(build_seed=647645)


#result = grid.get_best(X, y, knn1, percentage=(0.3,0.2,0.5), std=False, pca=0, one_hot=True, cat_col=cat_col, epochs=1, 
#                wmin=0, wmax=1, weights=None, start_config=None, data=data)
'''
we comment out the actual grid search because it requires some time to run. We instead save into a new variable
the weights found.
'''

optimal_weights = np.array([0, 0, 0, 0.86956522, 0, 7.53623188, 0, 0, 6.08695652, 0,
        0, 2.89855072, 0.43478261, 0.43478261, 1.01449275, 1.01449275, 1.01449275, 1.01449275, 1.01449275, 1.88405797,
        0, 1.88405797, 0.28985507, 0, 0, 2.02898551, 0.72463768, 1.01449275, 0.14492754, 1.88405797,
        0, 0, 0, 0, 0, 1.15942029, 1.01449275, 0.28985507, 1.01449275, 1.01449275,
        1.01449275, 1.15942029, 1.01449275, 0, 0, 0, 0, 0, 1.01449275, 1.01449275,
        2.17391304, 0.43478261, 0.86956522, 0.86956522, 0, 0.57971014, 0, 1.30434783, 1.01449275, 2.17391304, 0, 0.72463768, 
        1.01449275, 2.17391304, 1.01449275, 1.01449275, 0, 0.86956522, 1.01449275, 1.01449275])

train = data.get_train()[0]
mask = optimal_weights>0


'''
The final part of this code is the actual clustering part. We decided to implement two unsupervised clustering,
Hierarchical Clustering and KMeans, on a dataset modified through the weights found above. We then decided to try 
also two supervised approaches, with KNN (NON VOLEVAMO FARLO ?? SECONDO ME NON L'ABBIAMO MAI SCRITTO), 
which may be particularly relevant since we used it to optimize weights, and a decision tree.
'''

X_one_hot = data.get_one_hot().loc[:, mask]
temp = np.eye(X_one_hot.shape[1]) * optimal_weights[mask]

X_mod = pd.DataFrame(np.dot(X_one_hot, temp))
data = VCC.buildTrain(X_mod, y, perc=(0.8,0.2,0), std=False, pca=False, seed = 222253)

#KMeans:
for k in range(2, 8):
    km = KMeans(n_clusters=k)
    km.fit(*data.get_train())
    l = km.labels_
    print(('\n ****** kmeans: %i ******' % k))
    #print('\n',k, km.score(*data.get_valid()), metrics.v_measure_score(data.get_train()[1], l))
    for cl in range(k):
        #print('k-means',k, 'cluster', cl, 'proportion', np.sum(l == cl)/len(l))
        if np.sum(l == cl)/len(l) < 0.01:
            pass
            #print('k-means',k, 'cluster', cl)
    VCC.check_clusters(y=data.get_train()[1], clust_labels=l)
    
    
#hierarchical clustering:
for k in range(2, 8):
    hc = AgglomerativeClustering(n_clusters=k, linkage='complete')
    hc.fit(*data.get_train())
    l = hc.labels_
    print(('\n ****** kmeans: %i ******' % k))
    #print('\n',k, km.score(*data.get_valid()), metrics.v_measure_score(data.get_train()[1], l))
    for cl in range(k):
        #print('k-means',k, 'cluster', cl, 'proportion', np.sum(l == cl)/len(l))
        if np.sum(l == cl)/len(l) < 0.01:
            pass
            #print('k-means',k, 'cluster', cl)
    VCC.check_clusters(y=data.get_train()[1], clust_labels=l)
    
#Decision Tree:
print('\n Decision Tree:')
data = VCC.buildTrain(X, y,  perc=(0.3,0.2,0.5), std=False, pca=0, seed=None, one_hot=True, cat_col=cat_col)
my_forest = VCC.trees(build_seed=23456, data=data)
#We still pass X and y even unnecessary since we passed data into the constructor.
my_forest.train(X, y, percentage=(0.70,0.15,0.15), std=False, pca=0, threshold_unbalanced=0.6, tree_type='DT',
                max_depth=4)
#my_forest.view_tree(feature_names=feature_names)
#Volevo fare il plot del grafico ma non so piu dove abbiamo messo feature_names... forse son solo stanco