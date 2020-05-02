
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #Plotting library

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv('Data/train.csv')
test = pd.read_csv('Data/test.csv')


# In[3]:


test.set_index('Unnamed: 0', inplace=True)
train.set_index('Unnamed: 0', inplace=True)


# In[4]:


test.rename( columns={'Unnamed: 0':'Id'}, inplace=True )


# In[5]:


train[['MLast','NDon','TotV','MFirst','M2017']]=train[['Months since Last Donation', 'Number of Donations',
       'Total Volume Donated (c.c.)', 'Months since First Donation',
       'Made Donation in March 2007']]
test[['MLast','NDon','TotV','MFirst']]=test[['Months since Last Donation', 'Number of Donations',
       'Total Volume Donated (c.c.)', 'Months since First Donation']]

train.drop(['Months since Last Donation', 'Number of Donations',
       'Total Volume Donated (c.c.)', 'Months since First Donation',
       'Made Donation in March 2007'], axis = 1, inplace = True)
test.drop(['Months since Last Donation', 'Number of Donations',
       'Total Volume Donated (c.c.)', 'Months since First Donation'], axis = 1, inplace = True)


# In[6]:


train['Freq'] = (train['MFirst'] - train['MLast']) / train['NDon']
test['Freq'] = (test['MFirst'] - test['MLast']) / test['NDon']


# In[68]:


freq = list()
freq_x = list()
for i in range(1,30):
    a=i+1
    freq.append(train['M2017'][train["Freq"].between(i,a)].mean())
    freq_x.append(i)

plt.figure(figsize=(5, 5), dpi=120, facecolor='w', edgecolor='k')
plt.title('Donation rate(Freq)')
plt.plot(freq_x, freq, 'bo', ls = '-')


# In[12]:


ndon = list()
ndon.append(train['M2017'][train["NDon"].between(0,2)].value_counts(normalize=True).sort_index()[1])
ndon.append(train['M2017'][train["NDon"].between(2,3)].value_counts(normalize=True).sort_index()[1])
ndon.append(train['M2017'][train["NDon"].between(3,4.44)].value_counts(normalize=True).sort_index()[1])
ndon.append(train['M2017'][train["NDon"].between(4.44,6)].value_counts(normalize=True).sort_index()[1])
ndon.append(train['M2017'][train["NDon"].between(6,8)].value_counts(normalize=True).sort_index()[1])
ndon.append(train['M2017'][train["NDon"].between(8,11)].value_counts(normalize=True).sort_index()[1])
ndon.append(train['M2017'][train["NDon"].between(11,50)].value_counts(normalize=True).sort_index()[1])
ndon_x = list()
ndon_x=[2,3,4,6,8,11,50]


# In[13]:


plt.figure(figsize=(5, 5), dpi=120, facecolor='w', edgecolor='k')
plt.title('Donation rate(No Donations)')
plt.plot(ndon_x, ndon, 'bo', ls = '-')


# In[69]:


ndon = list()
ndon_x = list()
for i in range(1,50):
    a=i+1
    ndon.append(train['M2017'][train["NDon"].between(i,a)].mean())
    ndon_x.append(i)

plt.figure(figsize=(5, 5), dpi=120, facecolor='w', edgecolor='k')
plt.title('Donation rate(No Donations)')
plt.plot(ndon_x, ndon, 'bo', ls = '-')


# In[70]:


mfir = list()
mfir_x = list()
for i in range(1,100):
    a=i+1
    mfir.append(train['M2017'][train["MFirst"].between(i,a)].mean())
    mfir_x.append(i)

plt.figure(figsize=(5, 5), dpi=120, facecolor='w', edgecolor='k')
plt.title('Donation rate(Months since first donation)')
plt.plot(mfir_x, mfir, 'bo', ls = '-')


# In[72]:


mlas = list()
mlas_x = list()
for i in range(1,100):
    a=i+1
    mlas.append(train['M2017'][train["MLast"].between(i,a)].mean())
    mlas_x.append(i)

plt.figure(figsize=(5, 5), dpi=120, facecolor='w', edgecolor='r')
plt.title('Donation rate(Months since last donation)')
plt.plot(mlas_x, mlas, 'bo', ls = '-')


# In[75]:


tot = list()
tot_x = list()
for i in range(1,10000,500):
    a=i+500
    tot.append(train['M2017'][train["TotV"].between(i,a)].mean())
    tot_x.append(i)

plt.figure(figsize=(5, 5), dpi=120, facecolor='w', edgecolor='k')
plt.title('Donation rate(Total volume)')
plt.plot(tot_x, tot, 'bo', ls = '-')


# <h1> Model </h1>

# In[18]:


train.M2017 = train.M2017.astype(int)


# In[41]:


data=train.append(test)

data.drop('MFirst', axis = 1, inplace = True)

nauka = data.iloc[:575,:]

testy = data.iloc[576:,:]

testX = testy.drop('M2017', axis = 1)

trainY = nauka['M2017']

trainX = nauka.drop('M2017', axis = 1)


# <h1> MODEL </h1>

# In[42]:


nauka_D = trainX
nauka_W = trainY
testy = testX


# In[43]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

reglog = LogisticRegression()
reglog.fit(trainX, trainY)
Y_predLR = reglog.predict(testX)
acc_log = round(reglog.score(trainX, trainY) * 100, 2)
acc_log

knn = KNeighborsClassifier(n_neighbors = 11)
knn.fit(trainX, trainY)
Y_predKNN = knn.predict(testX)
acc_knn = round(knn.score(trainX, trainY) * 100, 2)
acc_knn

gaussian = GaussianNB()
gaussian.fit(trainX, trainY)
Y_predG = gaussian.predict(testX)
acc_gaussian = round(gaussian.score(trainX, trainY) * 100, 2)
acc_gaussian

perceptron = Perceptron()
perceptron.fit(trainX, trainY)
Y_predP = perceptron.predict(testX)
acc_perceptron = round(perceptron.score(trainX, trainY) * 100, 2)
acc_perceptron

svc = SVC()
svc.fit(trainX, trainY)
Y_pred = svc.predict(testX)
acc_svc = round(svc.score(trainX, trainY) * 100, 2)
acc_svc

linear_svc = LinearSVC()
linear_svc.fit(trainX, trainY)
Y_predLSVC = linear_svc.predict(testX)
acc_linear_svc = round(linear_svc.score(trainX, trainY) * 100, 2)
acc_linear_svc

sgd = SGDClassifier()
sgd.fit(trainX, trainY)
Y_predSGD = sgd.predict(testX)
acc_sgd = round(sgd.score(trainX, trainY) * 100, 2)
acc_sgd

decision_tree = DecisionTreeClassifier()
decision_tree.fit(trainX, trainY)
Y_predD = decision_tree.predict(testX)
acc_decision_tree = round(decision_tree.score(trainX, trainY) * 100, 2)
acc_decision_tree

random_forest = RandomForestClassifier(n_estimators=400, max_features = 'log2', oob_score = True, n_jobs = -1)
random_forest.fit(trainX, trainY)
Y_predRF = random_forest.predict(testX)
random_forest.score(trainX, trainY)
acc_random_forest = round(random_forest.score(trainX, trainY) * 100, 2)
acc_random_forest

print('LogisticRegression =', acc_log)
print('KNeighborsClassifier =', acc_knn)
print('GaussianNB =', acc_gaussian)
print('Perceptron =', acc_perceptron)
print('LinearSVC =', acc_linear_svc)
print('SVC =', acc_svc)
print('SGDClassifier =', acc_sgd)
print('DecisionTreeClassifier =', acc_decision_tree)
print('RandomForestClassifier =', acc_random_forest)


# In[59]:


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=10000, oob_score = True, n_jobs = -1, max_features = 'sqrt')
random_forest.fit(nauka_D, nauka_W)
Y = random_forest.predict_proba(testy)
acc_random_forest = round(random_forest.score(nauka_D, nauka_W) * 100, 2)
WYN = Y[:,1]
acc_random_forest


# In[60]:


test.reset_index(inplace = True)


# In[61]:


test.rename( columns={'Unnamed: 0':'id'}, inplace=True )


# In[62]:


submission = pd.DataFrame({
        "": test['id'],
        "Made Donation in March 2007": WYN
    })


# In[64]:


submission.to_csv('/home/jan/Dokumenty/Python_kaggle/Blood_donation/submission.csv', index=False)


# In[63]:


submission

