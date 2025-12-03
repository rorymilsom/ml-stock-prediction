import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

#Importing Data
tesla = pd.read_csv('Tesla.csv')
print(tesla.head())

#Exploratory Data Analysis (EDA)

#Visualise all of the closing prices
plt.figure(figsize=(15,5))
plt.plot(tesla["Close"])
plt.title("Tesla Close Price - per day")
plt.ylabel("Price ($)")
plt.xlabel("Days")
#plt.show()

#Checking if Close and Adj Close equal eachother so I can delete Adj Close
if tesla["Close"].equals(tesla["Adj Close"]):
    print("All Values Equal!")
else:
    print("Not All Values Equal")


#Equal so drop Adj Close
tesla.drop(['Adj Close'], axis=1,inplace=True)
print(tesla.columns)

#Print all the nulls (there is none)
print(tesla.isnull().sum())

#Plotting density of features

features = ['Open', 'High', 'Low', 'Close', 'Volume']
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
    plt.subplot(2,3, i+1)
    sb.distplot(tesla[col])
#plt.show()

#Every data has two high peaks and volume is left biased (0-0.75)

plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
    plt.subplot(2,3, i+1)
    sb.boxplot(tesla[col])
#plt.show()

#Splitting data by time and day
splitted = tesla["Date"].str.split('/', expand=True)
tesla['Month'] = splitted[0].astype(int)
tesla['Day'] = splitted[1].astype(int)
tesla["Year"] = splitted[2].astype(int)


tesla['is_quarter_end'] = np.where(tesla['Month']%3==0,1,0)

print(tesla.head())

date_grouped = tesla.drop('Date', axis=1).groupby('Year').mean()
plt.subplots(figsize=(20,10))
for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
    plt.subplot(2,2, i+1)
    date_grouped[col].plot.bar()
#plt.show()

is_quarter = tesla.drop('Date', axis=1).groupby('is_quarter_end').mean()
is_quarter.head()

tesla['open-close'] = tesla['Open'] - tesla['Close']
tesla['low-high'] = tesla['Low'] - tesla['High']
tesla['target'] = np.where(tesla['Close'].shift(-1) > tesla['Close'],1, 0)

plt.pie(tesla['target'].value_counts().values, labels=[0,1],autopct='%1.11f%%')
plt.show()

plt.figure(figsize=(10,10))

sb.heatmap(tesla.drop('Date', axis=1).corr() > 0.9, annot=True, cbar=False)
plt.show()

features = tesla[['open-close', 'low-high', 'is_quarter_end']]
scaler = StandardScaler()
scaled = scaler.fit_transform(features)
target = tesla['target']
X_train, X_valid, Y_train, Y_valid = train_test_split(features, target, test_size=0.1, random_state=30834)

models = [LogisticRegression(), SVC(kernel='poly', probability=True), XGBClassifier()]
for i in range(3):
    models[i].fit(X_train, Y_train)
    print("Model is: " + f'{models[i]}')
    print("Training Accuracy: ", metrics.roc_auc_score(Y_train, models[i].predict_proba(X_train)[:,1]))
    print("Validation Accuracy: ", metrics.roc_auc_score(Y_valid, models[i].predict_proba(X_valid)[:,1]))
