from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


data_link = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
col_names = ["variance", "skewness", "curtosis", "entropy", "class"]

bankdata = pd.read_csv(data_link, names=col_names, sep=",", header=None)
print(bankdata.head())

# Display summary statistics
# print(bankdata.describe().T)

y = bankdata['class']
X = bankdata.drop('class', axis=1) # axis=1 means dropping from the column axis


SEED = 42

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = SEED)

xtrain_samples = X_train.shape[0]
xtest_samples = X_test.shape[0]

print(f'There are {xtrain_samples} samples for training and {xtest_samples} samples for testing.')

svc = SVC(kernel='linear')
svc.fit(X_train, y_train)

svc_predictions = svc.predict(X_test)


cm = confusion_matrix(y_test, svc_predictions)
sns.heatmap(cm, annot=True, fmt='d').set_title('Confusion matrix of linear SVM') # fmt='d' formats the numbers as digits, which means integers

print(classification_report(y_test, svc_predictions))



















# import matplotlib.pyplot as plt

# for col in bankdata.columns[:-1]:
#     plt.title(col)
#     bankdata[col].plot.hist() #plotting the histogram with Pandas
#     plt.show();

# import seaborn as sns

# for feature_1 in bankdata.columns[:-1]:
#     for feature_2 in bankdata.columns[:-1]:
#         if feature_1 != feature_2: # test if the features are different
#             print(feature_1, feature_2) # prints features names
#             sns.scatterplot(x=feature_1, y=feature_2, data=bankdata, hue='class') # plots each feature points with its color depending on the class column value
#             plt.show();