#using necessary libraries
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#Loading iris dataset into panda data frame

iris = pd.read_csv('iris.csv')
#view of general data
print(iris.head(10))
print(iris.info())
print(iris.describe()) 
print(iris['species'].value_counts())


#visual inspection of whether there is a relationship between independent and dependent metrics (categories)
sns.pairplot(data = iris, hue = 'species')
#plt.savefig("pairplot.png")

#form a set of independent and dependent metrics
y = iris['species']
x = iris.drop(columns = 'species')
print("Total lens", len(x), len(y))

#formation of separate sets for forecast model training and testing
x_train, x_test, y_train, y_test = train_test_split(x,y)

print("Train len", len(x_train), len(y_train))
print("Test len", len(x_test), len(y_test))
print(x_train)

#random model evaluation
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
random_y_test = np.random.choice(np.unique(y), size=len(y_test))
#print(y_test)
#print(random_y_test)
random_accuracy_score = accuracy_score(y_test, random_y_test)
print("Random accuracy score = ", random_accuracy_score)

