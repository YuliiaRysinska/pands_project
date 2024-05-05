import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
import sklearn


#Loading iris dataset into panda data frame
file_name = 'iris.csv'
iris = pd.read_csv('iris.csv')
#view of general data
print(iris.head(1))
print(iris.info())
print(iris.describe()) 
print(iris['species'].value_counts())

#visual inspection of whether there is a relationship between independent and dependent metrics (categories)



