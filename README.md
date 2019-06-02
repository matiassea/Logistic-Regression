# Logistic-Regression

### Overview


Today I know that the most important thing to learn to become a Data Scientist is the pipeline, i.e, the process of getting and processing data, understanding the data, building the model, evaluating the results (both of the model and the data processing phase) and deployment. So as a TL;DR for this post: Learn Logistic Regression first to become familiar with the pipeline and not being overwhelmed with fancy algorithms.

![pipeline](https://user-images.githubusercontent.com/17385297/50397705-16eeb980-0751-11e9-9fe9-4da4cb716908.PNG)

[Source](https://www.kdnuggets.com/2018/05/5-reasons-logistic-regression-first-data-scientist.html/).


### Overall

Logistic regression is another technique borrowed by machine learning from the field of statistics. It is the go-to method for binary classification problems (problems with two class values).

Logistic regression is like linear regression in that the goal is to find the values for the coefficients that weight each input variable. Unlike linear regression, the prediction for the output is transformed using a non-linear function called the logistic function.

The logistic function looks like a big S and will transform any value into the range 0 to 1. This is useful because we can apply a rule to the output of the logistic function to snap values to 0 and 1 (e.g. IF less than 0.5 then output 1) and predict a class value.

[Source](https://www.kdnuggets.com/2018/02/tour-top-10-algorithms-machine-learning-newbies.html/).


### Because “Logistic Regression” is (sometimes) enough

Logistic regression is a very powerful algorithm, even for very complex problems it may do a good job. Take MNIST for example, you can achieve 95% accuracy using Logistic Regression only, it’s not a great result, but its more than good enough to make sure you pipeline works. Actually, with the right representation of the features, it can do a fantastic job. When dealing with non-linear problems, we sometimes try to represent the original data in a way that may be explained linearly. Here’s a small example of this idea: We want to perform a simple classification task on the following data:







### Application

Following are the steps involved in creating a well-defined ML project:

1. Understand and define the problem
2. Analyse and prepare the data
3. Apply the algorithms
4. Reduce the errors
5. Predict the result

DataBase


![db1](https://user-images.githubusercontent.com/17385297/50398129-ff650000-0753-11e9-9e2f-3d54fdff016d.PNG)



```python
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

#Load training dataset
url = "https://raw.githubusercontent.com/callxpert/datasets/master/data-scientist-salaries.cc"
names = ['Years-experience', 'Salary']
dataset = pandas.read_csv(url, names=names)

# shape
print(dataset.shape)

# imprimir las primeras 2 lineas
print(dataset.head(2))

# descripcion estadistica
print(dataset.describe())

#visualize
dataset.plot()
plt.show()

#Training dataset - used to train our model
#Testing dataset - used to test if our model is making accurate predictions
#Since our dataset is small (10 records) we will use 9 records for training the model and 1 record to evaluate the model. copy #paste the below commands to prepare our datasets.

X = dataset[['Years-experience']]
y = dataset['Salary']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)

predictions = model.predict(X_test)
print(accuracy_score(y_test,predictions))

#We are getting 1.0 which is 100% accuracy for our model. Which is the ideal accuracy score. In Production systems, anything #over a 90% is considered a successful model

print(model.predict(6.3))

```

[Source1](https://copycoding.com/your-first-machine-learning-project-in-python-with-step-by-step-instructions/).

[Source2](https://copycoding.com/your-second-machine-learning-project-with-this-famous-iris-dataset-in-python-part-5-of-9-/).

[Source3](https://copycoding.com/machine-learning-project-in-python-to-predict-loan-approval-prediction-part-6-of-6-/).




Logistic Regression solves the limitation of Linear Regression in which the outcome variable (y) must be continuous.

Logistic Regression is a Regression technique that is used when we have a categorical outcome (2 or more categories). This technique can be used to analyze and predict variables that are ‘Discrete’, ‘Nominal’ and ‘Ordered’. Logistic Regression is one of the most easily interpretable classification techniques in a Data Scientist’s portfolio. 

Logistic Regression does not make any assumptions of Normality, Linearity and Homogenity of Variance. This is one of the reasons that Logistic Regression could be more powerful as these assumptions are rarely or if ever satisfied in the real world.


![Logistic Regression 1](https://user-images.githubusercontent.com/17385297/58755931-fa9f3780-84bb-11e9-8d12-2a2bb3d0ff9f.PNG)



An easy way to think of the difference between Linear and Logistic Regression is in Linear Regression, a person can predict a student’s test score (continuous target). In Logistic Regression, a person can assign ‘Pass’, ‘Fail’ categories to student’s scores and predict whether a student passed or failed.


### Types of Logistic Regression

1.Binary
2.Multinomial
3.Ordinal

##### Binary Logistic Regression

[Source1](https://www.kdnuggets.com/2019/01/logistic-regression-concise-technical-overview.html)

    
    















