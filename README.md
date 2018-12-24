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



### Application

```python
X = A[['Unidad_Negocio_num', 'MERCH_AMT_BSE','week']] #seleccionar columnas del database
X1 = X.fillna(0) #simpre limpiar database
sns.set(style="ticks", color_codes=True)
g = sns.pairplot(X1)
```

