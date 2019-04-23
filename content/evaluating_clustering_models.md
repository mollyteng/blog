Title: Evaluating Clustering Models
Date: 2019-04-21 10:20
Tags: python
Slug: blog-3

This week I looked into some model evaluation metrics for clustering methods, which I'm going to summarize here. 

There are two situations when we are dealing with clustering model evaluation. The first one is simple, which is when our data has the ground truth attached to them (i.e., we have the true labels for group membership). In such case there are quite a few metrics we can use for model comparison. However, when our data doesn't have true labels, things become a little more complicated, but there is still some way which I'll talk about in the second part.  

# When there are true labels

Given the knowledge of the ground truth class assignment, there are a number of metrics we can compute for clustering evaluation. I'll introduce the ones that are available in scikit learn modules.

## Adjusted Rand index

The Adjusted Rand index is a function that measures the similarity of the two assignments ignoring permutations. It ranges from -1 to 1, with a value of 0 indicating random label assignment. Negative values are bad clustering results and positive values are good. A perfect match will result in a value of 1.0.


```python
from sklearn import metrics
labels_true = [0, 0, 0, 1, 1, 1]
labels_pred = [0, 0, 1, 1, 2, 2]
metrics.adjusted_rand_score(labels_true, labels_pred)
```




    0.24242424242424246



## Mutual Information (MI) based scores

The Mutual Information is also a function that measures the agreement of the two assignments ignoring permutations. A perfect labeling will return a score of 1.0, while bad labeling will return negative scores, and random labeling 0.0.


```python
from sklearn import metrics
labels_true = [0, 0, 0, 1, 1, 1]
labels_pred = [0, 0, 1, 1, 2, 2]

metrics.adjusted_mutual_info_score(labels_true, labels_pred)  
```




    0.2250422831983088



## Homogeneity, completeness and V-measure

Given the knowledge of the ground truth class assignments, we define homogeneity and completeness of clusters based on conditional entropy analysis:
- homogeneity: each cluster contains only members of a single class
- completeness: all members of a given class are assigned to the same cluster

Both measures are bounded between 0.0 and 1.0, the higher the better.


```python
from sklearn import metrics
labels_true = [0, 0, 0, 1, 1, 1]
labels_pred = [0, 0, 1, 1, 2, 2]

metrics.homogeneity_score(labels_true, labels_pred)  
```




    0.6666666666666669




```python
metrics.completeness_score(labels_true, labels_pred)
```




    0.420619835714305



V-measure is the harmonic mean of homogeneity and completeness:


```python
metrics.v_measure_score(labels_true, labels_pred) 
```




    0.5158037429793889



When we got a V-measure that is bad, we can look into the homogeneity and completeness scores to see what type of assignment mistakes there are. 

## Fowlkes-Mallows scores

The Fowlkes-Mallows index is the geometric mean of the pairwise precision and recall. It ranges from 0.0 to 1.0, with a higher value indicating good results.


```python
from sklearn import metrics
labels_true = [0, 0, 0, 1, 1, 1]
labels_pred = [0, 0, 1, 1, 2, 2]

metrics.fowlkes_mallows_score(labels_true, labels_pred)
```




    0.4714045207910317



# When there aren't true labels

When the ground truth labels are not known, evaluations are performed on the clustered data itself. This is also called internal evaluation schemes. The aim is to find sets of clusters that are compact, with a small variance within clusters and big variance between clusters. 

## Silhouette Coefficient

Silhouette Coefficient for a single data point is defined by $$s=\frac{b-a}{max(a,b)}$$ 

- a: the mean distance between the point and all other points in the same class
- b: the mean distance between the point and all other points in the next nearest cluster 

The Silhouette Coefficient for a set of data points is the mean of the Silhouette Coefficient for each point. It is bounded between -1 and +1, with higher scores indicating denser and well-separated clustering. Scores around zero indicate overlapping clusters.


```python
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets
dataset = datasets.load_iris()
X = dataset.data
y = dataset.target

import numpy as np
from sklearn.cluster import KMeans
kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X)
labels = kmeans_model.labels_
metrics.silhouette_score(X, labels, metric='euclidean')
```




    0.5528190123564091



## Calinski-Harabaz Index (aka Variance Ratio Criterion)

The Calinski-Harabaz index is also known as the Variance Ratio Criterion, which is used to evaluate clustering results when the ground truth labels are unknown. A higher value indicates a model with better defined clusters.


```python
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets
dataset = datasets.load_iris()
X = dataset.data
y = dataset.target

import numpy as np
from sklearn.cluster import KMeans
kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X)
labels = kmeans_model.labels_
metrics.calinski_harabaz_score(X, labels)
```




    561.62775662962



## Davies-Bouldin Index

The Davies-Bouldin index is also used when the ground truth labels are unknown. A lower value indicates a model with better separation results. Zero is the lowest possible score, and values closer to zero indicates better partitions.


```python
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
kmeans = KMeans(n_clusters=3, random_state=1).fit(X)
labels = kmeans.labels_
davies_bouldin_score(X, labels)
```




    0.6619715465007511



## The Elbow method

Because a good clustering result will produce clusters that are compact, we can use the within cluster sums of squares (also called the Inertia score) as a measure of model evaluation. Since the more clusters we have, likely the smaller the inertia we will get, there is a trade-off between number of clusters and minimizing inertia score. We want to find a best point where we have reasonable number of clusters and the minimum inertia possible. The Elbow method plots the total inertia against the number of clusters, and chose a number of clusters so that adding another cluster doesn't reduce the total inertia much (i.e., the bend in the plot, the 'elbow').

## Auxiliary supervised task

One more method is to set up a supervised learning algorithm as an auxiliary task to evaluate the performance of the unsupervised clustering algorithm. For example, after the unsupervised clustering algorithm produces several clusters, we can use these clusters as latent variables to feed into a supervised classifier to perform some task that is related to the domain the data comes from. The performance of the supervised method can then be used as a proxy of the performance of the unsupervised learner. A more concrete example looks like this (borrowed from https://stats.stackexchange.com/questions/79028/performance-metrics-to-evaluate-unsupervised-learning):

1. Learn representations of words using an unsupervised learner.
2. Use the learned representations as input for a supervised learner performing some NLP task like parts of speech tagging or named entity recognition.
3. Assess the performance of the unsupervised learner by its ability to improve the performance of the supervised learner compared to a baseline using a standard representation, like binary word presence features, as input.

As a conclusion, there is not and should not be a magic number that can summarize how well an unsupervised learning algorithm is performing without actually interpreting the results. Because "how well a particular unsupervised method performs will largely depend on why one is doing unsupervised learning in the first place".