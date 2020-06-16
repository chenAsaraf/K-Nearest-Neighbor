# K-Nearest-Neighbor
KNN implementation on HC temperature data set

downlowad [HC temperature data set](https://web.archive.org/web/20180422082424/http://www.math.hope.edu/swanson/statlabs/data.html)

###### Implementation details:
1. Sample 65 training points from the set. The remaining points are the test set.
2. For each of k=1,3,5,7,9 and p=1,2,∞, evaluate the k-NN classifier on the test set, under the lp distance.
(The base set of the classifier is the training set.) Compute the classifier error on the test set.
3. Repeat steps (a) and (b) 500 times, and print the average error for each k and p.  
