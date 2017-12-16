# Similarity-based classifiers. Nearest neighbor classifiers

> **Similarity-based classifiers** estimate the class label of a test sample based on the similarities between
the test sample and a set of labeled training samples, and the pairwise similarities between the
training samples. 

Similarity functions `ρ(x;y)`
may be asymmetric and fail to satisfy the other mathematical properties required for metrics or inner
products.
A popular approach to similarity-based classification is to treat the given dissimilarities as distances in some Euclidean space.

> **Nearest neighbor classifiers**
take training sample, object *u* that needs to be classified and optionally some parameters for weight funcion as input and outputs label (class) predicted for *u*. Algorithm sorts trainig sample by distance (similarity) to classified object in ascending order (so objects from trainig set with less *i* are closer to *u*). Object label is found as an argument that maximizes the sum of weight functions:

![nn](https://github.com/toxazol/machineLearning/blob/master/img/Screenshot%20from%202017-12-16%2015-30-24.png)

As calculations are delayed until *u* is known nearest neighbor classifier refers to *lazy learning* methods. Varying the weight function different similarity-based classifiers can be obtained.

Some of them will be considered here:

 1. k nearest neighbor algorithm
	 - k weighted nearest neighbor algorithm
 2. Parzen window algorithm
	 - Parzen window algorithm with variable window width
 3. Potential funcion algorithm

Algorithms will be tested on standard **R** *iris* data set (Fisher's Iris data set). To provide convinient graphical representation only sepal width and sepal length features are plotted as best separating the data set.
However algorithms are tested on a full feature set.

Also data compression methods will be observed by the example of STOLP algorithm.
At the end presented algorithms are compared in terms of generalization performance.

## k nearest neighbor algorithm
Parameter k is introduced and weight function looks like this:
`ω(i, u) = [i <= k]`. It means it returns 1 if *i* is less or equal to *k* and 0 otherwise.
In other words *u* gets the label of majority among its k nearest neigbors:

![kNN](https://github.com/toxazol/machineLearning/blob/master/img/Screenshot%20from%202017-12-16%2012-21-22.png?raw=true)

Listing of `metricClassifier` funciton implemented on R. This function generalizes metric classification methods and performs corresponding classification algorithm based on input parameters.

```R
# trainSet - variables, last row is label factor
metricClassifier <- function(trainSet, u, metric = dst1, method = 'knn', k, h, ker = ker1){
  rowsNum <- dim(trainSet)[1]
  varsNum <- dim(trainSet)[2]-1
  labels = levels(trainSet[rowsNum, varsNum+1])
  distances <- data.frame(trainSet[,varsNum+1])
  for(i in 1:rowsNum)
    distances[i, 2] <- metric(trainSet[i, 1:varsNum], u)
  orderedDist <- distances[order(distances[, 2]), ]
  
  if(method == 'knn'){
    while(TRUE){
      kNNeighbors <- orderedDist[1:k-1,1]
      lst <- orderedDist[k,2]
      kNNeighbors <- unlist(list(kNNeighbors, orderedDist[orderedDist$V2==lst,1])) # count neighbours \w same dst as lst
      countTable <- table(kNNeighbors)
      if(length(countTable[countTable == max(countTable)])==1)
        return(c(labels[which.max(countTable)], max(countTable)))
      k = k-1
    }
    
  }
  if(method == 'parzen'){
    if(!is.na(k)) h = orderedDist[k+1, 2]
    labelCount = numeric(length(labels))
    for(i in 1:rowsNum){
      labelCount[orderedDist[i,1]] = labelCount[orderedDist[i,1]] + ker(orderedDist[i, 2]/h)
    }
    return(c(labels[which.max(labelCount)], max(labelCount)))
  }
}
```
Slight modifications are made to the algorithm to improve stability. Instead of k nearest neigbors k closest distances are counted. If numbers of votes for some classes are equal k is decreased.

Parameter ***k*** is found through empirical risk minimization utilizing **LOO** cross-validation. 
> **LOO** - (leave one out) is a procedure of empirical evaluation of classification algorithm quality. Sample is being divided into a test set consisting of a single element and a training set comprising all other elements for every element in this sample. It outputs the sum of errors (wrong classification cases) divided by total number of tests (number of elements in the sample).

Here is **R** implementation:
```R
LOO <- function(classifier, dataSet, paramName, paramRange){
  rows <- dim(dataSet)[1]
  cols <- dim(dataSet)[2]-1
  risks <- vector('numeric', length(paramRange))
  minErr <- rows
  bestParam = 0
  i = 0
  for(param in paramRange){
    err = 0
    for(i in 1:rows){
      classifierArgs = list(dataSet[-i, ], dataSet[i, 1:cols])
      classifierArgs[paramName] = param
      class <- do.call(classifier, classifierArgs)[1]
      if(class != dataSet[i,cols+1])
        err <- err+1
    }
    if(err < minErr){
      minErr = err
      bestParam = param
    }
    risks[i] = err/rows
    i = i+1
    message(paste("current ", paramName, " = ", sep=''), param, " out of ", paramRange[length(paramRange)])
  }
  message(paste("best ", paramName, ": ", sep=''), bestParam)
  message("errors made: ", minErr)
  message("empirical risk: ", minErr/rows)
  plot(paramRange, risks, xlab='k', ylab='LOO', col='red', type='l')
  points(bestParam, risks[bestParam], bg='red', col='red', pch=21)
  text(bestParam, risks[bestParam], paste(paramName,"=",bestParam), pos=3)
  return(bestParam)
}
```

And here is optimal *k* found by LOO:

![LOOkNNplot](https://github.com/toxazol/machineLearning/blob/master/img/LOOkNN.png)![kNNplot](https://github.com/toxazol/machineLearning/blob/master/img/kNN19.png?raw=true)



## k **weighted** nearest neighbor algorithm
![kwNN](https://github.com/toxazol/machineLearning/blob/master/img/Screenshot%20from%202017-12-16%2012-56-36.png?raw=true)

> *ω(i)* is a monotonically decreasing sequence of real-number weights, specifying contribution of *i-th* neighbor in *u* classification.

Such sequence can be obtained as geometric sequence with scale factor *q*. Where optimal *q* is found by minimizing risk with LOO.

> **kwNN + LOO chart**


## Parzen window algorithm
Let' s define *ω(i, u)* as a function of distance rather than neighbor rank.

![parzenw](https://github.com/toxazol/machineLearning/blob/master/img/Screenshot%20from%202017-12-16%2013-27-53.png?raw=true) 

where *K(z)* is  nonincreasing on [0, ∞)  kernel function. Then our metric classifier will look like this:

![parzen](https://github.com/toxazol/machineLearning/blob/master/img/Screenshot%20from%202017-12-16%2013-31-51.png?raw=true)

This is how kernel functions graphs look like:
![kernels](https://github.com/toxazol/machineLearning/blob/master/img/main-qimg-ece54bb2db23a4f823e3fdb6058761e8.png?raw=true)

Here are some of them implemented in *R*:
```R
ker1 <- (function(r) max(0.75*(1-r*r),0)) # epanechnikov
ker2 <- (function(r) max(0,9375*(1-r*r),0)) # quartic
ker3 <- (function(r) max(1-abs(r),0)) # triangle
ker4 <- (function(r) ((2*pi)^(-0.5))*exp(-0.5*r*r)) # gaussian
ker5 <- (function(r) ifelse(abs(r)<=1, 0.5, 0)) # uniform
```

Parameter *h* is called "window width" and is similar to number of neighbors *k* in kNN.
Here optimal *h* is found using LOO:

> **Parzen window + LOO chart**

## Potential function algorithm
It's just a slight modification of parzen window algorithm:

![potential](https://github.com/toxazol/machineLearning/blob/master/img/Screenshot%20from%202017-12-16%2013-40-38.png)

Now window width *h* depends on training object *x*. *γ* represents
importance of each such object.

All these *2l* params can be obtained/adjusted with the following
procedure:

> **potential functions params lookup procedure**

Though this process on a sufficiently big sample can take considerable
amount of time.

> **potential functions params lookup results**

## STOLP algorithm
This algorithm implements data set compression by finding regular (etalon)
objects (stolps) and removing objects which do not or harmfully affect classification from sample.
To explain how this algorithm works idea of *margin* has to be introduced.
> **Margin** of an object (*M(x)*) shows us, how deeply current object lies within its class.
[!margin](https://github.com/toxazol/machineLearning/blob/master/img/Screenshot%from%2017-12-16%15-07-53.png)

Here is **R** implementation of STOLP algorithm, where **LDK:LKJ:GLDKJ:LSJ** is margin:
```R
STOLP <- function(set, threshold, err, metric = dst1, method = 'knn', k, h = 1, ker = ker1){
  rowsNum <- dim(set)[1]
  varsNum <- dim(set)[2]-1
  toDelete = numeric()
  labelsNum = levels(set[rowsNum,varsNum+1])
  maxRes = numeric(labelsNum)
  maxLabel = numeric(labelsNum)
  
  for(i in rowsNum:1){
    res = metricClassifier(set, set[i, 1:varsNum], metric, method, k, h, ker)
    if(res[1] != set[i, varsNum+1]){
      toDelete <- c(toDelete, i)
    }
    else if(res[2] > maxRes[res[1]]){
      maxRes[res[1]] = res[2]
      maxLabel[res[1]] = i
    }
  }
  resSet = set[-toDelete, ]
  return(resSet)
}
```

> **STOLP incorporated in nearest neighbor classification**

## Conclusion

> **charts representing algorithms comparison**