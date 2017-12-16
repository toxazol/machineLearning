# Similarity-based classifiers. Nearest neighbors classifiers

> **Similarity-based classifiers** estimate the class label of a test sample based on the similarities between
the test sample and a set of labeled training samples, and the pairwise similarities between the
training samples. 

> Similarity functions `ρ(x;y)`
may be asymmetric and fail to satisfy the other mathematical properties required for metrics or inner
products.
> A popular approach to similarity-based classification is to treat the given dissimilarities as distances in some Euclidean space. 


Далее рассматриваются и сравниваются следующие метрические классификаторы реализованные на языке **R**:

 1. Метод k ближайших соседей
	 - Метод взвешенных ближайших соседей
 2. Метод парзеновского окна
	 - Метод парзеновского окна с переменной шириной окна
 2. Метод потенциальных функций

Рассматривается алгоритм отбара эталонных объектов STOLP.
Оптимальные параметры алгоритмов обучения подбираются по критерию скользящего контроля LOO (leave one out). Также кросс-контроль LOO применяется для эмпирического оценивания (generalization ability, generalization performance) алгоритмов.

## k nearest neighbors algorithm
На вход алгоритма подается обучающая выборка, классифицируемый объект ***u*** и параметр ***k***. Алгоритм сортирует обучающую выборку по возрастанию расстояния до ***u*** , т.е. вычисления откладываются до момента, пока не станет известен классифицируемый объект ***u***. Такие алгоритмы относятся к методам ленивого обучения (lazy learning). Объекту присваивается класс, к которому относится большинство ближайших соседей (целесообразно выбирать нечетное k).

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
    kNNeighbors <- orderedDist[1:k,1] #orderedDist[1:k-1,1]
    #lst <- orderedDist[k,2]
    #kNNeighbors <- unlist(list(kNNeighbors, orderedDist[orderedDist$V2==lst,1])) # count neighbours \w same dst as lst
    countTable <- table(kNNeighbors)
    return(c(labels[which.max(countTable)], max(countTable)))
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
Parameter ***k*** is found through empirical risk minimization utilizing **LOO** cross-validation. 
> **LOO** - (leave one out) is a procedure of empirical evaluation of classification algorithm quality. Sample is being divided into a test set consisting of a single element and a training set comprising all other elements for every element in this sample. It outputs the sum of errors (wrong classification cases) divided by total number of tests (number of elements in the sample).

Here is **R** implementation:
```R
LOO <- function(classifier, k, h){
  minErr <- 150
  bestK = 0
  for(k in 1:30){
    err = 0
    for(i in 1:150){
      class <- metricClassifier(irisPetals[-i, ], irisPetals[i, 1:2], k=k)[1]
      if(class != iris[i,5])
        err <- err+1
    }
    if(err < minErr){
      minErr = err
      bestK = k
    }
    message("current k: ", k)
  }
  message("best k: ", bestK)
  message("error: ", minErr)
  return(error/total)
}
```
And here is optimal *k* found by LOO:
> **kNN + LOO chart**
> 
## k **weighted** nearest neighbors algorithm
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
