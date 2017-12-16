# Метрические классификаторы

> **Метрический классификатор** (similarity-based classifier) — алгоритм классификации, основанный на вычислении оценок сходства между
> объектами. Понятие сходства формализуется путем введения функции
> расстояния `ρ(x;y)` (не обязательно метрики - например, допустимо
> нарушение аксиомы треугольника). 


> Метрические классификаторы опираются на **гипотезу компактности**, которая
> предполагает, что схожие объекты чаще лежат в одном классе, чем в
> разных.

Далее рассматриваются и сравниваются следующие метрические классификаторы реализованные на языке **R**:

 - Метод ближайших соседей
	 - Метод взвешенных ближайших соседей
 - Метод парзеновского окна
	 - Метод парзеновского окна с переменной шириной окна
 - Метод потенциальных функций

Рассматривается алгоритм отбара эталонных объектов STOLP.
Оптимальные параметры алгоритмов обучения подбираются по критерию скользящего контроля LOO (leave one out). Также кросс-контроль LOO применяется для эмпирического оценивания (generalization ability, generalization performance) алгоритмов.

## Метод k ближайших соседей
На вход алгоритма подается обучающая выборка, классифицируемый объект ***u*** и параметр ***k***. Алгоритм сортирует обучающую выборку по возрастанию расстояния до ***u*** , т.е. вычисления откладываются до момента, пока не станет известен классифицируемый объект ***u***. Такие алгоритмы относятся к методам ленивого обучения (lazy learning). Объекту присваивается класс, к которому относится большинство ближайших соседей (целесообразно выбирать нечетное k).

![kNN](https://github.com/toxazol/machineLearning/blob/master/formulas/Screenshot%20from%202017-12-16%2012-21-22.png?raw=true)

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
### k weighted nearest neighbors algorithm
