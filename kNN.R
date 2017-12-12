norm <- (function(x) if(!is.factor(x))(x-min(x))/(max(x)-min(x)) else x)
dst1 <- (function(a,b) dist(rbind(a, b))) # euclidian 
dst2 <- (function(a,b) sum((a-b)^2)) # euclidian squared
ker1 <- (function(r) max(0.75*(1-r*r),0))
ker2 <- (function(r) max(0,9375*(1-r*r),0))
ker3 <- (function(r) max(1-abs(r),0))
ker4 <- (function(r) ((2*pi)^(-0.5))*exp(-0.5*r*r))
ker5 <- (function(r) ifelse(abs(r)<=1, 0.5, 0))

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
# LOO <- function(classifier, k, h){
#   minErr <- 150
#   bestK = 0
#   for(k in 1:30){
#     err = 0
#     for(i in 1:150){
#       class <- metricClassifier(irisPetals[-i, ], irisPetals[i, 1:2], k=k)[1]
#       if(class != iris[i,5])
#         err <- err+1
#     }
#     if(err < minErr){
#       minErr = err
#       bestK = k
#     }
#     message("current k: ", k)
#   }
#   message("best k: ", bestK)
#   message("error: ", minErr)
#   return(error/total)
# }

irisPetals = STOLP(iris[,3:5],0,0, k=5) # = as.data.frame(lapply(iris[,3:5], norm))

colors1 <- c("setosa"="#FF000044", "versicolor"="#00FF0044", "virginica"="#0000FF44")
colors2 <- c("setosa" = "#FF000088", "versicolor" = "#00FF0088", "virginica" = "#0000FF88")
plot(irisPetals[,1:2], pch=21, bg=colors1[irisPetals$Species], col=colors1[irisPetals$Species])
for(i in 1:150)
  points(iris[i,3:4], bg=colors2[iris[i,5]], col=colors2[iris[i,5]])
# xSteps = seq(1,7,0.1) # seq(0,1,0.02) if normalized
# ySteps = seq(0.1,2.5,0.1)
# for(x in xSteps)
#   for(y in ySteps){
#     lbl = metricClassifier(irisPetals, c(x,y), method = 'parzen', k=10, ker=ker4)[1]
#     points(x, y, bg=colors2[lbl], col=colors2[lbl])
#   }


