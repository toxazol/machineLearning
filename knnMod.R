kNNClassifier <- function(trainSet, u, metric = dst1, k){
  rowsNum <- dim(trainSet)[1]; varsNum <- dim(trainSet)[2]-1
  labels = levels(trainSet[rowsNum, varsNum+1])
  orderedDist <- sortByDist(trainSet, u, rowsNum, varsNum, metric)
  repeat{
    kNNeighbors <- orderedDist[1:k-1,1]
    lst <- orderedDist[k,2]
    kNNeighbors <- unlist(list(kNNeighbors, orderedDist[orderedDist$V2==lst,1])) # count neighbours \w same dst as lst
    countTable <- table(kNNeighbors)
    if(length(countTable[countTable == max(countTable)])==1){
      maxLabelIdx = which.max(countTable)
      return(c(labels[maxLabelIdx], abs(max(countTable)-max(countTable[-maxLabelIdx]))))
    }
    k = k-1
  }
}

# Slight modifications are made to the algorithm to improve stability. Instead of k nearest neigbors k closest distances are counted. If numbers of votes for some classes are equal k is decreased.