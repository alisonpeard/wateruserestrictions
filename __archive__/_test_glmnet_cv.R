# understanding how CV is used in glmnet
library(glmnet)

#cv.glmnet
#glmnet:::cv.glmnet.raw

nfolds = 1000
N = 313

foldid = sample(rep(seq(nfolds), length = N))
folds <- seq(nfolds)
counts <- list()
for(i in folds){
  which = foldid == i
  counts <- c(counts, sum(which)) # number of samples selected
}
plot(folds, counts, type='l')



