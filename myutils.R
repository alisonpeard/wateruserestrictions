# Option + Command + O to collapse all sections on MacOS
# Alt + O to collapse all on Windows
library(pROC)
LAMBDA <- "lambda.min"
NFOLDS <- 5

# ----Data processing functions----
ensemblewise <- function(fun, df, column, ...) {
  ensembles <- unique(df$ensemble)
  dfs <- vector("list", length(ensembles))
  for (i in seq_along(ensembles)) {
    df.ens <- df[df$ensemble == ensembles[i], ]
    df.ens <- fun(df.ens, column, ...)
    dfs[[i]] <- df.ens
  }
  dfs <- do.call(rbind, dfs)
  return(dfs)
}
decompose.column <- function(df, column='si6'){ # BUG!
  ts <- ts(df[[column]], frequency = 12)
  ts.decomp <- stl(ts, s.window='periodic')$time.series[,2]
  colname <- paste0(column, '.trend')
  df[[colname]] <- as.numeric(ts.decomp)
  return(df)
}
lag.column <- function(df, column, k=1){
  df[[paste0(column, '.lag', k)]] = dplyr::lag(df[[column]], k)
  return(df)
}
movavg.column <- function(df, col, n, type){
  col.ma <- movavg(df[[col]], n, type)
  df[[paste0(col, '.ma.',type,n)]] <- col.ma
  return(df)
}
make.conf.int <- function(df){
  Qlower <- subset(df, select=c('Date', 'lower'))
  Qupper <- subset(df, select=c('Date', 'upper'))
  names(Qlower) <- c('x', 'y'); Qlower <- Qlower[order(Qlower$x),]
  names(Qupper) <- c('x', 'y'); Qupper <- Qupper[order(Qupper$x, decreasing=TRUE),]
  rle.lower <- rle(!is.na(Qlower$y))
  rle.upper <- rle(!is.na(Qupper$y))
  Qlower$group <- as.factor(rep(seq_along(rle.lower$values), rle.lower$lengths))
  Qupper$group <- as.factor(rep(rev(seq_along(rev(rle.upper$values))), rle.upper$lengths)) # this line is tricky
  conf.int <- na.omit(rbind(Qlower, Qupper))
  conf.int <- conf.int[order(conf.int$group),]
  return(conf.int)
}
qZABI <- function(p, n, mu, nu){
  q.bin <- qBI(p, 1, nu)
  q.ber <- qBI(p, n, mu)
  q.zabi <- c(q.bin * q.ber)
  return(q.zabi)
}

# ----Metrics----
bce <- function(y, p){
  y <- c(1 - y, y)
  p <- c(1 - p, p)
  return(round(mean(y * log(p)), 4))
}
brier <- function(y, p){
  bs <- mean((y - p)**2)
  return(round(bs, 4))
}
confusion <- function(y, f){
  # y: truth, f: preds
  tp <- sum(y * f)
  fp <- sum((1 - y) * f)
  fn <- sum(y * (1 - f))
  tn <- sum((1 - y) * (1 - f))
  cm <- matrix(c(tp, fp, fn, tn), nrow=2, ncol=2)
  dimnames(cm) <- list(c("T", "F"), c("T", "F"))
  return(round(cm, 4))
}
accuracy <- function(y, f){cm <- confusion(y,f);round((cm[1,1] + cm[2,2]) / sum(cm), 4)}
precision <- function(y,f){cm <- confusion(y,f);round(cm[1,1] / (cm[1,1] + cm[2,1]), 4)}
recall <- function(y,f){cm <- confusion(y,f);round(cm[1,1] / (cm[1,1] + cm[1,2]), 4)}
f1 <- function(y,f){p <- precision(y,f);r <- recall(y,f);round(2 * p * r / (p + r), 4)}
f2 <- function(y,f){cm <- confusion(y,f);round(cm[1,1] / (cm[1,2] + cm[1,1] + cm[2,1]), 4)}
binary.metrics <- function(y, f, p, label='indicator'){
  score.bce <- bce(y, p)
  score.brier <- brier(y, p)
  score.auroc <- roc(y,p)$auc
  score.precision <- precision(y, f)
  score.recall <- recall(y, f)
  score.f1 <- f1(y, f)
  score.f2 <- f2(y, f)
  
  metric.names <- c('BCE', 'Brier', 'AUROC', 'Precision', 'Recall', 'F1', 'F2')
  metrics <- matrix(nrow=1, ncol=7, dimnames=list(NULL,metric.names))
  metrics[1,] <- c(score.bce, score.brier, score.auroc, score.precision,
                   score.recall, score.f1, score.f2)
  metrics <- as.data.frame(metrics)
  rownames(metrics) <- label
  return(metrics)
}
rmse <- function(y, f){
  rmse <- sqrt(mean((y - f)^2))
  return(round(rmse, 4))
}
cts.metrics <- function(y, f, p, label='indicator'){
  score.rmse <- rmse(y, f)
  
  metric.names <- c('RMSE')
  metrics <- matrix(nrow=1, ncol=1, dimnames=list(NULL,metric.names))
  metrics[1,] <- c(score.rmse)
  metrics <- as.data.frame(metrics)
  rownames(metrics) <- label
  return(metrics)
}

# ----Zero-adjusted Binomial model----
bernoulli.glm <- function(train, test, label, y='y.ber', X=c('si6'), lambda=LAMBDA){
  bd <- 1
  p.hat <- mean(train[[y]])
  
  train <- train[,c("ensemble", "Date", y,X, 'n')]
  test <- test[,c("ensemble", "Date", y,X, 'n')]
  
  # fit model
  model <- cv.glmnet(as.matrix(train[,X]), train[[y]], family='binomial')
  mu <- predict(model, newx=as.matrix(train[,X]), type='response', s=lambda)
  coefs <- as.data.frame(as.matrix(coef(model, s=lambda)))
  coef.names <- c('ber.intercept', lapply(X, function(x) paste0('ber.', x)))
  rownames(coefs) <- coef.names
  colnames(coefs) <- label # USE.INDICATOR
  coefs <- t(coefs)
  
  # fitted values
  mu <- as.numeric(mu)
  train$ber.q50 <- qBI(0.5, bd, mu)
  train$ber.lower <- qBI(0.025, bd, mu)
  train$ber.upper <- qBI(0.975, bd, mu)
  train$ber.p <- mu
  
  # predicted values
  mu <- predict(model, newx=as.matrix(test[,X]), type='response')
  mu <- as.numeric(mu)
  q50 <- qBI(0.5, bd, mu)
  test$ber.q50 <- q50
  test$ber.lower <- qBI(0.025, bd, mu)
  test$ber.upper <- qBI(0.975, bd, mu)
  test$ber.p <- mu
  metrics <- binary.metrics(test$y.ber[,2], q50, mu, label)
  results <- merge(coefs, metrics)

  return(list(fitted=train, predicted=test, summary=results))
}
binomial.glm <- function(train, test, label, y='y.bin', X=c('si6'), lambda=LAMBDA){
  # train on positives only
  train <- train[train[[y]][,2] > 0,]
  test <- test[test[[y]][,2] > 0,]
  
  train <- train[,c("ensemble", "Date", y, X, "n")]
  test <- test[,c("ensemble", "Date", y, X, "n")]
  
  # get MLE for binomial
  MLE <- train[[y]][,2] / train$n
  p.hat <- mean(MLE)
  
  # fit Binomial
  model <- cv.glmnet(as.matrix(train[,X]), train[[y]], family="binomial", nfolds=NFOLDS)
  mu <- predict(model, newx=as.matrix(train[,X]), type='response', s=lambda)
  coefs <- as.data.frame(as.matrix(coef(model, s=LAMBDA)))
  coef.names <- c('bin.intercept', lapply(X, function(x) paste0('bin.', x)))
  rownames(coefs) <- coef.names
  colnames(coefs) <- label
  coefs <- t(coefs)
  
  # fitted values
  bd <- train$n
  mu <- as.numeric(mu)
  train$bin.q50 <- qBI(0.5, bd, mu)
  train$bin.lower <- qBI(0.025, bd, mu)
  train$bin.upper <- qBI(0.975, bd, mu)
  train$bin.p <- mu
  
  # predicted values
  bd <- test$n
  mu <- predict(model, newx=as.matrix(test[,X]), type='response')
  mu <- as.numeric(mu)
  q50 <- qBI(0.5, bd, mu)
  test$bin.q50 <- q50
  test$bin.lower <- qBI(0.025, bd, mu)
  test$bin.upper <- qBI(0.975, bd, mu)
  test$bin.p <- mu
  
  # test set metrics
  metrics <- cts.metrics(test$y.bin[,2], q50, mu, label)
  results <- merge(coefs, metrics)
  
  return(list(fitted=train, predicted=test, summary=results))
}
zabi.glm <- function (train, test, label, X=c('si6'), lambda=LAMBDA){
  ber <- bernoulli.glm(train, test, label='si6', X=regressors)
  bin <- binomial.glm(train, test, label='si6', X=regressors)
  results <- cbind(ber$summary, bin$summary)
  
  # fitted values
  train$ber.p <- ber$fitted$ber.p
  train$bin.p <- merge(train, bin$fitted[,c('bin.p', 'bin.q50')], by=0, all.x=TRUE)$bin.q50
  train$bin.p[is.na(train$bin.p)] <- 0
  
  bd <- train$n
  mu.ber <- ber$fitted$ber.p
  mu.bin <- bin$fitted$bin.p
  q50 <- qZABI(0.5, bd, mu.bin, mu.ber)
  lower <- qZABI(0.4, bd, mu.bin, mu.ber)
  upper <- qZABI(0.6, bd, mu.bin,  mu.ber)
  train$q50 <- q50
  train$lower <- lower
  train$upper <- upper
  
  # predicted values
  test$ber.p <- ber$predicted$ber.p
  test$bin.p <- merge(test, bin$predicted[,c('bin.p', 'bin.q50')], by=0, all.x=TRUE)$bin.p
  test$bin.p[is.na(test$bin.p)] <- 0
  
  bd <- test$n
  mu.ber <- test$ber.p
  mu.bin <- test$bin.p
  print(min(mu.bin))
  print(max(mu.bin))
  q50 <- qZABI(0.5, bd, mu.bin, mu.ber)
  lower <- qZABI(0.4, bd, mu.bin, mu.ber)
  upper <- qZABI(0.6, bd, mu.bin,  mu.ber)
  test$q50 <- q50
  test$lower <- lower
  test$upper <- upper
  
  return(list(fitted=train, predicted=test, summary=results))
}
      
