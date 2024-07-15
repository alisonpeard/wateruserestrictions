rm(list=ls())
library(dplyr)
library(glarma)
library(gamlss)
library(glmnet)
library(lubridate)
library(ggplot2)
library(pROC)
library(pracma)
library(patchwork)

bdir <- '/Users/alison'
wdir <- paste0(bdir, '/Documents/RAPID/correlation-analysis')
data.dir <- paste0(wdir, '/data/results/full_timeseries/240403')
res.dir <- paste0(wdir, '/data/results')

decompose.column <- function(df, column){
  ts <- ts(df[[column]], frequency = 12)
  ts.decomp <- stl(ts, s.window='periodic')$time.series
  colnames(ts.decomp) <- c(paste0(column, '.seasonal'), paste0(column, '.trend'), paste0(column, '.remainder'))
  return(data.frame(ts.decomp))
}
lag.column <- function(df, column, k = 1){
  df[[paste0(column, '.lag', k)]] = dplyr::lag(df[[column]], k)
  return(df)
}
cm.binary <- function(x, y){
  tp <- sum(x * y)
  fp <- sum((1 - x) * y)
  fn <- sum(x * (1 - y))
  tn <- sum((1 - x) * (1 - y))
  cm <- matrix(c(tp, fp, fn, tn), nrow=2, ncol=2)
  dimnames(cm) <- list(c("T", "F"), c("T", "F"))
  return(cm)
}
brier.score <- function(p, y){
  bs <- mean((y-p)**2)
  return(bs)
}
make.conf.int <- function(df){
  Qlower <- subset(df, select=c('date', 'lower'))
  Qupper <- subset(df, select=c('date', 'upper'))
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

# load data
SCENARIO <- 'ff'
path <- paste0(data.dir, '/', SCENARIO, '/ts_with_levels.csv')
df.all <- read.csv(paste0(path))
df.all <- na.omit(df.all)
RZ_ID = 117 # London
df.all <- df.all[df.all$RZ_ID == RZ_ID,]
df.all$n <- lubridate::days_in_month(df.all$Date)

# training subset by ensemble
ENSEMBLE <- paste0(toupper(SCENARIO), '1')
df <- df.all[df.all$ensemble == ENSEMBLE,]
df.test <- df.all[df.all$ensemble != ENSEMBLE,]
n <- nrow(df)

USE.INDICATOR <- 'si6'
LASSO.BINOMIAL <- TRUE
NFOLDS <- 5
set.seed(278)
if(TRUE){
  INDICATORS <- c(USE.INDICATOR)
  df.model <- na.omit(df[, c('LoS', 'RZ_ID', INDICATORS)])
  df.model.test <- na.omit(df.test[, c('LoS', 'RZ_ID', INDICATORS)])
  df.model$LoS.binary <- as.numeric(df.model$LoS > 0)
  df.model.test$LoS.binary <- as.numeric(df.model.test$LoS > 0)
  p.hat <- mean(df.model$LoS.binary)
  
  for(INDICATOR in INDICATORS){
    df.model <- cbind(df.model, decompose.column(df.model, INDICATOR))
    df.model.test <- cbind(df.model.test, decompose.column(df.model.test, INDICATOR))
    INDICATORS <- c(INDICATORS, paste0(INDICATOR, '.trend'))
    INDICATORS <- c(INDICATORS, paste0(INDICATOR, '.seasonal'))
    INDICATORS <- c(INDICATORS, paste0(INDICATOR, '.remainder'))
  }
  for(INDICATOR in INDICATORS){
    for(k in 1:24){
      df.model <- lag.column(df.model, INDICATOR, k)
      df.model.test <- lag.column(df.model.test, INDICATOR, k)
      INDICATORS <- c(INDICATORS, paste0(INDICATOR, '.lag', k))
    }
  }
  df.model <- lag.column(df.model, 'LoS', 1)
  df.model.test <- lag.column(df.model.test, 'LoS', 1)
  df.model$ma <- movavg(df.model[[USE.INDICATOR]], (12*4), type="s")
  df.model.test$ma <- movavg(df.model.test[[USE.INDICATOR]], (12*4), type="s")
  INDICATORS <- c(INDICATORS) 
  
  y <- cbind(1 - df.model$LoS.binary, df.model$LoS.binary)
  y.test <- cbind(1 - df.model.test$LoS.binary, df.model.test$LoS.binary)
  df.model$y.ber <- y
  df.model.test$y.ber <- y.test
  df.sub <- na.omit(df.model[, c('y.ber', 'ma', INDICATORS)])
  df.sub.test <- na.omit(df.model.test[, c('y.ber', 'ma', INDICATORS)])
  bernoulli.later <- df.sub.test # compare later
  
  index <- rownames(df.sub)
  date <- as.Date(df.all[index,]$Date)
  
  INDICATORS <- c(paste0(USE.INDICATOR,'.trend'))
  regressors <- c(INDICATORS, paste(INDICATORS, '.lag', c(1, 2, 3, 4, 5, 6), sep=""))
  regressors <- c(regressors, 'ma')
  if(TRUE){
    LAMBDA <- 'lambda.min'
    bm <- cv.glmnet(as.matrix(df.sub[,regressors]), df.sub$y.ber, family='binomial')
    try(ber.mod <- cv.glmnet(as.matrix(df.sub[,regressors]), df.sub$y.ber, family='binomial', nfolds=NFOLDS))
    mu.ber <- predict(ber.mod, newx=as.matrix(df.sub[,regressors]), type='response', s=LAMBDA)
    coefs <- as.data.frame(as.matrix(coef(ber.mod, s=LAMBDA)))
    coef.names <- c('ber.intercept', 'ber.l0', 'ber.l1', 'ber.l2', 'ber.l3', 'ber.l4', 'ber.l5', 'ber.l6', 'ber.ma')
    rownames(coefs) <- coef.names
    colnames(coefs) <- USE.INDICATOR
    coefs <- t(coefs)
  } # fit with LASSO
  if(FALSE){
    regressors <- paste(regressors, collapse=" + ")
    formula <- as.formula(paste0('y.ber ~ ', regressors)); print(formula)
    ber.mod <- gamlss(formula,
                      data = df.sub,
                      mu.start = p.hat,
                      family = "BI",
                      method=RS(20))
    summary(ber.mod)
    mu <- ber.mod$mu.fv # fit
    mu.ber <- predict(ber.mod, newdata=df.sub.test, what="mu", type='response') # predict
  } # old (no LASSO)

  mu <- mu.ber
  bd <- 1
  q50 <- qBI(0.5, bd, mu)
  lower <- qBI(0.025, bd, mu)
  upper <- qBI(0.975, bd, mu)
  
  df.sub$q50 <- q50
  df.sub$lower <- lower
  df.sub$upper <- upper
  df.sub$date <- date
  
  Qlower <- subset(df.sub, select=c('date', 'lower'))
  Qupper <- subset(df.sub, select=c('date', 'upper'))
  names(Qlower) <- c('x', 'y'); Qlower <- Qlower[order(Qlower$x),]
  names(Qupper) <- c('x', 'y'); Qupper <- Qupper[order(Qupper$x, decreasing=TRUE),]
  conf.int <- rbind(Qlower, Qupper)
  
  ggplot(df.sub) + theme_bw() + 
    geom_polygon(data=conf.int, aes(x=x, y=y), fill='lightblue', alpha=0.5) +
    geom_line(aes(y=q50, x=date), col='blue') + 
    geom_point(aes(y=y.ber[,2], x=date), col='black', pch=16) + 
    xlab("Year") + ylab("LoS occurs (yes/no)") + 
    ggtitle('Binary binomial model for restrictions')

  if(TRUE){
    mu.ber <- predict(ber.mod, newx=as.matrix(df.sub.test[,regressors]), type='response')
    index <- rownames(df.sub.test)
    date <- as.Date(df.all[index,]$Date)
    df.sub.test$q50 <- qBI(0.5, bd, mu.ber)
    df.sub.test$lower <- qBI(0.025, bd, mu.ber)
    df.sub.test$upper <- qBI(0.975, bd, mu.ber)
    df.sub.test$date <- date
    
    y.true <- df.sub.test$y.ber[,2]
    y.pred <- df.sub.test$q50
    p <- cbind(1 - mu.ber, mu.ber)
    
    # metrics
    score.bce <-  - round(mean(df.sub.test$y * log(p)), 4) # want at least less than 0.2
    score.brier <- round(brier.score(p[,2], y.true), 4)
    score.cm <- round(cm.binary(y.true, y.pred), 4)
    score.accuracy <- round((score.cm[1,1] + score.cm[2,2]) / sum(score.cm), 4)
    score.precision <- round(score.cm[1,1] / (score.cm[1,1] + score.cm[2,1]), 4)
    score.recall <- round(score.cm[1,1] / (score.cm[1,1] + score.cm[1,2]), 4)
    score.f1 <- round(2 * score.precision * score.recall / (score.precision + score.recall), 4)
    
    # ROC curve
    roc.curve <- roc(y.true, p[,2])
    par(mfrow=c(1,1));plot(roc.curve)
    score.auc.roc <- round(roc.curve$auc, 4)
    
    #metrics <- data.frame(row.names=c(USE.INDICATOR), colnames=c('indicator', 'BCE', 'Brier', 'AUC-ROC', 'Precision', 'Recall', 'F1'))
    metric.names <- c('BCE', 'Brier', 'AUC-ROC', 'Precision', 'Recall', 'F1')
    metrics <- matrix(nrow=1, ncol=6, dimnames=list(NULL,metric.names))
    metrics[1,] <- c(score.bce, score.brier, score.auc.roc, score.precision, score.recall, score.f1)
    metrics <- as.data.frame(metrics)
    rownames(metrics) <- c(USE.INDICATOR)
    print(metrics)
  } # Evaluation metrics
  if(TRUE){
    look.at <- paste0(toupper(SCENARIO), 5)
    indices <- rownames(df.test[df.test$ensemble %in% look.at,])
    df.plotting <- df.sub.test[indices,]
    
    Qlower <- subset(df.plotting, select=c('date', 'lower'))
    Qupper <- subset(df.plotting, select=c('date', 'upper'))
    names(Qlower) <- c('x', 'y'); Qlower <- Qlower[order(Qlower$x),]
    names(Qupper) <- c('x', 'y'); Qupper <- Qupper[order(Qupper$x, decreasing=TRUE),]
    conf.int <- rbind(Qlower, Qupper)
    
    ggplot(df.plotting) + theme_bw() + 
      geom_polygon(data=conf.int, aes(x=x, y=y), fill='lightblue', alpha=0.5) +
      geom_line(aes(y=q50, x=date), col='blue') + 
      geom_point(aes(y=y.ber[,2], x=date), col='black', pch=16) + 
      xlab("Year") + ylab("LoS occurs (yes/no)") + 
      ggtitle(paste0('Binary binomial predictions on test set: ', look.at))
  } # visualise test results for subsample
  results <- merge(coefs, metrics)
  results
} # Bernoulli model for zeros
if(TRUE){
  INDICATORS <- c(USE.INDICATOR)
  df.model <- na.omit(df[, c('LoS', INDICATORS, 'n')])
  df.model.test <- na.omit(df.test[, c('LoS', INDICATORS, 'n')])
  
  # calculate lags and decomposition BEFORE removing zeros
  for(INDICATOR in INDICATORS){
    df.model <- cbind(df.model, decompose.column(df.model, INDICATOR))
    df.model.test <- cbind(df.model.test, decompose.column(df.model.test, INDICATOR))
    INDICATORS <- c(INDICATORS, paste0(INDICATOR, '.trend'))
    INDICATORS <- c(INDICATORS, paste0(INDICATOR, '.seasonal'))
    INDICATORS <- c(INDICATORS, paste0(INDICATOR, '.remainder'))
  }
  for(INDICATOR in INDICATORS){
    for(k in 1:24){
      df.model <- lag.column(df.model, INDICATOR, k)
      df.model.test <- lag.column(df.model.test, INDICATOR, k)
      INDICATORS <- c(INDICATORS, paste0(INDICATOR, '.lag', k))
    }
  }
  df.model <- lag.column(df.model, 'LoS', 1)
  df.model.test <- lag.column(df.model.test, 'LoS', 1)
  df.model$ma <- movavg(df.model[[USE.INDICATOR]], (12*4), type="s")
  df.model.test$ma <- movavg(df.model.test[[USE.INDICATOR]], (12*4), type="s")
  
  # binomial MLE
  df.model$MLE <- df.model$LoS / df.model$n
  df.model <- df.model[df.model$LoS > 0,] # take only positives to avoid ZIBI
  df.model.test <- df.model.test[df.model.test$LoS > 0,]
  p.hat <- mean(df.model$MLE) # MLE for binomial
    
  y <- cbind(df.model$n - df.model$LoS, df.model$LoS)
  y.test <- cbind(df.model.test$n - df.model.test$LoS, df.model.test$LoS)
  df.model$y <- y
  df.model.test$y <- y.test
  df.sub <- na.omit(df.model[, c('y', INDICATORS, 'ma', 'n')])
  df.sub.test <- na.omit(df.model.test[, c('y', INDICATORS, 'ma', 'n')])
  
  INDICATORS <- c(paste0(USE.INDICATOR, '.trend'))
  cols <- c(INDICATORS, paste(INDICATORS, '.lag', c(1, 2, 3, 4, 5, 6), sep=""))
  regressors <- c(cols, 'ma')
  if(LASSO.BINOMIAL){
    LAMBDA <- 'lambda.min'
    try(bin.mod <- cv.glmnet(as.matrix(df.sub[,regressors]), df.sub$y, family='binomial', nfolds=NFOLDS))
    mu <- predict(bin.mod, newx=as.matrix(df.sub[,regressors]), type='response', s=LAMBDA)
    mu.bin <- predict(bin.mod, newx=as.matrix(df.sub.test[,regressors]), type='response', s=LAMBDA)
    coefs <- as.data.frame(as.matrix(coef(bin.mod, s=LAMBDA)))
    coef.names <- c('bin.intercept', 'bin.l0', 'bin.l1', 'bin.l2', 'bin.l3', 'bin.l4', 'bin.l5', 'bin.l6', 'bin.ma')
    rownames(coefs) <- coef.names
    colnames(coefs) <- USE.INDICATOR
    coefs <- t(coefs)
    bin.regressors <- regressors # need to reuse this in ZIBI predictions
  }  # fitting with LASSO
  if(!LASSO.BINOMIAL){
    regressors <- paste(cols, collapse=" + ")
    formula <- as.formula(paste0('y ~ ', regressors)); print(formula)
    bin.mod <- gamlss(data = df.sub,
                      formula = formula,
                      mu.start = p.hat,
                      family = "BI",
                      method=RS(20))
    summary(bin.mod)
    mu <- bin.mod$mu.fv
    mu.bin <- predict(bin.mod, newx=df.sub.test, what="mu", type='response')
  } # fitting without LASSO
  if(TRUE){
    # view fit
    bd <- df.model$n
    q50 <- qBI(0.5, bd, mu)
    lower <- qBI(0.05, bd, mu)
    upper <- qBI(0.95, bd, mu)
    
    # get dates for plotting
    index <- rownames(df.sub)
    date <- as.Date(df.all[index,]$Date)
    all.dates <- data.frame(date=seq(from=min(date), to=max(date), by="month"))
    
    #ggplot
    df.sub$q50 <- q50
    df.sub$lower <- lower
    df.sub$upper <- upper
    df.sub$date <- date
    
    # NAs for missing dates
    df.plot <- data.frame(date = date, q50=q50, lower=lower, upper=upper, y=y[,2])
    df.plot <- merge(all.dates, df.plot, by = "date", all.x = TRUE)
    
    # CI polygons
    par(mfrow=c(3,1))
    Qlower <- subset(df.plot, select=c('date', 'lower'))
    Qupper <- subset(df.plot, select=c('date', 'upper'))
    names(Qlower) <- c('x', 'y'); Qlower <- Qlower[order(Qlower$x),]
    names(Qupper) <- c('x', 'y'); Qupper <- Qupper[order(Qupper$x, decreasing=TRUE),]
    rle.lower <- rle(!is.na(Qlower$y))
    rle.upper <- rle(!is.na(Qupper$y))
    Qlower$group <- as.factor(rep(seq_along(rle.lower$values), rle.lower$lengths))
    Qupper$group <- as.factor(rep(rev(seq_along(rev(rle.upper$values))), rle.upper$lengths)) # this line is tricky
    conf.int <- na.omit(rbind(Qlower, Qupper))
    conf.int <- conf.int[order(conf.int$group),]
    
    ggplot(df.plot) + theme_bw() + 
      geom_polygon(data=conf.int, aes(x=x, y=y, group=group), fill='lightblue', alpha=0.5) +
      geom_path(data=conf.int, aes(x=x, y=y, group=group), col='lightblue', alpha=0.5) +
      geom_line(aes(y=q50, x=date), col='blue') +
      geom_point(aes(y=y, x=date), col='black', pch=16) +
      xlab("Year") + ylab("LoS Days") + 
      ggtitle('Binomial model for positive counts')
  } # view fit
  if(TRUE){
    # predictions
    bd <- df.sub.test$n
    q50 <- qBI(0.5, bd, mu.bin)
    lower <- qBI(0.05, bd, mu.bin)
    upper <- qBI(0.95, bd, mu.bin)
    
    index <- rownames(df.sub.test)
    date <- as.Date(df.all[index,]$Date)
    all.dates <- data.frame(date=seq(from=min(date), to=max(date), by="month"))
    
    df.sub.test$q50 <- q50
    df.sub.test$lower <- lower
    df.sub.test$upper <- upper
    df.sub.test$date <- date
    
    # NAs for missing dates
    df.plot.test <- data.frame(date=date, q50=q50, lower=lower, upper=upper, y=df.sub.test$y[,2])
    df.plot.test <- merge(all.dates, df.plot.test, by = "date", all.x = TRUE)
    
    # CI polygons
    par(mfrow=c(3,1))
    Qlower <- subset(df.plot.test, select=c('date', 'lower'))
    Qupper <- subset(df.plot.test, select=c('date', 'upper'))
    names(Qlower) <- c('x', 'y'); Qlower <- Qlower[order(Qlower$x),]
    names(Qupper) <- c('x', 'y'); Qupper <- Qupper[order(Qupper$x, decreasing=TRUE),]
    rle.lower <- rle(!is.na(Qlower$y))
    rle.upper <- rle(!is.na(Qupper$y))
    Qlower$group <- as.factor(rep(seq_along(rle.lower$values), rle.lower$lengths))
    Qupper$group <- as.factor(rep(rev(seq_along(rev(rle.upper$values))), rle.upper$lengths)) # this line is tricky
    conf.int <- na.omit(rbind(Qlower, Qupper))
    conf.int <- conf.int[order(conf.int$group),]

    ggplot(df.plot.test) + theme_bw() + 
      geom_polygon(data=conf.int, aes(x=x, y=y, group=group), fill='lightblue', alpha=0.5) +
      geom_path(data=conf.int, aes(x=x, y=y, group=group), col='lightblue', alpha=0.5) +
      geom_line(aes(y=q50, x=date), col='blue') +
      geom_point(aes(y=q50, x=date), col='blue', pch=20, cex=.2) +
      geom_point(aes(y=y, x=date), col='black', pch=20) +
      xlab("Year") + ylab("LoS Days") + 
      ggtitle('Binomial predictions on test set')
    
    score.rmse <- sqrt(mean((df.sub.test$y[,2] - df.sub.test$q50)^2))
  } # prediction
  print(paste0("RMSE: ", round(score.rmse, 6)))
  results <- cbind(results, coefs)
  results['bin.rmse'] <- round(score.rmse, 4)
} # binomial count model
if(TRUE){
  if(TRUE){
    INDICATORS <- c(USE.INDICATOR)
    df.model.test <- na.omit(df.all[, c('LoS', 'RZ_ID', INDICATORS, 'n')])
    df.model.test$LoS.binary <- as.numeric(df.model.test$LoS > 0)
    for(INDICATOR in INDICATORS){
      df.model.test <- cbind(df.model.test, decompose.column(df.model.test, INDICATOR))
      INDICATORS <- c(INDICATORS, paste0(INDICATOR, '.trend'))
      INDICATORS <- c(INDICATORS, paste0(INDICATOR, '.seasonal'))
      INDICATORS <- c(INDICATORS, paste0(INDICATOR, '.remainder'))
    }
    for(INDICATOR in INDICATORS){
      for(k in 1:24){
        df.model.test <- lag.column(df.model.test, INDICATOR, k)
        INDICATORS <- c(INDICATORS, paste0(INDICATOR, '.lag', k))
        lines(df.model[[paste0('si6.lag', k)]], col='blue')
      }
    }
    y.bin <- cbind(df.model.test$n - df.model.test$LoS, df.model.test$LoS)
    df.bin <- df.model.test[, c('LoS', 'RZ_ID', INDICATORS, 'n')]
    df.bin$y <- y.bin
    df.bin$ma <- movavg(df.bin[[USE.INDICATOR]], (12*4), type="s")
    df.bin.sub <- na.omit(df.bin[, c('y', 'ma', INDICATORS, 'n')])
  } # prep data
  if(TRUE){
    # already have mu.ber for whole time series so only need to make mu.bin
    if(LASSO.BINOMIAL){
      mu.ber <- predict(ber.mod, newx=as.matrix(df.bin.sub[,regressors]),
                        type='response', s=LAMBDA)
      mu.bin <- predict(bin.mod, newx=as.matrix(df.bin.sub[,regressors]),
                        type='response', s=LAMBDA)
    }else{
      mu.bin <- predict(bin.mod, newdata=df.bin.sub, what="mu", type='response')
    }
    bd <- df.bin.sub$n
    q50 <- qZABI(0.5, bd, mu.bin, mu.ber)
    lower <- qZABI(0.4, bd, mu.bin, mu.ber)
    upper <- qZABI(0.6, bd, mu.bin,  mu.ber)
    
    index <- rownames(df.bin.sub)
    date <- as.Date(df.all[index,]$Date)
    ensemble <- df.all[index,]$ensemble
    
    df.bin.sub$q50 <- q50
    df.bin.sub$lower <- lower
    df.bin.sub$upper <- upper
    df.bin.sub$date <- date
    df.bin.sub$ensemble <- ensemble
    
    # split into train/test again
    df.fit <- df.bin.sub[df.bin.sub$ensemble == ENSEMBLE, ]
    df.preds <- df.bin.sub[df.bin.sub$ensemble != ENSEMBLE, ]
  } # predict on train and test
  if(FALSE){
    # CI polygons
    par(mfrow=c(3,1))
    Qlower <- subset(df.preds, select=c('date', 'lower'))
    Qupper <- subset(df.preds, select=c('date', 'upper'))
    names(Qlower) <- c('x', 'y'); Qlower <- Qlower[order(Qlower$x),]
    names(Qupper) <- c('x', 'y'); Qupper <- Qupper[order(Qupper$x, decreasing=TRUE),]
    rle.lower <- rle(!is.na(Qlower$y))
    rle.upper <- rle(!is.na(Qupper$y))
    Qlower$group <- as.factor(rep(seq_along(rle.lower$values), rle.lower$lengths))
    Qupper$group <- as.factor(rep(rev(seq_along(rev(rle.upper$values))), rle.upper$lengths)) # this line is tricky
    conf.int <- na.omit(rbind(Qlower, Qupper))
    conf.int <- conf.int[order(conf.int$group),]
    
    ggplot(df.preds) + theme_bw() + 
      geom_polygon(data=conf.int, aes(x=x, y=y, group=group), fill='lightblue', alpha=0.5) +
      geom_path(data=conf.int, aes(x=x, y=y, group=group), col='lightblue', alpha=0.5) +
      geom_line(aes(y=q50, x=date), col='blue') +
      geom_point(aes(y=q50, x=date), col='blue', pch=20, cex=.2) +
      geom_point(aes(y=y[,1], x=date), col='black', pch=20) +
      xlab("Year") + ylab("LoS Days") + 
      ggtitle('Binomial predictions on test set')
  } # plot (big!)
  score.rmse <- sqrt(mean((df.preds$y[,2] - df.preds$q50)^2))
  score.r2 <- cor(df.preds$y[,2], df.preds$q50)^2
  print(paste0("RMSE: ", round(score.rmse, 6)))
  print(paste0("R-squared: ", round(score.r2, 6)))
  results['zibi.rmse'] <- round(score.rmse, 4)
  results['zibi.r2'] <- round(score.r2, 4)
} # predict ZABI
results['indicator'] <- USE.INDICATOR
write.csv(results, paste0(res.dir, '/cv_glmnet/', USE.INDICATOR,
                          '_fits', ENSEMBLE, '.csv'))

if(TRUE){
  look.at <- 'FF1'
  df.test.subset <- df.bin.sub[df.bin.sub$ensemble == look.at,]
  conf.int <- make.conf.int(df.test.subset)
  # make plots
  title <- 'ZABI GLM fitted values'
  p1 <- ggplot(df.test.subset) + theme_bw() + 
    geom_polygon(data=conf.int, aes(x=x, y=y, group=group, fill='Q40-Q60'),
                 alpha=0.5) +
    geom_path(data=conf.int, aes(x=x, y=y, group=group), colour='lightblue',
              alpha=0.5) +
    geom_line(aes(y=q50, x=date, col='Q50 predictions'), cex=.1) +
    geom_point(aes(y=y[,2], x=date, col='Observations'), pch=20) +
    geom_point(aes(y=q50, x=date, col='Q50 predictions'), pch=1, cex=.8) +
    xlab("Year") + ylab("WUR Days") + 
    ggtitle(title) + 
    scale_color_manual(values=c("Observations"="black",
                                "Q50 predictions"="blue")) + 
    scale_fill_manual(values = c("Q40-Q60" = "lightblue")) +
    guides(color = guide_legend(title = NULL), fill=guide_legend(title=NULL))
  p2 <- ggplot(df.test.subset) + theme_bw() + 
    geom_line(aes(y=si6, x=date)) +
    xlab("Year") + ylab("SPI 6")
  p1 + p2 + plot_layout(nrow=2, heights=c(2,1))
}# plot fit
if(TRUE){
  look.at <- paste0(toupper(SCENARIO), sample(2:99, 1))
  look.at <- 'FF72'
  df.test.subset <- df.bin.sub[df.bin.sub$ensemble == look.at,]
  conf.int <- make.conf.int(df.test.subset)
  # make plots
  title <- paste0('Zero-adjusted binomial generalised linear model (ZABI GLM) predictions for ', look.at)
  p1 <- ggplot(df.test.subset) + theme_bw() + 
    geom_polygon(data=conf.int, aes(x=x, y=y, group=group, fill='Q40-Q60'), alpha=0.5) +
    geom_path(data=conf.int, aes(x=x, y=y, group=group), colour='lightblue', alpha=0.5) +
    geom_line(aes(y=q50, x=date, col='Q50 predictions'), cex=.1) +
    geom_point(aes(y=y[,2], x=date, col='Observations'), pch=20) +
    geom_point(aes(y=q50, x=date, col='Q50 predictions'), pch=1, cex=.8) +
    xlab("Year") + ylab("WUR Days") + 
    ggtitle(title) + 
    scale_color_manual(values=c("Observations"="black", "Q50 predictions"="blue")) + 
    scale_fill_manual(values = c("Q40-Q60" = "lightblue")) +
    guides(color = guide_legend(title = NULL), fill=guide_legend(title=NULL))
  p2 <- ggplot(df.test.subset) + theme_bw() + 
    geom_line(aes(y=si6, x=date)) +
    xlab("Year") + ylab("SPI 6")
  p1 + p2 + plot_layout(nrow=2, heights=c(2,1))
} # plot predictions


# EXTRA PLOTS
#######
if(FALSE){
  par(mfrow=c(3, 1))
  plot(df.test.subset$date, df.test.subset$y[,2], type='l')
  lines(df.test.subset$date, df.test.subset$q50, col='blue', lty=2)
  legend("topleft", legend=c('LoS', 'p(LoS)'), col=c('black', 'darkred'), lty=c(1, 2))
  plot(df.test.subset$date, df.test.subset$si6, type='l')
  si6.avg <- movavg(df.test.subset$si6, n=(12*4), type='s')
  plot(df.test.subset$date, si6.avg, type='l')
} # plot with moving average
if(FALSE){
  # get metrics
  ytrue <- df.bin.sub$y[,2]
  ypred <- df.bin.sub$q50
  positive.inds <- ytrue > 0
  ytrue.pos <- ytrue[positive.inds]
  ypred.pos <- ypred[positive.inds]
  
  n <- nrow(df.bin.sub)
  score.rmse <- sqrt(mean((ytrue - ypred)^2))
  score.rmse.pos <- sqrt(mean((ytrue.pos - ypred.pos)^2))
  score.mpe <- mean(abs(ytrue.pos - ypred.pos)/abs(ytrue.pos))
  print(paste0("RMSE: ", round(score.rmse, 6)))
  print(paste0("MPE: ", round(score.mpe, 6)))
  
  # compare errors
  par(mfrow=c(1, 1));plot(ytrue, type='l', col='red');lines(ypred, col='blue')
  plot(ytrue - ypred, type='l')
  ccf(ytrue, ypred) # distributed around lag 0?
} # RMSE, MPE, and ccf of errors

