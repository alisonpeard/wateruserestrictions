"
---- Notes ----
This code repeats the training procedure 100 times for different
random seeds to produce an ensemble of prediction metrics. The model trains
on ensemble member FF1, then predicts on the remaining 99 ensemble members and
records the binary and continuous prediction metrics. Uses package cv.glmnet
to select regularisation (LASSO) parameter lambda using k=NFOLDS cross-validation.
Results spread is sensitive to NFOLDs.

GMLNet documentation: https://glmnet.stanford.edu/articles/glmnet.html
alpha=1 by default for LASSO regression.
Using family=binomial() instead of family='binomial' uses IRLS with step size
halving.
"
rm(list=ls())
library(dplyr)
library(glarma)
library(gamlss)
library(glmnet)
library(lubridate)
library(ggplot2)
library(pROC)
library(pracma)
library(microbenchmark)
library(doMC)
registerDoMC(cores=detectCores())

bdir <- '/Users/alison'
wdir <- paste0(bdir, '/Documents/RAPID/correlation-analysis')
data.dir <- paste0(wdir, '/data/results/full_timeseries/240403')
res.dir <- paste0(wdir, '/data/results/')

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
decompose.column <- function(df, column){
  ts <- ts(df[[column]], frequency = 12)
  ts.decomp <- stl(ts, s.window='periodic')$time.series
  colnames(ts.decomp) <- c(paste0(column, '.seasonal'), paste0(column, '.trend'), paste0(column, '.remainder'))
  df <- cbind(df, data.frame(ts.decomp))
  return(df)
}
lag.column <- function(df, column, k=1){
  df[[paste0(column, '.lag', k)]] = dplyr::lag(df[[column]], k)
  return(df)
}
movavg.column <- function(df, col, n, type){
  col.ma <- movavg(df[[col]], n, type)
  df$ma <- col.ma
  return(df)
}
qZABI <- function(p, n, mu, nu){
  q.bin <- qBI(p, 1, nu)
  q.ber <- qBI(p, n, mu)
  q.zabi <- c(q.bin * q.ber)
  return(q.zabi)
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

# load data
WRZ = "london"
RZ_IDs = list(london=117, united_utilities_grid=122, ruthamford_north=22)
RZ_ID = RZ_IDs[[WRZ]]
SCENARIO <- 'ff'
MAXLAG <- 24
path <- paste0(data.dir, '/', SCENARIO, '/ts_with_levels.csv')
df.all <- read.csv(paste0(path))
df.all <- na.omit(df.all)
df.all <- df.all[df.all$RZ_ID == RZ_ID,]

# training subset by ensemble
LASSO.BINOMIAL <- TRUE
verbose <- TRUE # display all metrics
NFOLDS <- 20 # change to 10 later to reduce std
#UPPER.COEF <- 0 # don't allow positive relationship between rain and WUR
set.seed(2)
glmnet.control(mxitnr=100)
for(run in seq(1, 99, 10)){
  print(paste0("Training on ensemble member: ", toupper(SCENARIO), run))
  SEED <- sample(1:999, 1)
  set.seed(SEED)
  
  ENSEMBLE <- paste0(toupper(SCENARIO), run)
  ENSEMBLES <- sapply(c(1:run), function(x) paste0(toupper(SCENARIO), x))
  
  df <- df.all[df.all$ensemble %in% ENSEMBLES,]
  df.test <- df.all[!df.all$ensemble %in% ENSEMBLES,]
  df$n <- lubridate::days_in_month(df$Date)
  df.test$n <- lubridate::days_in_month(df.test$Date)
  n <- nrow(df)
  print(paste0("Training data has ", n, " observations."))
  
  results.all.inds <- data.frame()
  all.indicators <- c('si6', 'si12', 'si24') # 'ep_total', 'anomaly_q50', 
  for(i in 1:length(all.indicators)){
    USE.INDICATOR <- all.indicators[i]
    print(paste0("Fitting ", USE.INDICATOR))
    
    if(TRUE){
      INDICATORS <- c(USE.INDICATOR)
      df.model <- na.omit(df[, c('LoS', 'RZ_ID', INDICATORS, 'ensemble')])
      df.model.test <- na.omit(df.test[, c('LoS', 'RZ_ID', INDICATORS, 'ensemble')])
      df.model$LoS.binary <- as.numeric(df.model$LoS > 0)
      df.model.test$LoS.binary <- as.numeric(df.model.test$LoS > 0)
      p.hat <- mean(df.model$LoS.binary)
      
      # decompose and lag variables
      for(INDICATOR in INDICATORS){
        df.model <- ensemblewise(decompose.column, df.model, INDICATOR)
        df.model.test <- ensemblewise(decompose.column, df.model.test, INDICATOR)
        INDICATORS <- c(INDICATORS, paste0(INDICATOR, '.trend'))
        INDICATORS <- c(INDICATORS, paste0(INDICATOR, '.seasonal'))
        INDICATORS <- c(INDICATORS, paste0(INDICATOR, '.remainder'))
      } # decompose
      for(INDICATOR in INDICATORS){
        for(k in 1:MAXLAG){
          df.model <- ensemblewise(lag.column, df.model, INDICATOR, k)
          df.model.test <- ensemblewise(lag.column, df.model.test, INDICATOR, k)
          INDICATORS <- c(INDICATORS, paste0(INDICATOR, '.lag', k))
        }
      } # lag
      df.model <- ensemblewise(lag.column, df.model, 'LoS', 1)
      df.model.test <- ensemblewise(lag.column, df.model.test, 'LoS', 1)
      df.model <- ensemblewise(movavg.column, df.model, USE.INDICATOR, (12 * 4), "s")
      df.model.test <- ensemblewise(movavg.column, df.model.test, USE.INDICATOR, (12*4), "s")
      INDICATORS <- c(INDICATORS)
      
      y <- cbind(1 - df.model$LoS.binary, df.model$LoS.binary)
      y.test <- cbind(1 - df.model.test$LoS.binary, df.model.test$LoS.binary)
      df.model$y.ber <- y
      df.model.test$y.ber <- y.test
      df.sub <- na.omit(df.model[, c('y.ber', 'ma', INDICATORS)])
      df.sub.test <- na.omit(df.model.test[, c('y.ber', 'ma', INDICATORS)])
      bernoulli.later <- df.sub.test # compare later
      if(nrow(df.sub) < 100){
        warning(paste0('Not enough data to fit ', ENSEMBLE, ' for ', USE.INDICATOR, '. Skipping...'))
      }
      
      index <- rownames(df.sub)
      date <- as.Date(df.all[index,]$Date)
      
      INDICATORS <- c(paste0(USE.INDICATOR,'.trend'))
      regressors <- c(INDICATORS, paste(INDICATORS, '.lag', c(1, 2, 3, 4, 5, 6), sep=""))
      regressors <- c(regressors, 'ma')
      
      # fit with LASSO
      LAMBDA <- 'lambda.min'
      try(ber.mod <- cv.glmnet(as.matrix(df.sub[,regressors]), df.sub$y.ber, family=binomial(), nfolds=NFOLDS))
      if(!exists("ber.mod")){
        # skip modelling this indicator/ensemble member combination
        next
      }
      mu.ber <- predict(ber.mod, newx=as.matrix(df.sub[,regressors]), type='response', s=LAMBDA)
      coefs <- as.data.frame(as.matrix(coef(ber.mod, s=LAMBDA)))
      coef.names <- c('ber.intercept', 'ber.l0', 'ber.l1', 'ber.l2', 'ber.l3', 'ber.l4', 'ber.l5', 'ber.l6', 'ber.ma')
      rownames(coefs) <- coef.names
      colnames(coefs) <- USE.INDICATOR
      coefs <- t(coefs)
      
      bd <- 1
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
        score.f2 <- round(score.cm[1,1] / (score.cm[1,1]+score.cm[2,1]+score.cm[1,2]), 4)
        
        # ROC curve
        roc.curve <- roc(y.true, p[,2])
        par(mfrow=c(1,1));plot(roc.curve)
        score.auc.roc <- round(roc.curve$auc, 4)
        
        metric.names <- c('BCE', 'Brier', 'AUROC', 'Precision', 'Recall', 'F1', 'F2')
        metrics <- matrix(nrow=1, ncol=7, dimnames=list(NULL,metric.names))
        metrics[1,] <- c(score.bce, score.brier, score.auc.roc, score.precision, score.recall, score.f1, score.f2)
        metrics <- as.data.frame(metrics)
        rownames(metrics) <- c(USE.INDICATOR)
        if(verbose){
          print(metrics)
        }
      } # Evaluation metrics
      results <- merge(coefs, metrics)
      results
    } # Bernoulli model for zeros
    if(TRUE){
      INDICATORS <- c(USE.INDICATOR)
      df.model <- na.omit(df[, c('LoS', INDICATORS, 'n', 'ensemble')])
      df.model.test <- na.omit(df.test[, c('LoS', INDICATORS, 'n', 'ensemble')])
      
      # calculate lags and decomposition BEFORE removing zeros
      for(INDICATOR in INDICATORS){
        df.model <- ensemblewise(decompose.column, df.model, INDICATOR)
        df.model.test <- ensemblewise(decompose.column, df.model.test, INDICATOR)
        INDICATORS <- c(INDICATORS, paste0(INDICATOR, '.trend'))
        INDICATORS <- c(INDICATORS, paste0(INDICATOR, '.seasonal'))
        INDICATORS <- c(INDICATORS, paste0(INDICATOR, '.remainder'))
      }
      for(INDICATOR in INDICATORS){
        for(k in 1:24){
          df.model <- ensemblewise(lag.column, df.model, INDICATOR, k)
          df.model.test <- ensemblewise(lag.column, df.model.test, INDICATOR, k)
          INDICATORS <- c(INDICATORS, paste0(INDICATOR, '.lag', k))
        }
      }
      df.model <- ensemblewise(lag.column, df.model, 'LoS', 1)
      df.model.test <- ensemblewise(lag.column, df.model.test, 'LoS', 1)
      df.model <- ensemblewise(movavg.column, df.model, USE.INDICATOR, (12*4), "s")
      df.model.test <- ensemblewise(movavg.column, df.model.test, USE.INDICATOR, (12*4), "s")
      
      # binomial MLE
      df.model$MLE <- df.model$LoS / df.model$n
      df.model <- df.model[df.model$LoS > 0,] # take only positives to avoid ZIBI
      df.model.test <- df.model.test[df.model.test$LoS > 0,]
      p.hat <- mean(df.model$MLE) # MLE for binomial
      
      # need a matrix response for this
      par(mfrow=c(2, 1))
      plot(df.model$LoS, type="l", ylab='LoS days', xlab='Month')
      hist(df.model$LoS, xlab="Days", main='No LoS days per month')
      
      y <- cbind(df.model$n - df.model$LoS, df.model$LoS)
      y.test <- cbind(df.model.test$n - df.model.test$LoS, df.model.test$LoS)
      df.model$y <- y
      df.model.test$y <- y.test
      df.sub <- na.omit(df.model[, c('y', INDICATORS, 'ma', 'n')])
      df.sub.test <- na.omit(df.model.test[, c('y', INDICATORS, 'ma', 'n')])
      
      INDICATORS <- c(paste0(USE.INDICATOR, '.trend'))
      cols <- c(INDICATORS, paste(INDICATORS, '.lag', c(1, 2, 3, 4, 5, 6), sep=""))
      regressors <- c(cols, 'ma')
      
      # fitting with LASSO
      LAMBDA <- 'lambda.min'
      try(bin.mod <- cv.glmnet(as.matrix(df.sub[,regressors]), df.sub$y,
                               family=binomial(), nfolds=NFOLDS, parallel=TRUE))
      if(!exists("bin.mod")){
        warning(paste0("Error. Skipping modelling ", INDICATOR, " for ", ENSEMBLE, "."))
        next
      }
      #coef(bin.mod, s=LAMBDA)
      mu <- predict(bin.mod, newx=as.matrix(df.sub[,regressors]),
                    type='response', s=LAMBDA)
      mu.bin <- predict(bin.mod, newx=as.matrix(df.sub.test[,regressors]),
                        type='response', s=LAMBDA)
      coefs <- as.data.frame(as.matrix(coef(bin.mod, s=LAMBDA)))
      coef.names <- c('bin.intercept', 'bin.l0', 'bin.l1', 'bin.l2', 'bin.l3',
                      'bin.l4', 'bin.l5', 'bin.l6', 'bin.ma')
      rownames(coefs) <- coef.names
      colnames(coefs) <- USE.INDICATOR
      coefs <- t(coefs)
      bin.regressors <- regressors # need to reuse this in ZIBI predictions
      
      if(TRUE){
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
        score.r2 <- cor(df.sub.test$y[,2], df.sub.test$q50)^2
      } # prediction
      if(verbose){
        print(paste0("RMSE: ", round(score.rmse, 6)))
        print(paste0("R-squared: ", round(score.r2, 6)))
      }
      results <- cbind(results, coefs)
      results['bin.rmse'] <- round(score.rmse, 4)
      results['bin.r2'] <- round(score.r2, 4)
    } # binomial count model for WUR days
    if(TRUE){
      if(TRUE){ # prep data
        INDICATORS <- c(USE.INDICATOR)
        df.model.test <- na.omit(df.test[, c('LoS', 'RZ_ID', INDICATORS, 'n', 'ensemble')])
        df.model.test$LoS.binary <- as.numeric(df.model.test$LoS > 0)
        for(INDICATOR in INDICATORS){
          df.model.test <- ensemblewise(decompose.column, df.model.test, INDICATOR)
          INDICATORS <- c(INDICATORS, paste0(INDICATOR, '.trend'))
          INDICATORS <- c(INDICATORS, paste0(INDICATOR, '.seasonal'))
          INDICATORS <- c(INDICATORS, paste0(INDICATOR, '.remainder'))
        }
        for(INDICATOR in INDICATORS){
          for(k in 1:24){
            df.model.test <- ensemblewise(lag.column, df.model.test, INDICATOR, k)
            INDICATORS <- c(INDICATORS, paste0(INDICATOR, '.lag', k))
            lines(df.model[[paste0('si6.lag', k)]], col='blue')
          }
        }
        y.bin <- cbind(df.model.test$n - df.model.test$LoS, df.model.test$LoS)
        df.bin <- df.model.test[, c('LoS', 'RZ_ID', INDICATORS, 'n', 'ensemble')]
        df.bin$y <- y.bin
        df.bin <- ensemblewise(movavg.column, df.bin, USE.INDICATOR, (12*4), "s") # this is returning NULL?
        df.bin.sub <- na.omit(df.bin[, c('y', 'ma', INDICATORS, 'n')])
      } # prep data
      if(TRUE){
        # already have mu.ber for whole time series so only need to make mu.bin
        mu.bin <- predict(bin.mod, newx=as.matrix(df.bin.sub[,regressors]), type='response', s=LAMBDA)
        
        bd <- df.bin.sub$n
        q50 <- qZABI(0.5, bd, mu.bin, mu.ber)
        lower <- qZABI(0.4, bd, mu.bin, mu.ber)
        upper <- qZABI(0.6, bd, mu.bin,  mu.ber)
        
        index <- rownames(df.bin.sub)
        date <- as.Date(df.all[index,]$Date)
        
        df.bin.sub$q50 <- q50
        df.bin.sub$lower <- lower
        df.bin.sub$upper <- upper
        df.bin.sub$date <- date
      } # predict
      score.rmse <- sqrt(mean((df.bin.sub$y[,2] - df.bin.sub$q50)^2))
      score.r2 <- cor(df.bin.sub$y[,2], df.bin.sub$q50)^2
      
      if(verbose){
        print(paste0("RMSE: ", round(score.rmse, 6)))
        print(paste0("R-squared: ", round(score.r2, 6)))
      }
      results['zibi.rmse'] <- round(score.rmse, 4)
      results['zibi.r2'] <- round(score.r2, 4)
    } # predict ZIBI
    
    results['indicator'] <- USE.INDICATOR
    results.all.inds <- rbind(results.all.inds, results)
  } # fit ZIBI model
  
  # Save final results
  results.all.inds <- format(results.all.inds, digits=4)
  rownames(results.all.inds) <- results.all.inds$indicator
  results.all.inds$indicator <- NULL
  write.csv(results.all.inds, paste0(res.dir, '/cv_glmnet/', WRZ, '/fits_', ENSEMBLE, '__', SEED, '.csv'))
  results.all.inds
  rm(ber.mod, bin.mod) # prevent it being reused if next fit fails
} # incrementally increase training size
