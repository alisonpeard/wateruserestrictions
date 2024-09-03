"
Run (a) or (b).
(a) Cycles through all regressor combinations for London and output summary metrics 
  in ../cv/london/<trend>/<regressor>/<lag mode>/<lag type>/FF2.csv
(b) Cycles throught the chosen regressor combination for all WRZs and outputs the 
  same. 

Inference code repeats the training procedure 30 times for different
random seeds to produce an ensemble of prediction metrics. The model trains
on ensemble members FF1 and FF2 then predicts on the remaining 98 ensemble members and
records the metrics.

LASSO regulatisation using cv.glmnet.
GMLNet documentation: https://glmnet.stanford.edu/articles/glmnet.html
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
library(stringr)    # for title case
library(patchwork)
library(stargazer)
source("fitutils.R")

# setup env
bdir <- '/Users/alison'
wdir <- paste0(bdir, '/Documents/RAPID/correlation-analysis')
data.dir <- paste0(wdir, '/data/results/full_timeseries/240403')
res.dir <- paste0(wdir, '/data/results')

# ----Variables----
SCENARIO <- 'nf'
ENSEMBLE <- paste0(toupper(SCENARIO), '2')

# subset rz_keys by what time series are available
ts.path <- paste0(data.dir, '/', SCENARIO, '/ts_with_levels.csv')
rz_keys = read.csv(paste0(wdir, '/data', '/wrz_key.csv'))
df <- read.csv(paste0(ts.path))
rz_keys <- merge(rz_keys, df, by.y='RZ_ID', by.x='rz_id')
rz_keys <- unique(rz_keys[c('rz_id', 'wrz')]) # 29 unique

# (a) loop through all regressor options for London
RZ_ID <- 117
WRZ <- 'london'
INDs <- c('ep_total', 'si6', 'si12', 'si24')
TREND.MODES <- c('raw', 'trend')
LAG.MODES <- c('lag', 'ma')
TYPES <- c("s", "t")
if(TRUE){
  for(x in 1:length(INDs)){
    for(y in 1:length(TREND.MODES)){
      for(z in 1:length(LAG.MODES)){
        for(w in 1:length(TYPES)){
          # set up vars
          INDICATOR.BASE <- INDs[x]
          INDICATOR <- INDICATOR.BASE
          TREND.MODE <- TREND.MODES[y]
          LAG.MODE <- LAG.MODES[z]
          TYPE = TYPES[w]
          print(paste(x,y,z,w))
          print(paste('Training on:', INDICATOR.BASE, TREND.MODE, LAG.MODE, TYPE))
          
          WRZLABEL <- str_to_title(gsub("_", " ", WRZ))
          print(paste0('Fitting on ',WRZ))
          
          # load and process data
          df <- read.csv(paste0(ts.path))
          df <- df[df$RZ_ID == RZ_ID,]
          df <- na.omit(df)
          df$LoS.binary <- as.numeric(df$LoS > 0)
          df$n <- lubridate::days_in_month(df$Date)
          df <- na.omit(df[, c('LoS', 'LoS.binary', 'RZ_ID', INDICATOR, 'ensemble', 'n', 'Date')])
          
          # add moving average and decomposition terms
          INDICATORS <- c(INDICATOR)
          windows <- c(2, 3, 6, 9, 12, 24) # length of MA windows (in months) 
          types <- c(TYPE)
          if(TREND.MODE == 'trend'){
            for(j in length(INDICATORS)){ # decompose
              df <- ensemblewise(decompose.column, df, INDICATOR)
              INDICATORS[j] <- paste0(INDICATOR, '.trend')
            } # decompose
          }
          if(LAG.MODE == 'ma'){
            for(INDICATOR in INDICATORS){ # moving average
              for(i in seq_along(windows)){ 
                for(j in seq_along(types)){
                  df <- ensemblewise(movavg.column, df, INDICATOR, windows[i], types[j])
                  INDICATORS <- c(INDICATORS, paste0(INDICATOR, '.ma.', types[j], windows[i]))
                }
              }
            }
          }else if(LAG.MODE == 'lag'){
            if(TYPE == 's'){ # prevent doing same thing twice for 't' and 's'
              TYPE <- ''
            }else{
              print("Skipping repeat lag loop")
              next
            }
            for(INDICATOR in INDICATORS){ # pointwise lags
              for(i in seq_along(windows)){ 
                df <- ensemblewise(lag.column, df, INDICATOR, windows[i])
                INDICATORS <- c(INDICATORS, paste0(INDICATOR, '.lag.', windows[i]))
              }
            } # lagged variables
          }else{
            stop(paste0('Invalid LAG.TYPE: ', LAG.TYPE))
          }
          
          # add AR and binary terms
          df <- ensemblewise(lag.column, df, 'LoS', 1)
          df <- ensemblewise(lag.column, df, 'LoS.binary', 1)
          y.ber <- cbind(1 - df$LoS.binary, df$LoS.binary)
          y.bin <- cbind(df$n - df$LoS, df$LoS)
          df$y.ber <- y.ber
          df$y.bin <- y.bin
          df <- na.omit(df[, c('y.bin', 'y.ber', INDICATORS, 'Date', 'ensemble', 'n')])
          
          # training subset by ensemble
          train <- df[df$ensemble <= ENSEMBLE,]
          test <- df[df$ensemble > ENSEMBLE,]
          n <- nrow(train)
          
          # training subset by 30-member ensemble
          set.seed(2)
          K <- 30
          rows <- vector("list", K) 
          for(run in 1:K){
            print(paste0("Training with seed number: ", run))
            SEED <- sample(1:999, 1)
            set.seed(SEED)
            
            # First, fit the Bernoulli and Binomial GLMs (on a subset of regressors)
            regressors <- c(INDICATOR, sapply(windows, function(x){paste0(INDICATOR, '.', LAG.MODE, '.', TYPE, x)}))
            res <- zabi.glm(train, test, label=INDICATOR, X=regressors)
            rows[[run]] <- res$summary
            print(res$summary)
          }
          summary <- data.frame(do.call(rbind, rows))
          stargazer(summary, type='text')
          
          summary$rz_id <- RZ_ID
          outdir <- paste0(res.dir, '/cv/', WRZ, '/', TREND.MODE, '/', INDICATOR.BASE, '/', LAG.MODE, '.', TYPE)
          dir.create(outdir, recursive=TRUE)
          write.csv(summary, paste0(outdir, '/', ENSEMBLE, '.csv'))
          print(paste0("Saved!"))
          print(paste('Trained on:', INDICATOR.BASE, TREND.MODE, LAG.MODE, TYPE))
        }
      }
    }
  }
}

# (b) loop through water resource zones for chosen combination
INDICATOR.BASE <- 'si6'      # c('si6', 'si12', 'si24', 'ep_total')
TREND.MODE <- 'raw'           # c('trend', 'raw'), raw means no decomposition
LAG.MODE <- 'ma'              # c('lag', 'ma')
TYPE <- 's'                   # c("s", "t", "m", "e", "r", "")
K <- 30                       # 30
if(FALSE){
  rz_keys$success <- FALSE
  for(x in 1:nrow(rz_keys)){
    try({
      # i = 1 # (for dev)
      INDICATOR <- INDICATOR.BASE
      RZ_ID <- rz_keys$rz_id[x]
      WRZ <- rz_keys$wrz[x]
      WRZLABEL <- str_to_title(gsub("_", " ", WRZ))
      print(paste0('Fitting on ',WRZ))
      
      # load and process data
      df <- read.csv(paste0(ts.path))
      df <- na.omit(df)
      df <- df[df$RZ_ID == RZ_ID,]
      df$LoS.binary <- as.numeric(df$LoS > 0)
      df$n <- lubridate::days_in_month(df$Date)
      df <- na.omit(df[, c('LoS', 'LoS.binary', 'RZ_ID', INDICATOR, 'ensemble', 'n', 'Date')])
      
      # add moving average and decomposition terms
      INDICATORS <- c(INDICATOR)
      windows <- c(2, 3, 6, 9, 12, 24) # length of MA windows (in months) 
      types <- c("s", "t")
      if(TREND.MODE == 'trend'){
        for(j in length(INDICATORS)){ # decompose
          df <- ensemblewise(decompose.column, df, INDICATOR)
          INDICATORS[j] <- paste0(INDICATOR, '.trend')
        } # decompose
      }
      if(LAG.MODE == 'ma'){
        for(INDICATOR in INDICATORS){ # moving average
          for(i in seq_along(windows)){ 
            for(j in seq_along(types)){
              df <- ensemblewise(movavg.column, df, INDICATOR, windows[i], types[j])
              INDICATORS <- c(INDICATORS, paste0(INDICATOR, '.ma.', types[j], windows[i]))
            }
          }
        }
      }else if(LAG.MODE == 'lag'){
        TYPE <- ''
        for(INDICATOR in INDICATORS){ # pointwise lags
          for(i in seq_along(windows)){ 
            df <- ensemblewise(lag.column, df, INDICATOR, windows[i])
            INDICATORS <- c(INDICATORS, paste0(INDICATOR, '.lag.', windows[i]))
          }
        } # lagged variables
      }else{
        stop(paste0('Invalid LAG.TYPE: ', LAG.TYPE))
      }
      
      # add AR and binary terms
      df <- ensemblewise(lag.column, df, 'LoS', 1)
      df <- ensemblewise(lag.column, df, 'LoS.binary', 1)
      y.ber <- cbind(1 - df$LoS.binary, df$LoS.binary)
      y.bin <- cbind(df$n - df$LoS, df$LoS)
      df$y.ber <- y.ber
      df$y.bin <- y.bin
      df <- na.omit(df[, c('y.bin', 'y.ber', INDICATORS, 'Date', 'ensemble', 'n')])
      
      # training subset by ensemble
      train <- df[df$ensemble <= ENSEMBLE,]
      test <- df[df$ensemble > ENSEMBLE,]
      n <- nrow(train)
      
      # training subset by 30-member ensemble
      set.seed(2)
      rows <- vector("list", K) 
      for(run in 1:K){
        # run = 1 # for dev
        print(paste0("Training with seed number: ", run))
        SEED <- sample(1:999, 1)
        set.seed(SEED)
        
        # First, fit the Bernoulli and Binomial GLMs (on a subset of regressors)
        regressors <- c(INDICATOR, sapply(windows, function(x){paste0(INDICATOR, '.', LAG.MODE, '.', TYPE, x)}))
        res <- zabi.glm(train, test, label=INDICATOR, X=regressors)
        rows[[run]] <- res$summary
        print(res$summary)
      }
      summary <- data.frame(do.call(rbind, rows))
      stargazer(summary, type='text')
      
      summary$rz_id <- RZ_ID
      outdir <- paste0(res.dir, '/cv/', WRZ, '/', TREND.MODE, '/', INDICATOR.BASE, '/', LAG.MODE, '.', TYPE)
      dir.create(outdir, recursive=TRUE)
      write.csv(summary, paste0(outdir, '/', ENSEMBLE, '.csv'))
      rz_keys[x, 'success'] <- TRUE
      print(paste0("Saved!"))
    })
  }#
  print('Finished gridsearch.')
  rz_keys
}












