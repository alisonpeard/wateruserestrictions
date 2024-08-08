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
library(stringr) # for title case
library(patchwork)
library(stargazer)
source("myutils.R")

# ----Variables----
INDICATOR <- 'si24'
TYPE <- 't' # types <- c("s", "t", "m", "e", "r")
WRZ <- "united_utilities_grid"
RZ_IDs = list(london=117, united_utilities_grid=122, ruthamford_north=22)
RZ_ID = RZ_IDs[[WRZ]]
SCENARIO <- 'ff'
WRZLABEL <- str_to_title(gsub("_", " ", WRZ))

# setup env
bdir <- '/Users/alison'
wdir <- paste0(bdir, '/Documents/RAPID/correlation-analysis')
data.dir <- paste0(wdir, '/data/results/full_timeseries/240403')
res.dir <- paste0(wdir, '/data/results')

# load and process data
path <- paste0(data.dir, '/', SCENARIO, '/ts_with_levels.csv')
df <- read.csv(paste0(path))
df <- na.omit(df)
df <- df[df$RZ_ID == RZ_ID,]
df$LoS.binary <- as.numeric(df$LoS > 0)
df$n <- lubridate::days_in_month(df$Date)
df <- na.omit(df[, c('LoS', 'LoS.binary', 'RZ_ID', INDICATOR, 'ensemble', 'n', 'Date')])

# add moving average and decomposition terms
# Triangular MA: https://www.tradingview.com/script/MRkaCrh9-Triangular-Moving-Average/
INDICATORS <- c(INDICATOR)
windows <- c(2, 3, 6, 9, 12, 24, 36, 48) # length of MA windows (in months) 
types <- c("s", "t", "m", "e", "r")
for(INDICATOR in INDICATORS){ # decompose
  df <- ensemblewise(decompose.column, df, INDICATOR)
  INDICATORS <- c(INDICATORS, paste0(INDICATOR, '.trend'))
} # decompose
for(INDICATOR in INDICATORS){ # moving average
  for(i in seq_along(windows)){ 
    for(j in seq_along(types)){
      df <- ensemblewise(movavg.column, df, INDICATOR, windows[i], types[j])
      INDICATORS <- c(INDICATORS, paste0(INDICATOR, '.ma.', types[j], windows[i]))
    }
  }
} # moving average
df <- ensemblewise(lag.column, df, 'LoS', 1)
df <- ensemblewise(lag.column, df, 'LoS.binary', 1)
y.ber <- cbind(1 - df$LoS.binary, df$LoS.binary)
y.bin <- cbind(df$n - df$LoS, df$LoS)
df$y.ber <- y.ber
df$y.bin <- y.bin
df <- na.omit(df[, c('y.bin', 'y.ber', INDICATORS, 'Date', 'ensemble', 'n')])

# training subset by ensemble
ENSEMBLE <- paste0(toupper(SCENARIO), '1')
train <- df[df$ensemble <= ENSEMBLE,]
test <- df[df$ensemble > ENSEMBLE,]
n <- nrow(train)

# training subset by ensemble
set.seed(2)
K <- 30
rows <- vector("list", K) 
for(run in 1:K){
  print(paste0("Training with seed: ", run))
  SEED <- sample(1:999, 1)
  set.seed(SEED)
  
  # First, fit the Bernoulli and Binomial GLMs (on a subset of regressors)
  regressors <- c(INDICATOR, sapply(windows, function(x){paste0(INDICATOR, '.ma.', TYPE, x)}))
  res <- zabi.glm(train, test, label=INDICATOR, X=regressors)
  rows[[run]] <- res$summary
} # incrementally increase training size
summary <- data.frame(do.call(rbind, rows))
stargazer(summary, type='text')

write.csv(summary, paste0(res.dir, '/cv_glmnet/', WRZ, '/mafits_', '_', INDICATOR, ENSEMBLE, '.csv'))
