"Gridsearch different moving average types and window lengths."
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
library(rjson)

WRZ <- "london"
INDICATOR <- 'si12'
TREND <- FALSE
RZ_IDS <- list(london=117, united_utilities_grid=122, ruthamford_north=22)

setwd("/Users/alison/Documents/RAPID/_analysis/wateruserestrictions/scripts/01_analysis")

source("utils.R")

setwd("/Users/alison/Documents/RAPID/_analysis/wateruserestrictions/scripts/01_analysis/../..")

config <- fromJSON(file="config.json")

res.dir <- config$paths$results
data.dir <- paste0(res.dir, '/full_timeseries/')

scenario <- config$config$scenarios[config$config$scenario+1] 
rz_id <- RZ_IDS[[WRZ]]
wrz_label <- str_to_title(gsub("_", " ", WRZ))

# load and process data
path <- paste0(data.dir, scenario, '/ts_with_levels.csv')
df <- read.csv(paste0(path))
df <- na.omit(df)
df <- df[df$RZ_ID == rz_id, ]
df$LoS.binary <- as.numeric(df$LoS > 0)
df$n <- lubridate::days_in_month(df$Date)
df <- na.omit(df[, c('LoS', 'LoS.binary', 'RZ_ID', INDICATOR, 'ensemble', 'n', 'Date')])

# add moving average and decomposition terms
INDICATORS <- c(INDICATOR)
windows <- c(2, 3, 6, 9, 12, 24) # length of MA windows (in months) 
types <- c("s", "t") # , "m", "e", "r") # type of MA: s=simple, t=triangular, m=modified, e=exponential, r=recursive

if(TREND) {
  for(INDICATOR in INDICATORS){ # decompose
    df <- ensemblewise(decompose.column, df, INDICATOR)
    INDICATORS <- c(INDICATORS, paste0(INDICATOR, '.trend'))
  } # decompose
}

for(INDICATOR in INDICATORS) { # moving average
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
ENSEMBLE <- paste0(toupper(scenario), '2')
train <- df[df$ensemble <= ENSEMBLE,]
test <- df[df$ensemble > ENSEMBLE,]
n <- nrow(train)

# Fit the ZABI model
regressors <- c(INDICATOR, sapply(windows, function(x) {paste0(INDICATOR, '.ma.s', x)}))
res <- zabi.glm(train, test, label=INDICATOR, X=regressors)

# ----Results----
if(TRUE){
  look.at <- 'FF1'#ENSEMBLE
  preds <- res$fitted
  preds.subset <- preds[preds$ensemble == look.at,]
  preds.subset$y <- preds.subset$y.bin
  preds.subset$Date <- as.Date(preds.subset$Date)
  conf.int <- make.conf.int(preds.subset)
  
  # make plots
  title <- paste0('ZABI GLM fitted values for ', wrz_label, ' (', look.at, ')')
  p1 <- ggplot(preds.subset) + theme_bw() + 
    geom_polygon(data=conf.int, aes(x=x, y=y, group=group, fill='Q60 - Q40'), alpha=0.5) +
    geom_path(data=conf.int, aes(x=x, y=y, group=group), colour='lightblue', alpha=0.5) +
    geom_line(aes(y=q50, x=Date, col='Q50 predictions'), cex=.1) +
    geom_point(aes(y=y[,2], x=Date, col='Observations'), pch=20) +
    geom_point(aes(y=q50, x=Date, col='Q50 predictions'), pch=1, cex=.8) +
    xlab("Year") + ylab("WUR Days") + 
    ggtitle(title) + 
    scale_color_manual(values=c("Observations"="black", "Q50 predictions"="blue")) + 
    scale_fill_manual(values = c("Q60 - Q40" = "lightblue")) +
    guides(color = guide_legend(title = NULL), fill=guide_legend(title=NULL))
  p2 <- ggplot(preds.subset) + theme_bw() + 
    geom_line(aes(x=Date, y=get(INDICATOR))) +
    xlab("Year") + ylab(toupper(INDICATOR))
  p1 + p2 + plot_layout(nrow=2, heights=c(2,1))
} # plot fit

if(TRUE){
  look.at <- 'FF72'
  preds <- res$predicted
  preds.subset <- preds[preds$ensemble == look.at,]
  preds.subset$y <- preds.subset$y.bin
  preds.subset$Date <- as.Date(preds.subset$Date)
  conf.int <- make.conf.int(preds.subset)
  
  # make plots
  title <- paste0('ZABI GLM predictions for ', wrz_label, ' (', look.at, ')')
  p1 <- ggplot(preds.subset) + theme_bw() + 
    geom_polygon(data=conf.int, aes(x=x, y=y, group=group, fill='Q60 - Q40'), alpha=0.5) +
    geom_path(data=conf.int, aes(x=x, y=y, group=group), colour='lightblue', alpha=0.5) +
    geom_line(aes(y=q50, x=Date, col='Q50 predictions'), cex=.1) +
    geom_point(aes(y=y[,2], x=Date, col='Observations'), pch=20) +
    geom_point(aes(y=q50, x=Date, col='Q50 predictions'), pch=1, cex=.8) +
    xlab("Year") + ylab("WUR Days") + 
    ggtitle(title) + 
    scale_color_manual(values=c("Observations"="black", "Q50 predictions"="blue")) + 
    scale_fill_manual(values = c("Q60 - Q40" = "lightblue")) +
    guides(color = guide_legend(title = NULL), fill=guide_legend(title=NULL))
  p2 <- ggplot(preds.subset) + theme_bw() + 
    geom_line(aes(x=Date, y=get(INDICATOR))) +
    xlab("Year") + ylab(toupper(INDICATOR))
  p1 + p2 + plot_layout(nrow=2, heights=c(2,1))
} # plot predictions

stargazer(res$summary, type='text') # evaluation metrics
