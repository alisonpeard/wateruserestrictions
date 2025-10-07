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
library(stargazer) # to view results
library(rjson) # load config
library(ggthemes) # for plotting

WRZ <- "london"
INDICATORS <- c('si12')
TREND <- FALSE
RZ_IDS <- list(london=117, united_utilities_grid=122, ruthamford_north=22)

set.seed(42) # nb run at same time as fit so it's reprodicible

setwd("/Users/alison/Documents/drought-indicators/analysis/wateruserestrictions/scripts/01_analysis")

source("utils.R")

setwd("/Users/alison/Documents/drought-indicators/analysis/wateruserestrictions/scripts/01_analysis/../..")

config <- fromJSON(file="config.json")

VAR <- config$config$variables[config$config$variable+1]
var <- paste0(VAR, "_total")

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
df <- na.omit(df[, c('LoS', 'LoS.binary', 'RZ_ID', INDICATORS, 'ensemble', 'n', 'Date')])

# add moving average and decomposition terms
windows <- c(3, 6, 12, 24) # length of MA windows (in months) 
types <- c("s") # , "m", "e", "r") # type of MA: s=simple, t=triangular, m=modified, e=exponential, r=recursive

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

# training subset by ensembles 1 & 2
ENSEMBLE <- paste0(toupper(scenario), '2')
train <- df[df$ensemble <= ENSEMBLE,]
test <- df[df$ensemble > ENSEMBLE,]
n <- nrow(train)

# Fit the ZABI model
regressors <- c(INDICATOR, sapply(windows, function(x) {paste0(INDICATOR, '.ma.', types, x)}))
res <- zabi.glm(train, test, label=INDICATOR, X=regressors)

get_top_n_coefs <- function(res, n = 3, type = "ber.") {
  coefs <- names(res)[grepl(type, names(res))]
  coefs <- coefs[!grepl("intercept", coefs)]
  values <- unlist(res[coefs])
  inds <- order(abs(values), decreasing = TRUE)
  top_n <- names(values)[head(inds, n)]
  
  values <- res[top_n]
  keys <- sub(type, '', top_n, fixed = TRUE)
  
  return(list(
    keys = keys,
    values = values
  ))
}

ber.coefs <- get_top_n_coefs(res$summary)
bin.coefs <- get_top_n_coefs(res$summary, type = "bin.")

# ----Results----
theme_custom <- function() {theme(
  text = element_text(size = 12),
  panel.grid = element_blank(),
  panel.grid.major.y = element_line(
    colour = "#e3e1e1", linetype = 3, linewidth = 0.5),
  panel.grid.major.x =element_line(
    colour = "#e3e1e1", linetype = 3, linewidth = 0.5),
  plot.title.position = 'plot',
  legend.position = 'right',
  legend.title = element_blank(),
  axis.line=element_line(color = "black")
)}

convert_keys_to_names <- function(keys) {
  # Convert keys to readable names
  readable_names <- sapply(keys, function(key) {
    if (grepl("\\.ma\\.s\\d+$", key)) {
      # Extract the number after 's'
      months <- gsub(".*\\.ma\\.s(\\d+)$", "\\1", key)
      return(paste0(months, "-month MA"))
    } else if (grepl("^[a-zA-Z0-9]+$", key) && !grepl("\\.ma\\.", key)) {
      # Simple key without MA pattern
      return("Current")
    } else {
      # Fallback for any other patterns
      return(key)
    }
  }, USE.NAMES = FALSE)
  
  return(readable_names)
}

blues <- c("#193976", "#4C84E0", "#A0C4FF")
oranges <- c("#B91C1C", "#F26C4F", "#FDD292")

if(TRUE){
  look.at <- 'FF1'
  preds <- res$fitted
  preds.subset <- preds[preds$ensemble == look.at,]
  preds.subset$y <- preds.subset$y.bin
  preds.subset$Date <- as.Date(preds.subset$Date)
  conf.int <- make.conf.int(preds.subset)
  
  # make plots
  title <- paste0('ZABI GLM fitted values for ', wrz_label, ' (', look.at, ')')
  p1 <- ggplot(preds.subset) + theme_bw() + 
    geom_polygon(data=conf.int, aes(x=x, y=y, group=group, fill='Q10 - Q90'), alpha=0.5) +
    geom_path(data=conf.int, aes(x=x, y=y, group=group), colour="lightblue", alpha=0.5) +
    geom_line(aes(y=q50, x=Date, col='Q50 predictions'), cex=.2) +
    geom_point(aes(y=y[,2], x=Date, col='Observations'), pch=20) +
    geom_point(aes(y=q50, x=Date, col='Q50 predictions'), pch=1, cex=1) +
    xlab("Year") + ylab("WUR Days") + 
    ggtitle(title) + 
    scale_color_manual(values=c("Observations"="black", "Q50 predictions"="blue")) + 
    scale_fill_manual(values = c("Q10 - Q90" = "lightblue")) +
    guides(color = guide_legend(title = NULL), fill=guide_legend(title=NULL))
  
  keys <- ber.coefs$keys
  vals <- c(1,1,1) #ber.coefs$values # or c(1,1,1)
  
  p2 <- ggplot(preds.subset) + theme_bw() + 
    geom_line(aes(x=Date, y=get(keys[1]) * vals[[1]], col=keys[1])) +
    geom_line(aes(x=Date, y=get(keys[2]) * vals[[2]], col=keys[2])) +
    geom_line(aes(x=Date, y=get(keys[3]) * vals[[3]], col=keys[3])) +
    xlab("Year") + ylab("Bernoulli\n(Top 3)") +
    scale_color_manual(
      name = NULL,
      values = setNames(oranges, keys[1:3]),
      labels = convert_keys_to_names(keys)
    ) +
    guides(fill = "none") +
    theme_custom()

  keys <- bin.coefs$keys
  vals <- c(1,1,1) # bin.coefs$values
  
  p3 <- ggplot(preds.subset) + theme_bw() + 
    geom_line(aes(x=Date, y=get(keys[1]) * vals[[1]], col=keys[1])) +
    geom_line(aes(x=Date, y=get(keys[2]) * vals[[2]], col=keys[2])) +
    geom_line(aes(x=Date, y=get(keys[3]) * vals[[3]], col=keys[3])) +
    xlab("Year") + ylab("Binomial\n(Top 3)") +
    scale_color_manual(
      name = NULL,
      values = setNames(oranges, keys[1:3]),
      labels = keys[1:3]
    ) +
    guides(fill = "none") +
    theme_custom()
  
  p1 + p2 + p3 + plot_layout(nrow=3, heights=c(2,0.5,0.5))
} # plot fit

if(TRUE){
  look.at <- 'FF72'
  preds <- res$predicted
  preds.subset <- preds[preds$ensemble == look.at, ]
  preds.subset$y <- preds.subset$y.bin
  preds.subset$Date <- as.Date(preds.subset$Date)
  conf.int <- make.conf.int(preds.subset)
  
  # make plots
  title <- paste0('ZABI GLM predictions for ', wrz_label, ' (', look.at, ')')
  p1 <- ggplot(preds.subset) + theme_bw() + 
    geom_polygon(data=conf.int, aes(x=x, y=y, group=group, fill='Q10 - Q90'), alpha=0.5) +
    geom_path(data=conf.int, aes(x=x, y=y, group=group), colour='lightblue', alpha=0.5) +
    geom_line(aes(y=q50, x=Date, col='Q50 predictions'), cex=.1) +
    geom_point(aes(y=y[,2], x=Date, col='Observations'), pch=20) +
    geom_point(aes(y=q50, x=Date, col='Q50 predictions'), pch=1, cex=.8) +
    ylab("WUR Days") + 
    xlab("") +
    #ggtitle(title) + 
    scale_color_manual(values=c("Observations"="black", "Q50 predictions"="blue")) + 
    scale_fill_manual(values = c("Q10 - Q90" = "lightblue")) +
    guides(color = guide_legend(title = NULL), fill=guide_legend(title=NULL))
  
  vals <- c(1,1,1) #ber.coefs$values # or c(1,1,1)
  
  p2 <- ggplot(preds.subset) + theme_bw() + 
    geom_line(aes(x=Date, y=get(ber.coefs$keys[1]) * vals[[1]], col=ber.coefs$keys[1])) +
    geom_line(aes(x=Date, y=get(ber.coefs$keys[2]) * vals[[2]], col=ber.coefs$keys[2])) +
    geom_line(aes(x=Date, y=get(ber.coefs$keys[3]) * vals[[3]], col=ber.coefs$keys[3])) +
    xlab("") + ylab("Bernoulli\n(Top 3)") +
    scale_color_manual(
      name = NULL,
      values = setNames(oranges[1:3], ber.coefs$keys[1:3]),
      labels = convert_keys_to_names(ber.coefs$keys[1:3]),
      breaks = ber.coefs$keys[1:3]
    ) +
    guides(fill = "none") +
    theme_custom()
  
  p1 + p2 + plot_layout(nrow=2, heights=c(2, 0.55))

  p3 <- ggplot(preds.subset) + theme_bw() + 
    geom_line(aes(x=Date, y=get(bin.coefs$keys[1]) * vals[[1]], col=bin.coefs$keys[1])) +
    geom_line(aes(x=Date, y=get(bin.coefs$keys[2]) * vals[[2]], col=bin.coefs$keys[2])) +
    geom_line(aes(x=Date, y=get(bin.coefs$keys[3]) * vals[[3]], col=bin.coefs$keys[3])) +
    xlab("Year") + ylab("Binomial top 3") +
    scale_color_manual(
      name = NULL,
      values = setNames(oranges, bin.coefs$keys[1:3]),
      labels = convert_keys_to_names(bin.coefs$keys[1:3])
    ) +
    guides(fill = "none") +
    theme_custom()
  
  p1 + p2 + p3 + plot_layout(nrow=3, heights=c(1.5,0.75,0.75))
} # plot predictions

if(TRUE) {
  library(ggplot2)
  
  length.out <- 50
  
  probs <- preds$ber.p
  obs <- preds$y.ber[, 2]
  
  breaks <- quantile(probs, probs = seq(0, 1, length.out = length.out), na.rm = TRUE)
  breaks <- seq(0, 1, length.out = length.out)
  labels <- breaks[1:length(breaks)-1]
  
  df <- data.frame(probs = probs, obs = obs)
  df$bin <- cut(df$probs, breaks, labels = labels, include.lowest = TRUE, right = FALSE)
  
  reliability <- df |> 
    group_by(bin) |> 
    summarise(
      p = mean(probs, na.rm = TRUE),
      y = mean(obs, na.rm = TRUE),
      count = n(),
      se = sqrt(mean(obs) * (1 - mean(obs)) / n())  # Standard error
    ) |>
    filter(count >= 10) |>  # Only keep bins with sufficient data
    arrange(p)

  ggplot(reliability, aes(x = p, y = y)) +
    geom_point(aes(size = count), alpha = 1, pch = 1) +
    geom_errorbar(aes(ymin = y - 1.96*se, ymax = y + 1.96*se), width = 0.02) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
    xlim(0, 1) + ylim(0, 1) +
    labs(x = "Mean Predicted Probability", 
         y = "Mean Observed Frequency",
         #title = "Reliability Diagram",
         size = "Count") +
    theme_minimal() +
    theme_custom()
  
  # Calculate Brier score decomposition
  brier_score <- mean((probs - obs)^2)
  reliability_component <- sum(reliability$count * (reliability$p - reliability$y)^2) / length(probs)
  resolution_component <- sum(reliability$count * (reliability$y - mean(obs))^2) / length(probs)
  
  cat("Brier Score:", round(brier_score, 4), "\n")
  cat("Reliability (lower is better):", round(reliability_component, 4), "\n")
  cat("Resolution (higher is better):", round(resolution_component, 4), "\n")
  
  # Calculate base rate variance (Uncertainty)
  base_rate <- mean(obs)
  uncertainty <- base_rate * (1 - base_rate)
  
  cat("Uncertainty (base rate variance):", round(uncertainty, 4), "\n")
  cat("Resolution:", round(resolution_component, 4), "\n")
  cat("Resolution as % of Uncertainty:", round(100 * resolution_component / uncertainty, 2), "%\n")
} # reliability diagram

stargazer(res$summary, type='text') # evaluation metrics

# CRPS skill score for duration model
if (TRUE) {
  mask <- preds$y.ber[, 2] > 0
  observed_counts <- preds$y.bin[mask, 2]
  total_days <- preds$n[mask]
  predicted_probs <- preds$bin.p[mask]
  
  model_crps <- crpsBI(y = observed_counts, n = total_days, mu = predicted_probs)
  
  # Climatology: predict the mean proportion for every observation
  mean_proportion <- mean(observed_counts / total_days)
  climatology_probs <- rep(mean_proportion, length(observed_counts))
  
  climatology_crps <- crpsBI(y = observed_counts, n = total_days, mu = climatology_probs)
  
  skill_score <- 1 - (model_crps / climatology_crps)
  
  cat("Model CRPS:", round(model_crps, 3), "days\n")
  cat("Climatology CRPS:", round(climatology_crps, 3), "days\n") 
  cat("CRPS Skill Score:", round(100 * skill_score, 1), "%\n")
}

# Brier skill score for occurrence model
if (TRUE) {
  obs <- preds$y.ber[, 2]
  probs <- preds$ber.p
  
  model_brier <- brier(obs, probs)
  
  # Climatology: predict the mean proportion for every observation
  
  mean_proportion <- mean(obs > 0)
  climatology_probs <- rep(mean_proportion, length(obs))
  
  climatology_brier <- brier(obs, climatology_probs)
  
  skill_score <- 1 - (model_brier / climatology_brier)
  
  cat("Model Brier:", round(model_brier, 3), "days\n")
  cat("Climatology Brier:", round(climatology_brier, 3), "days\n") 
  cat("Brier Skill Score:", round(100 * skill_score, 1), "%\n")
}

