library(ggplot2)
library(lightgbm)

N <- 10000

fulldat <- data.frame(x = seq(1, N, 1))
fulldat$y <- rnorm(N, tanh(exp(1:N/N)^2)*500, (exp(sin(1:N/(N/22))) + (exp(1:N/N)^2)/10)^3)^3/1000000
plot(fulldat$x, fulldat$y)

set.seed(123)
hist(fulldat$y)

ggplot(fulldat) + geom_point(aes(x = x, y = y), alpha = 0.05, shape = 16) + theme_bw(8)

#LightGBM training:
test <- sample(1:nrow(fulldat), nrow(fulldat)/10)
train <- (1:nrow(fulldat))[-which((1:nrow(fulldat)) %in% test)]

traindat <- fulldat[train,]
testdat <- fulldat[test,]

dtrain <- lgb.Dataset(data = as.matrix(traindat[,1]), label = as.matrix(traindat[,2]))
dtest <- lgb.Dataset(data = as.matrix(testdat[,1]), label = as.matrix(testdat[,2]))

watch <- list(train=dtrain, eval=dtest)

modpars <- list(num_leaves = 6, min_data = 33, feature_fraction = 1, bagging_fraction = 0.67, bagging_freq = 1, max_bin = 255, learning_rate = 0.01)
modpars

# Train upper quantile regression:
time <- Sys.time()
upper <- lgb.train(data = dtrain, params = modpars, nrounds = 5000, objective = "quantile", alpha = 0.90, reg_sqrt = TRUE, eval_metric = "logloss", valids = watch, early_stopping_rounds = 50)
Sys.time() - time

# # Train lower quantile regression:
time <- Sys.time()
lower <- lgb.train(data = dtrain, params = modpars, nrounds = 5000, objective = "quantile", alpha = 0.10, reg_sqrt = TRUE, eval_metric = "logloss", valids = watch, early_stopping_rounds = 50)
Sys.time() - time

# Predict on all data:
preddat <- fulldat
preddat$upper <- predict(upper, as.matrix(preddat$x))
preddat$lower <- predict(lower, as.matrix(preddat$x))

# Plot points and prediction intervals (outputs of quantile regression):
ggplot(preddat) + geom_point(aes(x = x, y = y), alpha = 0.05, shape = 16) + theme_bw(8) +
  geom_line(aes(x = x, y = upper), col = "orange", alpha = 0.75, size = 0.75) +
  geom_line(aes(x = x, y = lower), col = "blue", alpha = 0.75, size = 0.75) +
  ggtitle("LightGBM quantile regression (10th and 90th percentiles)")
ggsave("lighgbm quantile regression example.png", type = "cairo")
  
# Evaluate accuracy of quantile regression on held out test data:
evals <- preddat[test,]

# What proportion of points are within the requested quantiles?
# These should match the specified alpha values in lgb.train
round(length(which(evals$y < evals$upper))/nrow(evals),2)
round(length(which(evals$y < evals$lower))/nrow(evals),2)

# How about on all of the data?
# (Just to check that models have aimed for the right thing)
round(length(which(preddat$y < preddat$upper))/nrow(preddat),2)
round(length(which(preddat$y < preddat$lower))/nrow(preddat),2)