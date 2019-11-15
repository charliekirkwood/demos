library(ggplot2)
library(grf)

N <- 10000

fulldat <- data.frame(x = seq(1, N, 1))
fulldat$y <- rnorm(N, tanh(exp(1:N/N)^2)*500, (exp(sin(1:N/(N/22))) + (exp(1:N/N)^2)/10)^3)^3/1000000
plot(fulldat$x, fulldat$y)

set.seed(123)
hist(fulldat$y)

ggplot(fulldat) + geom_point(aes(x = x, y = y), alpha = 0.05, shape = 16) + theme_bw(8)

# Split data into train & test:
test <- sample(1:nrow(fulldat), nrow(fulldat)/10)
train <- (1:nrow(fulldat))[-which((1:nrow(fulldat)) %in% test)]

traindat <- fulldat[train,]
testdat <- fulldat[test,]

# Train quantile forest:

quantmod <- quantile_forest(X = as.matrix(traindat[,1]), Y = as.matrix(traindat[,2]), quantiles = c(0.1,0.5,0.9),
                            regression.splitting = FALSE, sample.fraction = 0.125, mtry = NULL,
                            num.trees = 1000, min.node.size = 21, honesty = TRUE,
                            alpha = 0.25, imbalance.penalty = 0.5,
                            clusters = NULL, samples_per_cluster = NULL)

# Predict on all data:
preddat <- cbind(fulldat, predict(quantmod, fulldat, quantiles = c(0.05, 0.5, 0.95)))
names(preddat) <- c("x", "y", "lower", "median", "upper")

# Plot points and prediction intervals (outputs of quantile regression):
ggplot(preddat) + geom_point(aes(x = x, y = y), alpha = 0.05, shape = 16) + theme_bw(8) +
  geom_line(aes(x = x, y = upper), col = "orange", alpha = 0.75, size = 0.75) +
  geom_line(aes(x = x, y = median), col = "darkred", alpha = 0.75, size = 0.75) +
  geom_line(aes(x = x, y = lower), col = "blue", alpha = 0.75, size = 0.75) +
  ggtitle("GRF quantile regression (10th and 90th percentiles)")
ggsave("GRF quantile regression example.png", type = "cairo")

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