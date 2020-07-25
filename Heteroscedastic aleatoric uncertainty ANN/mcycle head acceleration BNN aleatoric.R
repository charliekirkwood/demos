library(keras)
library(tensorflow)
library(tfprobability)
library(MASS)
library(data.table)

library(dplyr)
library(tidyr)
library(ggplot2)

### Load data, and normalise predictors for ANN

data <- as.data.table(mcycle)

ggplot(data) + geom_point(aes(x = times, y = accel))

meantime <- mean(data$times)
sdtime <- sd(data$times)

datanorm <- copy(data)
datanorm[, times := (times-meantime)/sdtime, ]

ggplot(datanorm) + geom_point(aes(x = times, y = accel))

### Split data into train and test

testsamps <- sample(1:nrow(datanorm), nrow(datanorm)/10)

x_train <- as.matrix(datanorm[!testsamps, times])
y_train <- as.matrix(datanorm[!testsamps, accel])

x_test <- as.matrix(datanorm[testsamps, times])
y_test <- as.matrix(datanorm[testsamps, accel])

### Train neural network

model <- keras_model_sequential() %>%
  layer_dense(units = 8, activation = "relu", input_shape = 1) %>%
  layer_dense(units = 2, activation = "linear") %>%
  layer_distribution_lambda(function(x)
    tfd_normal(loc = x[, 1, drop = FALSE],
               scale = 1e-3 + tf$math$softplus(x[, 2, drop = FALSE])
    )
  )
negloglik <- function(y, model) - (model %>% tfd_log_prob(y))
model %>% compile(
  optimizer = optimizer_adam(lr = 0.01),
  loss = negloglik
)

history <- model %>% fit(
  x_train, y_train,
  batch_size = 32,
  epochs = 100,
  verbose = 2,
  validation_data = list(x_test, y_test),
  callbacks = list(callback_early_stopping(monitor = "val_loss", patience = 100, verbose = 1))
)


yhats <- purrr::map(1:100, function(x) model(tf$constant(x_test)))
means <-
  purrr::map(yhats, purrr::compose(as.matrix, tfd_mean)) %>% abind::abind()
sds <-
  purrr::map(yhats, purrr::compose(as.matrix, tfd_stddev)) %>% abind::abind()

means_gathered <- data.frame(cbind(x_test, means)) %>%
  gather(key = run, value = mean_val,-X1)
sds_gathered <- data.frame(cbind(x_test, sds)) %>%
  gather(key = run, value = sd_val,-X1)

lines <-
  means_gathered %>% inner_join(sds_gathered, by = c("X1", "run"))
mean <- apply(means, 1, mean)

ggplot(data.frame(x = x_test, y = y_test, mean = as.numeric(mean)), aes(x, y)) +
  geom_point() +
  theme(legend.position = "none") +
  geom_line(aes(x = x_test, y = mean), color = "violet", size = 1.5) +
  geom_line(
    data = lines,
    aes(x = X1, y = mean_val, color = run),
    alpha = 0.6,
    size = 0.5
  ) +
  geom_ribbon(
    data = lines,
    aes(
      x = X1,
      ymin = mean_val - 2 * sd_val,
      ymax = mean_val + 2 * sd_val,
      group = run
    ),
    alpha = 0.05,
    fill = "grey",
    inherit.aes = FALSE
  )

### Evaluate performance on held out test data

eval_dat <- data.frame(obs = y_test, preds = predict(model, x_test))
ggplot(eval_dat) + geom_point(aes(x = obs, y = preds)) + geom_abline(intercept = 0, slope = 1) +
  ggtitle(label = paste0("RMSE = ", round(sqrt(mean((eval_dat$obs-eval_dat$preds)^2)), 2), "   R\U00B2 =", round(cor(eval_dat$obs,eval_dat$preds)^2, 2)))

### Predict for full range of input data

predictions <- data.frame(times = seq(min(datanorm$times), max(datanorm$times), length.out =  1000), accel_preds = NA)
preds <- replicate(1000, predict(model, predictions$times), simplify = "matrix")
predictions$accel_preds <- predict(model, predictions$times)

ggplot(datanorm) + geom_point(aes(x = times, y = accel)) + geom_line(data = predictions, aes(x = times, y = accel_preds))
ggsave("mcycle head acceleration ANN demo.png")

