library(keras)
library(tensorflow)
library(MASS)
library(ggplot2)
library(data.table)

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

### Construct neural network

# scale the KL divergence by number of training examples
n <- nrow(x_train) %>% tf$cast(tf$float32)
kl_div <- function(q, p, unused)
  tfd_kl_divergence(q, p) / n

model <- keras_model_sequential()
model %>%
  layer_dense_flipout(units = 64, activation = 'relu', input_shape = 1, 
                      kernel_divergence_fn = kl_div) %>%
  layer_dense_flipout(units = 32, activation = 'relu', 
                      kernel_divergence_fn = kl_div) %>%
  layer_dense(units = 1, activation = 'linear')

nll <- function(y, model) - (model %>% tfd_log_prob(y))

# the KL part of the loss
kl_part <-  function(y_true, y_pred) {
  kl <- tf$reduce_sum(model$losses)
  kl
}

# the NLL part
nll_part <- function(y_true, y_pred) {
  cat_dist <- tfd_one_hot_categorical(logits = y_pred)
  nll <- - (cat_dist %>% tfd_log_prob(y_true) %>% tf$reduce_mean())
  nll
}

model %>% compile(
  loss = nll,
  optimizer = optimizer_adam(lr = 0.01),
  metrics = c("accuracy"),  
  custom_metric("kl_part", kl_part),
  custom_metric("nll_part", nll_part)),
experimental_run_tf_function = FALSE
)

### Train neural network

history <- model %>% fit(
  x_train, y_train,
  batch_size = 32,
  epochs = 100,
  verbose = 2,
  validation_data = list(x_test, y_test),
  callbacks = list(callback_early_stopping(monitor = "val_loss", patience = 100, verbose = 1),
                   callback_model_checkpoint(paste0(getwd(),"/mcycle_two_layer_bayesian_mlp_relu.hdf5"), save_best_only = TRUE))
)

### Load best model

model <- load_model_hdf5(paste0(getwd(),"/mcycle_two_layer_bayesian_mlp_relu.hdf5"))

### Evaluate performance on held out test data

eval_dat <- data.frame(obs = y_test, preds = predict(model, x_test))
ggplot(eval_dat) + geom_point(aes(x = obs, y = preds)) + geom_abline(intercept = 0, slope = 1) +
  ggtitle(label = paste0("RMSE = ", round(sqrt(mean((eval_dat$obs-eval_dat$preds)^2)), 2), "   R\U00B2 =", round(cor(eval_dat$obs,eval_dat$preds)^2, 2)))

### Predict for full range of input data

predictions <- data.frame(times = seq(min(datanorm$times), max(datanorm$times), length.out =  1000), accel_preds = NA)
predictions$accel_preds <- predict(model, predictions$times)

ggplot(datanorm) + geom_point(aes(x = times, y = accel)) + geom_line(data = predictions, aes(x = times, y = accel_preds))
ggsave("mcycle head acceleration ANN demo.png")

