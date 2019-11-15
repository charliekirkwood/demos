library(keras)
library(data.table)
library(ggplot2)

size <- 1000
demodata <- data.table(data.frame(x = runif(size, 0, 1), y = as.numeric(NA)))

# Step function
demodata[x <= 0.5, y:= 0, ]
demodata[x > 0.5, y:= 1, ]

# Step function with overlap (bimodal)
demodata[x <= 0.4, y:= 0, ]
demodata[x > 0.4 & x < 0.6, y:= rbinom(n = size, prob = 0.5, size = 1)]
demodata[x >= 0.6, y:= 1, ]

plot(demodata)

# Split data into train and test sets

trainset <- sample(1:size, (size/10)*9)
testset <- (1:size)[-trainset]

# Standardise input variables to mean zero standard deviation one

means <- unlist(lapply(demodata[trainset][, -2], function(x) mean(x, na.rm = TRUE)))
means

sds <- unlist(lapply(demodata[trainset][, -2], function(x) sd(x, na.rm = TRUE)))
sds

anndemodata <- copy(demodata)
for(i in names(means)){
  anndemodata[, (i) := (.SD-means[i])/sds[i], .SDcols = i]
}

for (j in seq_len(ncol(anndemodata))){
  set(anndemodata,which(is.na(anndemodata[[j]])),j,0)
}

# Then assign train, test, eval x and y:

x_train <- as.matrix(anndemodata[trainset][, x])
y_train <- as.matrix(anndemodata[trainset][, y])

x_test <- as.matrix(anndemodata[testset][, x])
y_test <- as.matrix(anndemodata[testset][, y])

# Now define neural network architecture, and train:

dropout_1 <- layer_dropout(rate = 0.5)

anninput <- layer_input(shape = c(ncol(x_train)))
annoutput <- anninput %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 1024, use_bias = TRUE, input_shape = c(ncol(x_train))) %>%
  layer_batch_normalization() %>%
  layer_activation("relu") %>%
  dropout_1(training = TRUE) %>%
  layer_dense(units = 512, use_bias = TRUE) %>%
  layer_batch_normalization() %>%
  layer_activation("relu") %>%
  dropout_1(training = TRUE) %>%
  layer_dense(units = 256, use_bias = TRUE) %>%
  layer_batch_normalization() %>%
  layer_activation("tanh") %>%
  dropout_1(training = TRUE) %>%
  layer_dense(units = 1, activation = 'linear')

annmodel <- keras_model(anninput, annoutput) 

annmodel %>% compile(
  loss = 'mean_absolute_error',
  optimizer = optimizer_sgd(lr = 0.01),
  metrics = 'mean_squared_error'
)

history <- annmodel %>% fit(
  x_train, y_train, 
  epochs = 300, batch_size = 256, shuffle = TRUE,
  validation_data = list(x_test, y_test),
  callbacks = list(callback_early_stopping(monitor = "val_loss", patience = 100, verbose = 1),
                   callback_model_checkpoint(paste0(getwd(),"/MC_dropout_ann.rds"), save_best_only = TRUE))
)

annmodel <- load_model_hdf5(paste0(getwd(),"/MC_dropout_ann.rds"))

npreds <- 100

preddata <- copy(demodata)
for(i in 1:npreds){
  preddata[, paste0("pred_", i):= predict(annmodel, as.matrix(anndemodata[, x])), ]
}

ggplot(melt(preddata[, -2], id.vars = "x")) + geom_line(aes(x = x, y = value, group = variable), alpha = 0.01) +
  geom_point(data = demodata, aes(x = x, y = y), col = "red") + theme_bw()



