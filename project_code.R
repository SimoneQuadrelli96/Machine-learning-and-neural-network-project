library(jpeg)
library(imager)
library(class)
library(e1071)
library(FactoMineR)
library(factoextra)
library(keras)
library(tensorflow)

read_fruit_grayscale <- function(path,set,fruit, dim = 50){
  filenames <- list.files(paste(path,set,fruit,sep=""), pattern = "*.jpg")
  images_grayscale = list()
  i = 1
  for (name in  filenames){
    path_name = paste(path,set,fruit,"/",name,sep="")
    img <- grayscale(load.image(path_name))
    inmg <- resize(img,dim,dim)
    images_grayscale[[i]] = as.vector(img)
    i = i+1
  }
  return(images_grayscale)
}


#number of pixels to be used
dim = 50
path = "../fruits-360_dataset/fruits-360/"

# grayscale extraction
set = "Training/"
training_features_matrix <- NULL
training_labels_vector <- NULL

fruits = c ("Apple Golden 1", "Apple Golden 2", "Apple Red 2","Apple Red 1", "Banana", "Apricot", "Blueberry")
i = 0
for (fruit in fruits){
  images_grayscale <- read_fruit_grayscale(path,set,fruit)
  images_grayscale <- matrix(unlist(images_grayscale), ncol = dim*dim, byrow = TRUE)
  training_features_matrix <- rbind(training_features_matrix,images_grayscale)
  labels <-replicate(dim(images_grayscale)[1], i)
  training_labels_vector <- c(training_labels_vector,labels)
  i <- i +1
}


set = "Test/"
test_features_matrix <- NULL
test_labels_vector <- NULL
i = 0
for (fruit in fruits){
  images_grayscale <- read_fruit_grayscale(path,set,fruit)
  images_grayscale <- matrix(unlist(images_grayscale), ncol = dim*dim, byrow = TRUE)
  test_features_matrix <- rbind(test_features_matrix,images_grayscale)
  labels <-replicate(dim(images_grayscale)[1], i)
  test_labels_vector <- c(test_labels_vector,labels)
  i <- i +1
}


#PCA
#i 150 principali componenti sono sufficienti a spiegare il 100% della varianza
# i primi 50 componenti sono sufficienti a spiegare circa  il 95% della varianza
t1 <- Sys.time()
pca <- prcomp(training_features_matrix,center = TRUE, scale=TRUE)
t2 <- Sys.time()
pca_execution_time <- t2 - t1
# pca_execution_time = Time difference of 5.653888 mins
pr_var <- (pca$sdev[1:200])^2
prop_varex <- pr_var/sum(pr_var)
plot(cumsum(prop_varex), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained")


training_features_matrix_pc <- pca$x
test_features_matrix_pc <- predict(pca, newdata = test_features_matrix)

#BASELINE: fitting knn model 
neighbours <- c(1,3,5,7,9,11)
features <- c(3,10,50,100,150)
res_knn <- c()
time_knn <- c()
i <- 1
for (n_f in features){
  for (n in neighbours){
    t1 <- Sys.time()
    knn <-knn(training_features_matrix_pc[,1:n_f], test_features_matrix_pc[,1:n_f], as.factor(training_labels_vector), k=n)
    t2 <- Sys.time()
    time_knn[i] <- as.double(difftime(t2,t1,  units = "secs"))
    res_knn[i] <- sum(knn == test_labels_vector)/length(test_labels_vector)
    i <- i +1
  }
}

time_knn
#[1]  0.3044703  0.3045428  0.3166747  0.3128932  0.3145876  0.3214788  0.7349553  0.7253215  0.7385712  0.7419629  0.7507675
#[12]  0.7547581  5.7262924  5.6277361  5.5811265  5.5949266  5.5787666  5.3013055 15.2888229 14.8315446 15.0276814 14.8071477
#[23] 15.3399296 14.8435581 22.7163107 22.8648741 22.9841282 22.9756999 22.9742455 22.8733926

res_knn 
#<- c(
# 0.7918860, 0.7940789, 0.7980263 ,0.7971491, 0.7907895, 0.7945175 ,0.8703947 ,0.8633772 ,0.8603070 ,0.8589912 ,0.8557018,
# 0.8517544,
#0.8701754 ,0.8662281 ,0.8629386, 0.8638158 ,0.8592105 ,0.8567982, 0.8699561 ,0.8618421 ,0.8605263 ,0.8572368 ,0.8548246,
#0.8530702,
# 0.8686404, 0.8605263 ,0.8600877 ,0.8578947, 0.8539474, 0.8486842)

precision <- matrix(res_knn,ncol=length(features))
matplot(precision, type = c("b"),pch=1,col = 1:length(features),  xlab = "Neighbours", ylab="Accuracy", main="KNN ACCUARCY")
legend("left", legend = features, col=1:length(features), pch=1, title="Features")

precision <- matrix(time_knn,ncol=length(features))
matplot(precision, type = c("b"),pch=1,col = 1:length(features),  xlab = "Neighbours", ylab="Execution time (s)", main="KNN EXECUTION TIME")
legend("left", legend = features, col=1:length(features), pch=1, title="Features")

features <- c(3,10,50,100,150)
res_svm <- c()
time_svm <- c()
i <- 1
for (n_f in features){
  training_features_matrix_svm <- as.data.frame(training_features_matrix_pc[,1:n_f])
  training_features_matrix_svm$target <- as.factor(training_labels_vector)
  test_features_matrix_svm <- as.data.frame(test_features_matrix_pc[,1:n_f])
  
  t1 <- Sys.time()
  SVM <- svm(target ~ ., data = training_features_matrix_svm, kernel = "radial")
  t2 <- Sys.time()
  res <- predict(SVM, newdata = test_features_matrix_svm)
  time_svm[i] <- as.double(difftime(t2,t1,  units = "secs"))
  res_svm[i] <-     sum(res == test_labels_vector)/length(test_labels_vector)
  i <- i +1
}

time_svm
#[1]  9.517319  7.648561 25.033115 58.825680 95.674626
res_svm
#[1] 0.6857456 0.8225877 0.8407895 0.8245614 0.8245614

plot(res_svm, type = c("b"),pch=1 , xlab = "Features", ylab="Precision", main="SVM ACCURACY", xaxt = "n")
axis(1, at=1:length(features), labels=features)
legend("right", legend = features, col=1:length(features), pch=1, title="Features")

plot(time_svm, type = c("b"),pch=1 , xlab = "Features", ylab="Execution time (s)", main="SVM EXECUTION TIME", xaxt = "n")
axis(1, at=1:length(features), labels=features)


# confusion matrix
#table(predict(modSVM), training_labels_vector)

#CONSIDERA IL KERNEL UTILIZZATO
#summary(modSVM)

#CNN KERAS

y_train <- to_categorical(training_labels_vector, length(fruits))
y_test <- to_categorical(test_labels_vector, length(fruits))
x_train<-array_reshape(training_features_matrix,c(nrow(training_features_matrix),dim,dim,1))
x_test<-array_reshape(test_features_matrix,c(nrow(test_features_matrix),dim,dim,1))

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(50, 50, 1)) %>%
  layer_max_pooling_2d(pool_size = c(4,4)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(4, 4)) %>%
  #layer_conv_2d(filters = 128, kernel_size = c(2, 2), activation = "tanh") %>%
  #layer_max_pooling_2d(pool_size = c(4, 4)) %>%
  #layer_conv_2d(filters = 128, kernel_size = c(2, 2), activation = "tanh") %>%
  layer_flatten() %>%
  layer_dropout(rate=0.4) %>%
  layer_dense(units = 300, activation = "relu") %>%
  layer_dense(units = 200, activation = "relu") %>%
  layer_dense(units = length(fruits), activation = "softmax")

model %>% compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics='accuracy')

t1 <- Sys.time()
history<- model %>% fit(
  x_train,y_train,
  epochs=3,
  batch_size=50)
t2 <- Sys.time()
cnn_execution_time <-  as.double(difftime(t2,t1,  units = "secs"))
model %>% evaluate(x_test, y_test)
#accuracy = 90%

plot(history)

# key findings:
# 1) knn risulta molto efficiente e preciso rispetto alle svm --> 
# 2) provare con frutti simili
# 3) risultati con cnn
# 4) risultati con rgb