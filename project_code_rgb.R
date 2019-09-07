library(jpeg)
library(imager)
library(class)
library(e1071)
library(FactoMineR)
library(factoextra)
library(keras)
library(tensorflow)

read_fruit_rgb <- function(path,set,fruit, dim = 50){
  filenames <- list.files(paste(path,set,fruit,sep=""), pattern = "*.jpg")
  images_rgb= list()
  i = 1
  for (name in  filenames){
    path_name = paste(path,set,fruit,"/",name,sep="")
    img <- load.image(path_name)
    inmg <- resize(img,dim,dim)
    images_rgb[[i]] = as.vector(img)
    i = i+1
  }
  return(images_rgb)
}

#number of pixels to be used
dim = 50
path = "../fruits-360_dataset/fruits-360/"

# grayscale extraction

set = "Training/"
training_labels_vector <- NULL
#fruits = c ("Apricot","Banana", "Blueberry")
fruits = c ("Apple Golden 1", "Apple Golden 2", "Apple Red 2","Apple Red 1", "Banana", "Apricot", "Blueberry")

i = 0
training_features_matrix_rgb <- NULL
for (fruit in fruits){
  images_rgb <- read_fruit_rgb(path,set,fruit)
  images_rgb <- matrix(unlist(images_rgb), ncol = dim*dim*3, byrow = TRUE)
  labels <-replicate(dim(images_rgb)[1], i)
  training_labels_vector <- c(training_labels_vector,labels)
  training_features_matrix_rgb <- rbind(training_features_matrix_rgb,images_rgb)
  i <- i +1
}


set = "Test/"
test_labels_vector <- NULL

i = 0
test_features_matrix_rgb <- NULL
for (fruit in fruits){
  images_rgb <- read_fruit_rgb(path,set,fruit)
  images_rgb <- matrix(unlist(images_rgb), ncol = dim*dim*3, byrow = TRUE)
  test_features_matrix_rgb <- rbind(test_features_matrix_rgb,images_rgb)
  labels <-replicate(dim(images_rgb)[1], i)
  test_labels_vector <- c(test_labels_vector,labels)
  i <- i +1
}

#RGB SECTION
t1 <- Sys.time()
pca_rgb <- prcomp(training_features_matrix_rgb,center = TRUE, scale=TRUE)
t2 <- Sys.time()
pca_rgb_execution_time <- as.double(difftime(t2,t1,  units = "min")) # 57.14827 minutes
pr_var_rgb <- (pca_rgb$sdev[1:200])^2
prop_varex_rgb <- pr_var_rgb/sum(pr_var_rgb)
plot(cumsum(prop_varex_rgb), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained")

training_features_matrix_rgb_pc <- pca_rgb$x
test_features_matrix_rgb_pc <- predict(pca_rgb, newdata = test_features_matrix_rgb)

neighbours <- c(1,3,5,7,9,11)
features <- c(3,10,50,100,150)
res_knn <- c()
time_knn <- c()
i <- 1
for (n_f in features){
  for (n in neighbours){
    t1 <- Sys.time()
    knn <-knn(training_features_matrix_rgb_pc[,1:n_f], test_features_matrix_rgb_pc[,1:n_f], as.factor(training_labels_vector), k=n)
    t2 <- Sys.time()
    time_knn[i] <- as.double(difftime(t2,t1,  units = "secs"))
    res_knn[i] <- sum(knn == test_labels_vector)/length(test_labels_vector)
    i <- i +1
  }
}

time_knn
#[1]  0.3174582  0.3071220  0.3119910  0.3195369  0.3226392  0.3241997  0.7329173  0.7252805  0.7408412  0.7422488  0.7687256
#[12]  0.7552888  5.2286835  5.2341237  5.3759677  5.3922064  5.3973079  5.6475928 15.7594717 15.1007993 15.3384907 15.3406219
#[23] 15.1126273 14.6999736 23.1744928 23.1385210 23.0463626 23.1748335 23.1040637 23.2457619
res_knn
#[1] 0.7901316 0.7853070 0.7817982 0.7750000 0.7719298 0.7688596 0.9260965 0.9228070 0.9192982 0.9182018 0.9140351 0.9122807
#[13] 0.9320175 0.9293860 0.9267544 0.9247807 0.9190789 0.9182018 0.9296053 0.9234649 0.9201754 0.9186404 0.9109649 0.9087719
#[25] 0.9291667 0.9223684 0.9184211 0.9164474 0.9087719 0.9041667


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
  training_features_matrix_svm <- as.data.frame(training_features_matrix_rgb_pc[,1:n_f])
  training_features_matrix_svm$target <- as.factor(training_labels_vector)
  test_features_matrix_svm <- as.data.frame(test_features_matrix_rgb_pc[,1:n_f])
  
  t1 <- Sys.time()
  SVM <- svm(target ~ ., data = training_features_matrix_svm, kernel = "radial")
  t2 <- Sys.time()
  res <- predict(SVM, newdata = test_features_matrix_svm)
  time_svm[i] <- as.double(difftime(t2,t1,  units = "secs"))
  res_svm[i] <-     sum(res == test_labels_vector)/length(test_labels_vector)
  i <- i +1
}

time_svm

res_svm


plot(res_svm, type = c("b"),pch=1 , xlab = "Features", ylab="Precision", main="SVM ACCURACY", xaxt = "n")
axis(1, at=1:length(features), labels=features)
legend("right", legend = features, col=1:length(features), pch=1, title="Features")

plot(time_svm, type = c("b"),pch=1 , xlab = "Features", ylab="Execution time (s)", main="SVM EXECUTION TIME", xaxt = "n")
axis(1, at=1:length(features), labels=features)

features <- c(3,10,50,100,150)
res_svm <- c()
time_svm <- c()
i <- 1
for (n_f in features){
  training_features_matrix_svm <- as.data.frame(training_features_matrix_rgb_pc[,1:n_f])
  training_features_matrix_svm$target <- as.factor(training_labels_vector)
  test_features_matrix_svm <- as.data.frame(test_features_matrix_rgb_pc[,1:n_f])
  
  t1 <- Sys.time()
  SVM <- svm(target ~ ., data = training_features_matrix_svm, kernel = "radial")
  t2 <- Sys.time()
  res <- predict(SVM, newdata = test_features_matrix_svm)
  time_svm[i] <- as.double(difftime(t2,t1,  units = "secs"))
  res_svm[i] <-     sum(res == test_labels_vector)/length(test_labels_vector)
  i <- i +1
}

time_svm
#[1] 11.048466  7.334074 20.145182 42.065335 66.613935 radial kernel
res_svm 
#0.6754386 0.8802632 0.9164474 0.9195175 0.9151316 radial kernel

plot(res_svm, type = c("b"),pch=1 , xlab = "Features", ylab="Precision", main="SVM ACCURACY", xaxt = "n")
axis(1, at=1:length(features), labels=features)

plot(time_svm, type = c("b"),pch=1 , xlab = "Features", ylab="Execution time (s)", main="SVM EXECUTION TIME", xaxt = "n")
axis(1, at=1:length(features), labels=features)



y_train <- to_categorical(training_labels_vector, length(fruits))
y_test <- to_categorical(test_labels_vector, length(fruits))
x_train<-array(training_features_matrix_rgb,c(nrow(training_features_matrix_rgb),dim,dim,3))
x_test<-array(test_features_matrix_rgb,c(nrow(test_features_matrix_rgb),dim,dim,3))

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 16, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(50, 50, 3)) %>%
  layer_max_pooling_2d(pool_size = c(4,4)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(2, 2), activation = "relu") %>%
  layer_flatten() %>%
  layer_dropout(rate=0.5) %>%
  layer_dense(units = 300, activation = "relu") %>%
  layer_dense(units = 200, activation = "tanh") %>%
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
#$acc
#[1] 0.9311404
# accuracy 93%


