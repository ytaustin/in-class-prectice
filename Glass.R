library(tidyverse)
library(caret)
library(mlbench)
data("Glass")
Glass <- as.tibble(Glass)
ggplot(data=Glass, mapping= aes(x= Type, y= RI))+geom_boxplot()
ggsave("RI.jpg")
ggplot(data=Glass, mapping= aes(x= Type, y= Na))+geom_boxplot()
ggsave("Na.jpg")
ggplot(data=Glass, mapping= aes(x= Type, y= Mg))+geom_boxplot()
ggsave("Mg.jpg")
ggplot(data=Glass, mapping= aes(x= Type, y= Al))+geom_boxplot()
ggsave("Al.jpg")
ggplot(data=Glass, mapping= aes(x= Type, y= Si))+geom_boxplot()
ggsave("Si.jpg")
ggplot(data=Glass, mapping= aes(x= Type, y= K))+geom_boxplot()
ggsave("K.jpg")
ggplot(data=Glass, mapping= aes(x= Type, y= Ca))+geom_boxplot()
ggsave("Ca.jpg")
ggplot(data=Glass, mapping= aes(x= Type, y= Ba))+geom_boxplot()
ggsave("Ba.jpg")
ggplot(data=Glass, mapping= aes(x= Type, y= Fe))+geom_boxplot()
ggsave("Fe.jpg")

set.seed(1)
trainIndex <-
  createDataPartition(Glass$Type,
                      p = 0.8,
                      list = FALSE,
                      times = 1)
GlassTrain <- Glass[trainIndex, ]
GlassTest <- Glass[-trainIndex, ]
scaler <- preProcess(Glass, method = c("center", "scale"))
GlassTrain <- predict(scaler, GlassTrain)
GlassTest <- predict(scaler, GlassTest)

knnmodel_RI <-
  train(Type ~ RI, data = GlassTrain, method = "knn")
GlassPredictions_RI <-
  predict(knnmodel_RI, GlassTest)
confusionMatrix(GlassPredictions_RI, GlassTest$Type)


knnmodel_Na <-
  train(Type ~ Na, data = GlassTrain, method = "knn")
GlassPredictions_Na <-
  predict(knnmodel_Na, GlassTest)
confusionMatrix(GlassPredictions_Na, GlassTest$Type)

knnmodel_Mg <-
  train(Type ~ Mg, data = GlassTrain, method = "knn")
GlassPredictions_Mg <-
  predict(knnmodel_Mg, GlassTest)
confusionMatrix(GlassPredictions_Mg, GlassTest$Type)

knnmodel_Al <-
  train(Type ~ Al, data = GlassTrain, method = "knn")
GlassPredictions_Al <-
  predict(knnmodel_Al, GlassTest)
confusionMatrix(GlassPredictions_Al, GlassTest$Type)

knnmodel_Si <-
  train(Type ~ Si, data = GlassTrain, method = "knn")
GlassPredictions_Si <-
  predict(knnmodel_Si, GlassTest)
confusionMatrix(GlassPredictions_Si, GlassTest$Type)

knnmodel_K <-
  train(Type ~ K, data = GlassTrain, method = "knn")
GlassPredictions_K <-
  predict(knnmodel_K, GlassTest)
confusionMatrix(GlassPredictions_K, GlassTest$Type)

knnmodel_Ca <-
  train(Type ~ Ca, data = GlassTrain, method = "knn")
GlassPredictions_Ca <-
  predict(knnmodel_Ca, GlassTest)
confusionMatrix(GlassPredictions_Ca, GlassTest$Type)

knnmodel_Ba <-
  train(Type ~ Ba, data = GlassTrain, method = "knn")
GlassPredictions_Ba <-
  predict(knnmodel_Ba, GlassTest)
confusionMatrix(GlassPredictions_Ba, GlassTest$Type)

knnmodel_Fe <-
  train(Type ~ Fe, data = GlassTrain, method = "knn")
GlassPredictions_Fe <-
  predict(knnmodel_Fe, GlassTest)
confusionMatrix(GlassPredictions_Fe, GlassTest$Type)

knnmodel_RI_Na <-
  train(Type ~ RI+ Na, data = GlassTrain, method = "knn")
GlassPredictions_RI_Na <-
  predict(knnmodel_RI_Na, GlassTest)
confusionMatrix(GlassPredictions_RI_Na, GlassTest$Type)

knnmodel_RI_Mg <-
  train(Type ~ RI+ Mg, data = GlassTrain, method = "knn")
GlassPredictions_RI_Mg <-
  predict(knnmodel_RI_Mg, GlassTest)
confusionMatrix(GlassPredictions_RI_Mg, GlassTest$Type)

knnmodel_RI_Al <-
  train(Type ~ RI+ Al, data = GlassTrain, method = "knn")
GlassPredictions_RI_Al <-
  predict(knnmodel_RI_Al, GlassTest)
confusionMatrix(GlassPredictions_RI_Al, GlassTest$Type)

knnmodel_RI_Si <-
  train(Type ~ RI+ Si, data = GlassTrain, method = "knn")
GlassPredictions_RI_Si <-
  predict(knnmodel_RI_Si, GlassTest)
confusionMatrix(GlassPredictions_RI_Si, GlassTest$Type)

knnmodel_RI_K <-
  train(Type ~ RI+ K, data = GlassTrain, method = "knn")
GlassPredictions_RI_K <-
  predict(knnmodel_RI_K, GlassTest)
confusionMatrix(GlassPredictions_RI_K, GlassTest$Type)

knnmodel_RI_Ca <-
  train(Type ~ RI+ Ca, data = GlassTrain, method = "knn")
GlassPredictions_RI_Ca <-
  predict(knnmodel_RI_Ca, GlassTest)
confusionMatrix(GlassPredictions_RI_Ca, GlassTest$Type)

knnmodel_RI_Ba <-
  train(Type ~ RI+ Ba, data = GlassTrain, method = "knn")
GlassPredictions_RI_Ba <-
  predict(knnmodel_RI_Ba, GlassTest)
confusionMatrix(GlassPredictions_RI_Ba, GlassTest$Type)

knnmodel_RI_Fe <-
  train(Type ~ RI+ Fe, data = GlassTrain, method = "knn")
GlassPredictions_RI_Fe <-
  predict(knnmodel_RI_Fe, GlassTest)
confusionMatrix(GlassPredictions_RI_Fe, GlassTest$Type)

knnmodel_RI_Mg_Na <-
  train(Type ~ RI+ Mg + Na, data = GlassTrain, method = "knn")
GlassPredictions_RI_Mg_Na <-
  predict(knnmodel_RI_Mg_Na, GlassTest)
confusionMatrix(GlassPredictions_RI_Mg_Na, GlassTest$Type)

knnmodel_RI_Mg_Al <-
  train(Type ~ RI+ Mg + Al, data = GlassTrain, method = "knn")
GlassPredictions_RI_Mg_Al <-
  predict(knnmodel_RI_Mg_Al, GlassTest)
confusionMatrix(GlassPredictions_RI_Mg_Al, GlassTest$Type)

knnmodel_RI_Mg_Si <-
  train(Type ~ RI+ Mg + Si, data = GlassTrain, method = "knn")
GlassPredictions_RI_Mg_Si <-
  predict(knnmodel_RI_Mg_Si, GlassTest)
confusionMatrix(GlassPredictions_RI_Mg_Si, GlassTest$Type)

knnmodel_RI_Mg_K <-
  train(Type ~ RI+ Mg + K, data = GlassTrain, method = "knn")
GlassPredictions_RI_Mg_K <-
  predict(knnmodel_RI_Mg_K, GlassTest)
confusionMatrix(GlassPredictions_RI_Mg_K, GlassTest$Type)

knnmodel_RI_Mg_Ca <-
  train(Type ~ RI+ Mg + Ca, data = GlassTrain, method = "knn")
GlassPredictions_RI_Mg_Ca <-
  predict(knnmodel_RI_Mg_Ca, GlassTest)
confusionMatrix(GlassPredictions_RI_Mg_Ca, GlassTest$Type)

knnmodel_RI_Mg_Ba <-
  train(Type ~ RI+ Mg + Ba, data = GlassTrain, method = "knn")
GlassPredictions_RI_Mg_Ba <-
  predict(knnmodel_RI_Mg_Ba, GlassTest)
confusionMatrix(GlassPredictions_RI_Mg_Ba, GlassTest$Type)


knnmodel_RI_Mg_Fe <-
  train(Type ~ RI+ Mg + Fe, data = GlassTrain, method = "knn")
GlassPredictions_RI_Mg_Fe <-
  predict(knnmodel_RI_Mg_Fe, GlassTest)
confusionMatrix(GlassPredictions_RI_Mg_Fe, GlassTest$Type)

knnmodel_RI_Mg_K_Na <-
  train(Type ~ RI+ Mg + K +Na, data = GlassTrain, method = "knn")
GlassPredictions_RI_Mg_K_Na <-
  predict(knnmodel_RI_Mg_K_Na, GlassTest)
confusionMatrix(GlassPredictions_RI_Mg_K_Na, GlassTest$Type)


knnmodel_RI_Mg_K_Al <-
  train(Type ~ RI+ Mg + K +Al, data = GlassTrain, method = "knn")
GlassPredictions_RI_Mg_K_Al <-
  predict(knnmodel_RI_Mg_K_Al, GlassTest)
confusionMatrix(GlassPredictions_RI_Mg_K_Al, GlassTest$Type)


knnmodel_RI_Mg_K_Si <-
  train(Type ~ RI+ Mg + K +Si, data = GlassTrain, method = "knn")
GlassPredictions_RI_Mg_K_Si <-
  predict(knnmodel_RI_Mg_K_Si, GlassTest)
confusionMatrix(GlassPredictions_RI_Mg_K_Si, GlassTest$Type)


knnmodel_RI_Mg_K_Ca <-
  train(Type ~ RI+ Mg + K +Ca, data = GlassTrain, method = "knn")
GlassPredictions_RI_Mg_K_Ca <-
  predict(knnmodel_RI_Mg_K_Ca, GlassTest)
confusionMatrix(GlassPredictions_RI_Mg_K_Ca, GlassTest$Type)


knnmodel_RI_Mg_K_Ba <-
  train(Type ~ RI+ Mg + K +Ba, data = GlassTrain, method = "knn")
GlassPredictions_RI_Mg_K_Ba <-
  predict(knnmodel_RI_Mg_K_Ba, GlassTest)
confusionMatrix(GlassPredictions_RI_Mg_K_Ba, GlassTest$Type)


knnmodel_RI_Mg_K_Fe <-
  train(Type ~ RI+ Mg + K +Fe, data = GlassTrain, method = "knn")
GlassPredictions_RI_Mg_K_Fe <-
  predict(knnmodel_RI_Mg_K_Fe, GlassTest)
confusionMatrix(GlassPredictions_RI_Mg_K_Fe, GlassTest$Type)