# upload the dataset
data_US_github <- "https://raw.githubusercontent.com/erassek/US_speech_data/master/US%20Presidential%20Data.csv"
data_US <- read.csv(data_US_github)


# load library
library(caret)
library(e1071)


# Transforming the dependent variable to a factor
data_US$Win.Loss = as.factor(data_US$Win.Loss)

# Explore data_US dataset
str(data_US)
dim(data_US)
names(data_US)
head(data_US)

# Exploratory data analysis
ggplot(gather(data_US[,2:ncol(data_US)]), aes(value)) + 
  geom_histogram(bins = 5, fill = "blue", alpha = 0.6) + 
  facet_wrap(~key, scales = 'free_x')

#Partitioning the data into training and validation data
set.seed(101)
index = createDataPartition(data_US$Win.Loss, p = 0.7, list = F )
train_set = data_US[index,]
validation_set = data_US[-index,]

# Explore data
dim(train_set)
dim(validation_set)
names(train_set)
head(train_set)
head(validation_set)

# Setting levels for both training and validation data
levels(train_set$Win.Loss) <- make.names(levels(factor(train_set$Win.Loss)))
levels(validation_set$Win.Loss) <- make.names(levels(factor(validation_set$Win.Loss)))

# Setting up train controls
repeats = 3
numbers = 10
tunel = 10

set.seed(1234)
x <- trainControl(method = "repeatedcv",
                 number = numbers,
                 repeats = repeats,
                 classProbs = TRUE,
                 summaryFunction = twoClassSummary)

model_knn <- train(Win.Loss~., data= train_set, method="knn", 
                preProcess=c("center","scale"),
                trControl=x,metric="ROC",
                tuneLength= tunel)

# Summary of model
model_knn
plot(model_knn)

# Validation
valid_pred <- predict(model_knn,validation_set, type = "prob")

#Storing Model Performance Scores

# computing a simple ROC curve
library(ROCR)
pred_val <-prediction(valid_pred[,2],validation_set$Win.Loss)

# Calculating Area under Curve (AUC)
perf_val <- performance(pred_val,"auc")
perf_val

# Plot AUC (x-axis: fpr, y-axis: tpr)
perf_val <- performance(pred_val, "tpr", "fpr")
plot(perf_val, col = "green", lwd = 1.5)
auc <- auc@y.values[[1]]

#Calculating and plotting KS statistics
ks <- max(attr(perf_val, "y.values")[[1]]-(attr(perf_val, "x.values")[[1]]))
plot(perf_val,main=paste0(' KS=',round(ks*100,1),'%'))
lines(x = c(0,1),y=c(0,1))
ks

#sensitivity/specificity curve (x-axis: specificity,
#y-axis: sensitivity)
perf_val2 <- performance(pred_val, "sens", "spec")
plot(perf_val2)

