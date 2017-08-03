setwd("C:/Users/Andrew/Google Drive/Work/ActiveCampaign")
library(car)
df = read.table("train.csv", sep = ",", header = T)
df = df[,-c(1, 4, 9, 11)]


# hist(df$Survived)
# table(df$Survived)
# 342 Survived
# 549 Perished

# df$Age = ifelse(df$Age < 13, 1, 
#          ifelse(df$Age < 18, 2, 
#          ifelse(df$Age < 35, 3, 
#          ifelse(df$Age < 70, 4, 5))))
df$Age = ifelse(is.na(df$Age), 20, df$Age)
df$Embarked = ifelse(is.na(df$Embarked), 0, df$Embarked)
df$Male = ifelse(df$Sex == "male", 1, 0)
df = df[,-3]
# table(df$Age)

fix(df)
summary(df)
sum(is.na(df$Fare))

###################################################################
#                        K Means Clustering
###################################################################

# 2 groups, survived/perished
# Therefore, 2 clusters

ndf = df[,-1]
result = kmeans(ndf, 2)
result$size
table(df$Survived)

# Scatterplot of Actual Survival
scatterplot(Fare ~ Age | Survived, data = df, reg.line = F, boxplots = F, 
            xlab = "Age", ylab = "Fare", 
            legend.title = "Survival", grid = F, smoother = F, 
            legend.coords = "topright", main = "Titanic Survival Plot")

# Scatterplot of K Means Results
scatterplot(Fare ~ Age | result$cluster, data = df, reg.line = F, boxplots = F, 
            xlab = "Age", ylab = "Fare", 
            legend.title = "Survival", grid = F, smoother = F, 
            legend.coords = "topright", main = "Titanic Survival Plot")

# scatterplot() function does not allow side-by-side plots :(
# it calls the layout() function within the scatterplot() function
# therefore anything prior, par(mfrow) or layout() will be overridden

# Logisitic Regression Coursera Code

ex2d1 = read.table("ex2data1.txt", sep = ",")
ex2d1 = as.data.frame(ex2d1)
names(ex2d1) = c("EX1", "EX2", "Admit")
result2 = kmeans(ex2d1, 2)
result2

scatterplot(EX1 ~ EX2 | Admit, data = ex2d1, reg.line = F, boxplots = F, 
            xlab = "Exam 1 Score", ylab = "Exam 2 Score", 
            legend.title = "Groups", grid = F, smoother = F, 
            legend.coords = "topright", main = "Admission Based on Two Exam Scores")

scatterplot(EX1 ~ EX2 | result2$cluster, data = ex2d1, reg.line = F, boxplots = F, 
            xlab = "Exam 1 Score", ylab = "Exam 2 Score", 
            legend.title = "Groups", grid = F, smoother = F, 
            legend.coords = "topright", main = "Admission Based on Two Exam Scores")


###################################################################
#                            PCA
###################################################################

par(mfrow=c(1,1))
pairs(df)
df.pca = princomp(ndf, scores = T, cor = T)
summary(df.pca)
plot(df.pca)
biplot(df.pca)
df.pca$loadings

###################################################################
#                         CV SVM with PCA
###################################################################

library(caret)
control = trainControl(
  method = "cv", 
  number = 5, 
  allowParallel = T)

grid = expand.grid(.sigma = c(0.001, 0.01, 0.1, 0.5, 1, 5), 
                    .C = seq(1, 20, 1))
# sigma for RBF is value that contains the nonlinearity
# C is cost for SVM for how wide the margin should be when building the classifier

model = train(as.factor(Survived)~., 
               data = df, 
               trControl = control, 
               method = "svmRadial", 
               tuneGrid = grid)
model

# Predict using train after CV
pred = predict(model, df)
out = table(pred, df$Survived)
sum(diag(out))/nrow(df)
# sigma = 0.05 and C = 10
# 0.8462

################################################################
#                       Test
################################################################

df_test_orig = read.table("test.csv", sep = ",", header = T)
df_test = df_test_orig[,-c(1, 3, 8, 10)]
df_test$Age = ifelse(df_test$Age < 13, 1, 
                ifelse(df_test$Age < 18, 2, 
                ifelse(df_test$Age < 35, 3, 
                ifelse(df_test$Age < 70, 4, 5))))
df_test$Age = ifelse(is.na(df_test$Age), 0, df_test$Age)
df_test$Fare = ifelse(is.na(df_test$Fare), mean(df_test$Fare), df_test$Fare)

fix(df_test)
dim(df_test)

pred_test = predict(model, df_test)
pred_test
sub = as.data.frame(cbind(df_test_orig$PassengerId, pred_test))
names(sub) = c("PassengerId", "Survived")
fix(sub)
sub$Survived = ifelse(sub$Survived == 1, 0, 1)
write.table(sub, file = "submission.csv", sep = ",", row.names = F, col.names = T)
