

library(MASS)
library(ggplot2)
library(gmodels)
library(e1071)
library(rpart)
library(randomForest)
library(cluster) #we need a new package for this plot
library(DMwR)
# install.packages("ROSE")
# library(ROSE)
# install.packages("pROC")
library(pROC)

df <- read.csv("bank-additional-full.csv", header = TRUE, sep = ";")

head(df)

colnames(df)

levels(df$month)

df <- df[-c(5,9,11,13)]

#test train split

test_perc = 0.2
test_rnums = sample(nrow(df), nrow(df)*test_perc)
dftest  <- df[ test_rnums, ]
df2     <- df[-test_rnums, ]


#validation  split from training data
valid_perc = 0.25
valid_rnums = sample(nrow(df2), nrow(df2)*valid_perc)
dfvalid <- df2[ valid_rnums, ]
dftrain <- df2[-valid_rnums,  ]

# using SMOTE to balance out the data
dftrain_bal <- SMOTE(y~., data=dftrain, perc.over=300, perc.under=150)
# SMOTE overwrote the default S3 method, which caused inconsistencies in running models. Using alternate ROSE for the same
# Ref: https://journal.r-project.org/archive/2014-1/menardi-lunardon-torelli.pdf
# dftrain_bal <- ovun.sample(y ~ ., data = dftrain, method = "both",N = 20000, p=0.5)$data
length(dftrain_bal)

summary(dftrain$y)
summary(dftrain_bal$y)

model_comp_df<-data.frame(modelname=(character()),
                          thres=double(), 
                          acc=double(), 
                          jsi=double(), 
                          auc=double(), stringsAsFactors=FALSE) # create an empty df for capturing accuracy of different models

# Check column types
for (col in 1:length(colnames(df))){
    print(c(col,  colnames(df)[col]  , is.factor(df[,col]) ,  is.numeric(df[,col])  ))
}

logit_full_model <- glm(y ~ . ,data=dftrain_bal,family=binomial("logit"))

summary(logit_full_model)

logit_sw_model <- stepAIC(logit_full_model, trace = FALSE)
summary(logit_sw_model)

logit_model2 <- glm(formula = y ~ job + contact + day_of_week + campaign + 
     poutcome + emp.var.rate + cons.price.idx + cons.conf.idx, 
    family = binomial("logit"), data = dftrain_bal)

summary(logit_model2)

logit_model3 <- glm(formula = y ~ job + contact + day_of_week + campaign + 
     emp.var.rate + cons.price.idx + cons.conf.idx, 
    family = binomial("logit"), data = dftrain_bal)

summary(logit_model3)

logit_fmodel = logit_full_model
formu <- logit_fmodel$formula

logit_yhattrain=predict(formula=formu, newdata=dftrain_bal, type='response') # use selected model to predict probabilities (yhat)

logit_pred_thr<-data.frame(thres=double(), acc=double(), jsi=double(), auc = double()) # create an empty df for capturing accuracy vs different thresholds

# iterate over thresholds and record accuracy
for (t in seq(0.2, 0.8, by=0.05)){
    pred.val <- ifelse(logit_yhattrain> t,1,0)
    ct=table(dftrain_bal$y, pred.val)
    acc = sum(diag(ct))/(sum(ct)) 
    jsi = ct[2,2]/(sum(ct)-ct[1,1]) 
    vauc = auc(ifelse(dftrain_bal$y=="yes",1,0), pred.val)
    logit_pred_thr[nrow(logit_pred_thr) + 1,] <- c(round(t,3), acc,jsi,vauc )
}

logit_pred_thr  # show accuracies

logit_pred_thr <- logit_pred_thr[order(-logit_pred_thr[,4]),]  # Sort the df by accuracy (decending) 
logit_thres = logit_pred_thr[1,1]                        # Threshold is assigned as the value in 1st row
logit_thres

logit_yhatvalid=predict(logit_fmodel, newdata=dfvalid, type='response')
logit_pred <- ifelse(logit_yhatvalid> logit_thres,1,0)
ct=table(dfvalid$y, logit_pred)
logit_jsi = round(ct[2,2]/(sum(ct)-ct[1,1]),2) 
logit_acc = round(sum(diag(ct))/sum(ct),2)
logit_auc = round(auc(ifelse(dfvalid$y=="yes",1,0), logit_pred),2)

model_comp_df[nrow(model_comp_df)+1,] <- c("Logistic", logit_thres, logit_acc, logit_jsi, logit_auc)

lda_model <- lda(formula=formu ,data=dftrain_bal)
summary(lda_model)

(lda_model)$prior    

(lda_model)$scaling

lda_yhat=predict(lda_model, dfvalid, type='response') # use selected model to predict probabilities (yhat)
lda_pred_val <- ifelse(lda_yhat$class=="yes",1,0)

lda_ct=table(dfvalid$y, lda_pred_val)
lda_jsi = round(lda_ct[2,2]/(sum(lda_ct)-lda_ct[1,1]),2) 
lda_acc = round(sum(diag(lda_ct))/sum(lda_ct),2)
lda_auc = round(auc(ifelse(dfvalid$y=="yes",1,0), lda_pred_val),2)

model_comp_df[nrow(model_comp_df)+1,] <- c("LDA", 0, lda_acc, lda_jsi, lda_auc)

nb_full_model <- naiveBayes(formula=formu ,data=dftrain_bal) # Remember to convert y into as.factor (y)
summary(nb_full_model)
nb_full_model

nb_yhat=predict(nb_full_model, dfvalid) # use selected model to predict probabilities (yhat)
nb_pred_val <- ifelse(nb_yhat=="yes",1,0)

ct=table(dfvalid$y, nb_pred_val)
nb_jsi = round(ct[2,2]/(sum(ct)-ct[1,1]),2) 
nb_acc = round(sum(diag(ct))/sum(ct),2)
nb_auc = round(auc(ifelse(dfvalid$y=="yes",1,0), nb_pred_val),2)

model_comp_df[nrow(model_comp_df)+1,] <- c("NB", 0, nb_acc, nb_jsi, nb_auc)

cart_model <- rpart(formula=formu ,data=dftrain_bal, method="class") 
# summary(cart_model)
# printcp(cart_model)
plot(cart_model, uniform=TRUE, main=" Classification Tree")
text(cart_model, use.n=TRUE, all=TRUE, cex=.8)

printcp(cart_model)

cart_yhat=predict(cart_model, dfvalid, type="class") # use selected model to predict probabilities (yhat)
cart_pred_val <-  ifelse(cart_yhat=="yes",1,0)

ct=table(dfvalid$y, cart_pred_val)
cart_jsi = round(ct[2,2]/(sum(ct)-ct[1,1]),2) 
cart_acc = round(sum(diag(ct))/sum(ct),2)
cart_auc = round(auc(ifelse(dfvalid$y=="yes",1,0), cart_pred_val),2)

model_comp_df[nrow(model_comp_df)+1,] <- c("CART", 0, cart_acc, cart_jsi, cart_auc)

rf_pred_thr<-data.frame(tree=double(), acc=double(), jsi=double(), auc = double())

# iterate over thresholds and record accuracy
for (t in seq(100, 2000, by=100)){
    rf_model <- randomForest(formula=formula=formu ,data=dftrain_bal, ntree=t) 
    rf_yhat  <-predict(rf_model, dftrain_bal, predict.all=TRUE)$aggregate 
    rf_pred <-  ifelse(rf_yhat=="yes",1,0)
    ct=table(dftrain_bal$y, rf_pred)
    rfacc = sum(diag(ct))/(sum(ct)) 
    rfjsi = ct[2,2]/(sum(ct)-ct[1,1]) 
    rfauc = auc(ifelse(dftrain_bal$y=="yes",1,0), rf_pred)
    rf_pred_thr[nrow(rf_pred_thr) + 1,] <- c(round(t,3), rfacc, rfjsi, rfauc)
}

rf_pred_thr

ntr <- rf_pred_thr[order(-rf_pred_thr[,4]),][1,1]
ntr

rf_model <- randomForest(formula=formu ,data=dftrain_bal, ntree=ntr) 
rf_yhat  <-predict(rf_model, dfvalid, predict.all=TRUE)$aggregate
rf_pred  <-  ifelse(rf_yhat=="yes",1,0)
clusplot(dfvalid[,-18], rf_yhat,color=TRUE, shade=TRUE, labels=4, lines=0, main="Random Forest, holdout data")

ct=table(dfvalid$y, rf_pred)
rf_acc = round(sum(diag(ct))/(sum(ct)) ,2)
rf_jsi = round(ct[2,2]/(sum(ct)-ct[1,1]),2)
rf_auc = round(auc(ifelse(dfvalid$y=="yes",1,0), rf_pred),2)

model_comp_df[nrow(model_comp_df)+1,] <- c("Random Forest", ntr, rf_acc, rf_jsi, rf_auc)

logit_nnet <- function(X, Y, step_size = 0.5, reg = 0, niteration){  
    # get dim of input 
    N <- nrow(X) # number of examples  
    K <- ncol(Y) # number of classes  
    D <- ncol(X) # dimensionality   
    # initialize parameters randomly 
    W <- 0.01 * matrix(rnorm(K), nrow = D)  
    #b <- matrix(0, nrow = 1, ncol = h)  
    #W2 <- 0.01 * matrix(rnorm(h*K), nrow = h)  
    #b2 <- matrix(0, nrow = 1, ncol = K)   
    # gradient descent loop to update weight and bias  
    for (i in 0:niteration){    
        scores <- X%*%W # class score     
        # compute and normalize class probabilities   
        probs=matrix(0,N,2) 
        exp_scores <- exp(-(scores))    
        probs[,1] <- 1/ (1+exp_scores) 
        probs[,2] <- exp_scores / (1+exp_scores)  
        # compute the loss: sofmax and regularization    
        corect_logprobs <- -log(probs)    
        data_loss <- mean(corect_logprobs*Y)  #cost function (Cross Entropy Cost Function)   
        reg_loss <- reg*sum(W*W)  #regularization term for parameters   
        loss <- data_loss + reg_loss    
        if (i%%1000 == 0 | i == niteration){print(paste("iteration", i,': loss', loss))} #check progress every 1000 iterations    
        # compute the gradient on scores (final delta)    
        dscores <- probs[,1]-Y[,1]    
        dscores <- dscores/N   
        dW <- t(X)%*%dscores  # finally into W,b 
        dW <- dW + reg *W      # add regularization gradient contribution  
        W <- W-step_size*dW    # update parameter
    }
    #return(list(W, b, W2, b2))
    return(list(W))
}

# https://www.r-bloggers.com/fitting-a-neural-network-in-r-neuralnet-package/
# install.packages("neuralnet")
library(neuralnet)

nn_model <- neuralnet(formula=y~.,data=dftrain,hidden=c(5,3),linear.output=FALSE)
pr.nn <- compute(nn_model, dftrain)









model_comp_df

model_comp_df$thres <- as.numeric(model_comp_df$thres)
model_comp_df$acc   <- as.numeric(model_comp_df$acc)
model_comp_df$jsi   <- as.numeric(model_comp_df$jsi)
model_comp_df$auc   <- as.numeric(model_comp_df$auc)

model_comp_df <- model_comp_df[order(-model_comp_df[,4]),]

model <- model_comp_df[1,1]
thres <- model_comp_df[1,2]
acc   <- model_comp_df[1,3]
jsi   <- model_comp_df[1,4]
auc   <- model_comp_df[1,5]


if (model=="Logistic"){
    yhattest=predict(logit_fmodel, dftest,type="response" ) 
    pred_test <- ifelse(yhattest> thres,1,0)
} else if (model=="LDA"){
    yhattest=predict(lda_model, dftest,type="response" ) 
    pred_test <- ifelse(yhattest$class=="yes",1,0)
} else if (model=="NB"){
    yhattest=predict(nb_full_model, dftest) # use selected model to predict probabilities (yhat)
    pred_test <- ifelse(yhattest=="yes",1,0) 
} else if (model=="CART"){
    yhattest=predict(cart_model, dftest,type="class") # use selected model to predict probabilities (yhat)
    pred_test <-  ifelse(yhattest=="yes",1,0) 
} else if (model=="Random Forest"){
    yhattest=predict(rf_model, dftest, predict.all=TRUE)$aggregate
    pred_test <- ifelse(yhattest=="yes",1,0)
} else {}

model
thres

("Confusion Matrix")
("-----------------------------")
(ctv=table(dftest$y, pred_test))

('')
("Accuracy")
("-----------------------------")
sum(diag(prop.table(ctv))) 

('')
("Jaccard Similarity Index")
("-----------------------------")
ctv[2,2]/(sum(ctv)-ctv[1,1]) 

seg.summ<-function(data,groups){
    aggregate(data, list(groups), function(x) mean(as.numeric(x)))
} # computing group-level mean values


X<-df[sample(nrow(df), "10000"), -length(df) ]
X2<-df[sample(nrow(df), "10000"), c(1,5,6,7,12:16) ]


seg.dist<-daisy(X) #compute dissimilarity matrix, default=euclidean distance
# daisy: compute all the pairwise dissimilarities (distances) between observations in the data set

as.matrix(seg.dist)[1:5,1:5]
dim(as.matrix(seg.dist)) #distances between 300 members

seg.hc<-hclust(seg.dist, method="complete")
# complete linkage method evaluates the distance between every member
plot(seg.hc) #resulting tree for all N=300 observations of seg.df.



## decide number of segments based on dendrogram
cut(as.dendrogram(seg.hc), h=0.7)
plot(cut(as.dendrogram(seg.hc), h=0.7)$lower[[1]])#cut with 0.5 in the plot


### Check similarity example ###
X[c(101,107),]

### Specifying the number of groups we want ###
plot(seg.hc)
rect.hclust(seg.hc, k=3, border="red") #prespecified K=4



#assignment vector
seg.hc.segment <- cutree(seg.hc, k=3) #membership for 4 groups
table(seg.hc.segment)

# for your convenience, it's good to use the saved function..
heirclust<-seg.summ(X, seg.hc.segment) #the function for computing segment-level means#please see some relevant graphics in textbook
heirclust

heirclust$Group.Desc <- c("New users (users not contacted before) when emp.var.rate is a little higher than last month, and eubor is high",
                          "Users when Eubor is same, but the unemployment going up",
                          "Recurring customers when employement a ittle low and interest rate is low")

heirclust

#k-means
##let's change categorical items to numbers for computing distance
seg.df.num<-X2
seg.df.num$housing <- ifelse(seg.df.num$housing=="no", 0, 1) 
seg.df.num$loan <- ifelse(seg.df.num$loan=="no", 0, 1)
seg.df.num$contact <- ifelse(seg.df.num$contact=="telephone", 0, 1)
#this is what we can do for this data...
set.seed(1000) # please try this without set.seed first
seg.k<-kmeans(seg.df.num, centers=3)

kmeans <- seg.summ(seg.df.num, seg.k$cluster)

kmeans$Group.Desc <- c("Landline users when emp.var.rate is same as last month, and eubor is high",
                       "Cell phone users when Eubor is same, but the unemployment going up",
                       "Cell phone users when employement is up and interest rate is high")

kmeans


