# Principal Component Analysis Using R
#
# Mushroom dataset 
# https://archive.ics.uci.edu/ml/datasets/mushroom
# Mushroom records drawn from The Audubon Society Field Guide to North American Mushrooms (1981). 
# G. H. Lincoff (Pres.), New York: Alfred A. Knopf 
# The dataset has 8124 samples and 22 attributes
# This is a classification problem classifying whether the mushroom is edible or poisonous.

# Here, the goal of this script is not classification. Instead, this script simplifies
# the dataset using Principal Component Analaysis into two main Principal Components.
# The Pricipal Components are evaluated using the covariance method.

library(ggplot2)
# Read mushroom dataset into a dataframe
# All the attributes are converted into factors
mushroom <- read.csv('mushrooms.csv',header=T)
# Remove constant column
# Get the dimension (row, column) of the dataframe
data_dim = dim(mushroom)

# Convert the factor variables to numeric
mushroom.N <- data.frame(sapply(mushroom, function(x) as.numeric(x)))
mushroom.N <- mushroom.N[,-which(names(mushroom.N) %in% c("veil.type"))]
# Divide the data into the target and attributes
mushroom.target <- mushroom.N[,"class"]
mushroom.attr <- mushroom.N[,names(mushroom.N) !="class"]
sca.mush.attr <- apply(mushroom.attr, 2, scale)
# Calculate eigen values and vectors using the covariance matrix of the attributes
mush.cov <- cov(sca.mush.attr)
# mush.eigen vector and values are PCA components
mush.eigen <- eigen(mush.cov)
# Select first two principal components
mush.phi <- mush.eigen$vectors[,1:3]
mush.PC1 <- as.matrix(sca.mush.attr) %*% mush.phi[,1]
mush.PC2 <- as.matrix(sca.mush.attr) %*% mush.phi[,2]
mush.PC3 <- as.matrix(sca.mush.attr) %*% mush.phi[,3]

mush.PVE <- mush.eigen$values / sum(mush.eigen$values)*100
barplot(mush.PVE, main='Scree Plot', xlab='Principal Component', ylab='Percent Variation')

# Plot the mushroom class in terms of PC1 and PC2
pca.df <- data.frame(class=mushroom.target,X=mush.PC1,Y=mush.PC2)
ggplot(pca.df,aes(X,Y,col=class))+geom_point()+xlab("PC1 ")+ylab("PC2")+ggtitle("PC2 vs PC1")

# Loading score (attribute vector) of PC1 component
scores1 <- mush.eigen$vectors[,1]
names(scores1)<- names(mushroom.attr)
abs.scores.1 <- abs(scores1)
abs.scores.1.ranked <- sort(abs.scores.1, decreasing = T)
top10.factors.1 <- names(abs.scores.1.ranked[1:10])
scores1[top10.factors.1]

# Loading score (attribute vector) of PC2 component
scores2 <- mush.eigen$vectors[,2]
names(scores2)<- names(mushroom.attr)
abs.scores.2 <- abs(scores2)
abs.scores.2.ranked <- sort(abs.scores.2, decreasing = T)
top10.factors.2 <- names(abs.scores.2.ranked[1:10])
scores2[top10.factors.2]

# Loading score (attribute vector) of PC3 component
scores3 <- mush.eigen$vectors[,3]
names(scores3)<- names(mushroom.attr)
abs.scores.3 <- abs(scores3)
abs.scores.3.ranked <- sort(abs.scores.3, decreasing = T)
top10.factors.3 <- names(abs.scores.3.ranked[1:10])
scores3[top10.factors.3]


