# Linear Regression Using R

# Load packages
library(tidyverse)  # data manipulation and visualization
library(modelr)     # provides easy pipeline modeling functions
library(broom)      # helps to tidy up model outputs

# Read the data
advertising <- read.csv("http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv") %>%  select(-X1)
set.seed(123)
sample <- sample(c(TRUE,FALSE), nrow(advertising), replace=T, prob=c(0.6,0.4))
train <- advertising[sample, ]
test <- advertising[!sample, ]

# Linear regression 1
# Sales as a function of TV advertisement
# sales = beta0 + beta1 * TV + error
model1 <- lm(sales ~ TV, data = train)

# Summary of model 1
summary(model1)

# Confidence interval of model 1
confint(model1)

# Residual Standard Error
# The deviation of sales from the regression line
sigma(model1)
# The percentage error of sales over average sales
sigma(model1)/mean(train$sales)

# R^2 
# The result suggests that TV advertising budget can 
# explain 64% of the variability in our sales data
rsquare(model1, data = train)
# R2 value equals cor(train$TV, train$Sales)^2 for a simple linear regression model

sm <- summary(model1)
# Obtain F statistics value
sm$fstatistic

# Plot the data and the regression line
ggplot(train, aes(TV,sales)) + geom_point() + geom_smooth(method="lm")+geom_smooth(se=F, color='red')

# Plot diagnostic figures.
plot(model1)

# Add prediction to the test dataset
(test <- test %>% add_predictions(model1))

# test MSE
test %>% add_predictions(model1) %>% summarise(MSE = mean((sales - pred)^2))
# Train MSE
train %>% add_predictions(model1) %>% summarise(MSE = mean((sales - pred)^2))

model1_results <- augment(model1, train)

# Multiple linear regression
# Y = beta0 + beta1 * X1 + beta2 * X2 + beta3 * X3 .....
model2 <- lm(sales ~ TV + radio + newspaper, data = train)

# Model 1 statistics
broom::glance(model1)
# Model 2 statistics
broom::glance(model2)



