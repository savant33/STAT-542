###Required Packages###############
#rm(list=ls())

if (!require("pacman")) install.packages("pacman")

pacman::p_load(
  #"catboost",
  "glmnet",
  "randomForest",
  "xgboost",
  "kernlab",
  "e1071",
  "lubridate",
  "lime"
)


##############Read dataset###################
start.time = proc.time()

loan_data = read.csv("loan_stat542.csv")
test_id = read.csv("Project4_test_id.csv")


############## Function for Fitting Lasso Model ##########

logLoss = function(y, p){
  if (length(p) != length(y)){
    stop('Lengths of prediction and labels do not match.')
  }
  
  if (any(p < 0)){
    stop('Negative probability provided.')
  }
  
  p = pmax(pmin(p, 1 - 10^(-15)), 10^(-15))
  mean(ifelse(y == 1, -log(p), -log(1 - p)))
}

Null_Fill_Function = function(item){
  ##Fill with mean with unskewed else use median
  if(skewness(item,na.rm = T)<0.1)
  {
    item[is.na(item)] = mean(item, na.rm=T)
  }
  else{
  item[is.na(item)] = median(item, na.rm=T)
  }
  return (item)
}

make_int = function(s){
  if (is.na(s)) {
    s
  }
  else{
    s = as.numeric(strsplit(s, ' ')[[1]][1])
    s
  }
}



log_transform_function = function(item) return(log(item+1))

############### Train and Test Data ###############
splits = c("test1","test2","test3")

model_output = NULL

for (j in 1:3){
test_data = loan_data[loan_data$id %in% test_id[, splits[j]], ]
train_data = loan_data[!loan_data$id %in% test_id[, splits[j]], ]
col.names = colnames(train_data)
NULL_names_train = colnames(train_data)[apply(is.na(train_data), 2, any)] 
NULL_names_test = colnames(test_data)[apply(is.na(train_data), 2, any)]

##Since the same NULL values are missing in train and test, we can row bind the columns and fill out null values
Ytrain = ifelse(train_data$loan_status == "Fully Paid",0,1)
new_train = train_data[,-13]
new_test = test_data[,-13]
data = rbind(new_train,new_test)
new_col = colnames(data)
#data$loan_status = ifelse(data$loan_status == "Fully Paid",0,1)
                       
################ Pre-processing and handling of missing values #######################
remove_col = c('emp_title','title','zip_code','grade','fico_range_high',
               'fico_range_low', 'open_acc') ##Removing these columns
data$mean_fico = (data$fico_range_high + data$fico_range_low)/2
data = data[,!colnames(data) %in% remove_col]
data$emp_length = as.character(data$emp_length)
data$emp_length = replace(data$emp_length, data$emp_length == '10+ years', '10 years')
data$emp_length = replace(data$emp_length, data$emp_length == '< 1 year', '0 year')
data$emp_length = as.numeric(sapply(data$emp_length, make_int))

### Earliest credit line
data$earliest_cr_line = as.character(data$earliest_cr_line)
data$earliest_cr_line = sapply(data$earliest_cr_line, function(x) as.integer(substr(x, nchar(x)-3, nchar(x))))

##Missing 

Null_cols = colnames(data)[apply(is.na(data), 2, any)]
Null_cols_continuous = Null_cols[1:length(Null_cols)]

###Replacing NA values with mean/median depending on skewness

for(i in 1: length(Null_cols_continuous)){
  data[,Null_cols_continuous[i]] = Null_Fill_Function(data[,Null_cols_continuous[i]])
}

##Replace Na in emp_length which is a categorical variable
# levels(data$emp_length) = c(levels(data$emp_length),"None")
# data$emp_length[is.na(data$emp_length)] = "None"
##Making emp_length as integer and replace missing value


###Converting term to integers
data$term = ifelse(data$term=="36 months",1, 0)

##Use log-transformation for skewed continuous variables
log_transform_var = c('annual_inc','installment') #,'dti' 'mean_fico') #'fico_range_high', 'fico_range_low')
mytest = apply(data[,log_transform_var],1,log_transform_function)
data[,log_transform_var] = mytest

##Home-wonership data
## the home_ownership, replace any and none with 'other'
data$home_ownership = as.character(data$home_ownership)
data$home_ownership = replace(data$home_ownership,which(data$home_ownership == 'OTHER'), 'RENT')
data$home_ownership = as.factor(data$home_ownership)



#Model matrix for level

mydata = model.matrix(~.,data=data)[,-c(1,2)]
train_X = mydata[1:nrow(train_data),]
test_X = mydata[nrow(train_data)+1:nrow(test_data),]
final_train = data.frame(cbind(train_X,Ytrain))

# #logistic regression
mymodel = glm(Ytrain ~ ., data=final_train, family = binomial)
prediction = predict(mymodel, data.frame(test_X), type="response")



YActual = ifelse(test_data$loan_status == "Fully Paid",0,1) 
model_glm = logLoss(YActual,prediction) 




##XGB Boost Model
set.seed(8788)

dtrain = xgb.DMatrix(train_X, label = Ytrain)
dtest = xgb.DMatrix(test_X)

param = list(objective = 'binary:logistic', max_depth =7, eta =0.2475)

model.xgb = xgb.train(dtrain, nrounds=141, params = param)
xgb.pred = predict(model.xgb, dtest)
model_xgb = logLoss(YActual, xgb.pred)




###Write output to text file

bst.pred = cbind(test_id[, splits[j]], xgb.pred)
colnames(bst.pred) = c('id', 'prob')
write.csv(bst.pred, file = paste0('mysubmission_',splits[j], '.txt'), row.names = F, quote = F)


### Total Run Time
run.time = (proc.time() - start.time)[[1]]


### Model Output
model_output = rbind(model_output, c("Test Splits" = paste0("Test-", j),
                "GLM MOdel" = model_glm, "XGB" = model_xgb, "Run Time" = run.time))
print(j)
print(model_output)
}

### Using best model (XGB Boost) on C and LoanStats_2018Q4 Data

##LoanStats_2018Q3 Data

loan_q3 = read.csv("LoanStats_2018Q3.csv")
yvalues_q3 = ifelse(loan_q3$loan_status == "Fully Paid",0,1) 
###Filtering columns to match original data
loan_q3$mean_fico = (loan_q3$fico_range_high + loan_q3$fico_range_low)/2
loan_q3_data = loan_q3[,colnames(loan_q3) %in% colnames(data)]
##Pre-processing Employee length, earliest_cr_lione and term
loan_q3_data$emp_length[loan_q3_data$emp_length=='n/a'] = NA 
loan_q3_data$emp_length = as.character(loan_q3_data$emp_length)
loan_q3_data$emp_length = replace(loan_q3_data$emp_length, loan_q3_data$emp_length == '10+ years', '10 years')
loan_q3_data$emp_length = replace(loan_q3_data$emp_length, loan_q3_data$emp_length == '< 1 year', '0 year')
loan_q3_data$emp_length = as.numeric(sapply(loan_q3_data$emp_length, make_int))
loan_q3_data$term = ifelse(loan_q3_data$term=="36 months",1, 0)
loan_q3_data$earliest_cr_line = as.character(loan_q3_data$earliest_cr_line)
loan_q3_data$earliest_cr_line = sapply(loan_q3_data$earliest_cr_line, function(x) as.integer(substr(x, nchar(x)-3, nchar(x))))
loan_q3_data$home_ownership = as.character(loan_q3_data$home_ownership)
loan_q3_data$home_ownership = replace(loan_q3_data$home_ownership,which(loan_q3_data$home_ownership == 'OTHER'), 'RENT')
loan_q3_data$home_ownership = as.factor(loan_q3_data$home_ownership)

##Replacing Null_values
loan_q3_data$emp_length[is.na(loan_q3_data$emp_length)] = mean(loan_q3_data$emp_length, na.rm=T)
loan_q3_data$dti[is.na(loan_q3_data$dti)] = mean(loan_q3_data$dti, na.rm=T)

###Taking Log transform
log_transform_var = c('annual_inc','installment') #,'dti' 'mean_fico') #'fico_range_high', 'fico_range_low')
loan_q3_data_log = apply(loan_q3_data[,log_transform_var],1,log_transform_function)
loan_q3_data[,log_transform_var] = loan_q3_data_log

###Removing % from intrate and revolutil
loan_q3_data$int_rate = as.numeric(strsplit(as.character(loan_q3_data$int_rate), '%')[[1]][1])
loan_q3_data$revol_util = as.numeric(strsplit(as.character(loan_q3_data$revol_util), '%')[[1]][1])

##Creating Model matrix and predicting
q3_data = data.frame(model.matrix(~.,data=loan_q3_data)[,-c(1,2)])
None_list = colnames(mydata)[!colnames(mydata) %in% colnames(q3_data)]
for (i in 1:length(None_list)) {
  q3_data[,None_list[i]] = 0
}

q3_data = q3_data[,colnames(mydata)]
q3_data = as.matrix(q3_data)
q3_data_new = xgb.DMatrix(q3_data)

q3.pred = predict(model.xgb, q3_data_new)
q3_pred = cbind(loan_q3_data$id, q3.pred)
colnames(q3_pred) = c('id', 'prob')
write.csv(bst.pred, file = 'mysubmission_2018Q3.txt', row.names = F, quote = F)

###Visualizationcode for LIME
random_label = c(sample(1:nrow(q3_data_new),4))
x_explain1 = as.data.frame(q3_data[random_label,])
x_explain2 = as.data.frame(train_X)
#x_explain = data.frame(q3_data[random_label,])
explainer = lime(x_explain2, model = model.xgb )
explanation =explain(x_explain1, explainer, n_labels = 2, n_features = 6)
plot_features(explanation)

##LoanStats_2018Q4 Data

loan_q4 = read.csv("LoanStats_2018Q4.csv")
yvalues_q4 = ifelse(loan_q4$loan_status == "Fully Paid",0,1) 
###Filtering columns to match original data
loan_q4$mean_fico = (loan_q4$fico_range_high + loan_q4$fico_range_low)/2
loan_q4_data = loan_q4[,colnames(loan_q4) %in% colnames(data)]
##Pre-processing Employee length, earliest_cr_lione and term
loan_q4_data$emp_length[loan_q4_data$emp_length=='n/a'] = NA 
loan_q4_data$emp_length = as.character(loan_q4_data$emp_length)
loan_q4_data$emp_length = replace(loan_q4_data$emp_length, loan_q4_data$emp_length == '10+ years', '10 years')
loan_q4_data$emp_length = replace(loan_q4_data$emp_length, loan_q4_data$emp_length == '< 1 year', '0 year')
loan_q4_data$emp_length = as.numeric(sapply(loan_q4_data$emp_length, make_int))
loan_q4_data$term = ifelse(loan_q4_data$term=="36 months",1, 0)
loan_q4_data$earliest_cr_line = as.character(loan_q4_data$earliest_cr_line)
loan_q4_data$earliest_cr_line = sapply(loan_q4_data$earliest_cr_line, function(x) as.integer(substr(x, nchar(x)-3, nchar(x))))
loan_q4_data$home_ownership = as.character(loan_q4_data$home_ownership)
loan_q4_data$home_ownership = replace(loan_q4_data$home_ownership,which(loan_q4_data$home_ownership == 'OTHER'), 'RENT')
loan_q4_data$home_ownership = as.factor(loan_q4_data$home_ownership)

##Replacing Null_values
loan_q4_data$emp_length[is.na(loan_q4_data$emp_length)] = mean(loan_q4_data$emp_length, na.rm=T)
loan_q4_data$dti[is.na(loan_q4_data$dti)] = mean(loan_q4_data$dti, na.rm=T)

###Taking Log transform
log_transform_var = c('annual_inc','installment') #,'dti' 'mean_fico') #'fico_range_high', 'fico_range_low')
loan_q4_data_log = apply(loan_q4_data[,log_transform_var],1,log_transform_function)
loan_q4_data[,log_transform_var] = loan_q4_data_log

###Removing % from intrate and revolutil
loan_q4_data$int_rate = as.numeric(strsplit(as.character(loan_q4_data$int_rate), '%')[[1]][1])
loan_q4_data$revol_util = as.numeric(strsplit(as.character(loan_q4_data$revol_util), '%')[[1]][1])

##Creating Model matrix and predicting
q4_data = data.frame(model.matrix(~.,data=loan_q4_data)[,-c(1,2)])
None_list = colnames(mydata)[!colnames(mydata) %in% colnames(q4_data)]
for (i in 1:length(None_list)) {
  q4_data[,None_list[i]] = 0
}

q4_data = q4_data[,colnames(mydata)]
q4_data = as.matrix(q4_data)
q4_data_new = xgb.DMatrix(q4_data)

q4.pred = predict(model.xgb, q4_data_new)
q4_pred = cbind(loan_q4_data$id, q4.pred)
colnames(q4_pred) = c('id', 'prob')
write.csv(bst.pred, file = 'mysubmission_2018Q4.txt', row.names = F, quote = F)
#xgb.save(model.xgb, "xgboost.model")

###Visualization Code for LIME
x_explain3 = as.data.frame(q4_data[random_label,])
explanation2 =explain(x_explain3, explainer, n_labels = 2, n_features = 6)
plot_features(explanation2)