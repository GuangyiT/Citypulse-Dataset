from preprocesssing import model_diff,model_noise_diff,label
trainset_perc = 0.7

#limiting data to have a smaller dataset to test

#time_limit = datas.shape[1]
time_limit = 15000

X = model_diff.values[:time_limit,:]
y = label[:time_limit]
train_size = int(trainset_perc*time_limit)

#X_outliers = model_sensor.values[:time_limit]
#X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 42,test_size = testset_perc)
X_train,y_train = X[:train_size,:],y[:train_size]
X_test,y_test = X[train_size:,:],y[train_size:]
