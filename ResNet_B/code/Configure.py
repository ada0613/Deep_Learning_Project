model_configs = {
	"name": 'MyModel',
	"save_dir": 'saved_models',
	"resnet_version": 2,
	"learning_rate": 0.1,
	"model_dir": 'saved_models',
	"num_classes": 10,
	"first_num_filters": 32,
	"resnet_size": 3,
	# ...
}

training_configs = {
	"learning_rate": 0.1,
	"model_dir": 'saved_models',
	"max_epochs":100,
	"batch_size":128,
	"save_interval": 5
	# ...
}
prediction_configs = {
	"private_model": 'saved_models',
	"checkpoint_name": 'model-bottleneck_res-20',
	"result_dir": 'result'
}


### END CODE HERE