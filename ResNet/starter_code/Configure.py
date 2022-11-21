# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE

model_configs = {
	"name": 'MyModel',
	"save_dir": 'saved_models',
	# "depth": 2,
	"resnet_version": 1,
	"learning_rate": 0.1,
	"model_dir": 'saved_models',
	"num_classes": 10,
	"first_num_filters": 16,
	"resnet_size": 3,
	'weight_decay': 2e-4
	# ...
}

training_configs = {
	"learning_rate": 0.1,
	"model_dir": 'saved_models',
	"max_epochs":200,
	"batch_size": 128,
	"save_interval": 10
	# ...
}
prediction_configs = {
	"private_model": 'saved_models',
	# "checkpoint_name": 'model-standard_res-180',
	"checkpoint_name": 'model-bottleneck_res-80',
	"result_dir": 'result'
}


### END CODE HERE