import torch
import os, argparse
import numpy as np
from Model import MyModel
from DataLoader import load_data, train_valid_split, load_testing_images
from Configure import model_configs, training_configs, prediction_configs
from ImageUtils import visualize


parser = argparse.ArgumentParser()
parser.add_argument("mode", help="train, test or predict")
parser.add_argument("--data_dir", default='cifar-10-batches-py', help="path to the data")
parser.add_argument("--save_dir",default='saved_models', help="path to save the results")
# parser.add_argument("--private_dir", default='private_test', help="path to private test data")
parser.add_argument("--result_dir", default="result", type=str, help="path to save the private results")

args = parser.parse_args()
# torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
	print(args.mode, args.data_dir)
	model = MyModel(model_configs)

	if args.mode == 'train':
		x_train, y_train, x_test, y_test = load_data(args.data_dir)
		x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)

		model.train(x_train, y_train, training_configs, x_valid, y_valid)
		# model.evaluate(x_valid, y_valid,[1])

	elif args.mode == 'test':
		# Testing on public testing dataset
		_, _, x_test, y_test = load_data(args.data_dir)
		checkpoint_num_list = [20]
		model.evaluate(x_test, y_test,checkpoint_num_list)

	elif args.mode == 'predict':
		# Predicting and storing results on private testing dataset
		x_test = load_testing_images(args.data_dir)
		os.makedirs(args.result_dir, exist_ok=True)
		predictions = model.predict_prob(x_test, prediction_configs)
		np.save(os.path.join(args.result_dir, 'predictions.npy'), predictions)
		print("Predictions saved")
### END CODE HERE

