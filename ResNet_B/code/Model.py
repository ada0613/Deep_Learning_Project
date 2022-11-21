import torch
import os, time
import numpy as np
import torch.nn as nn
from Network import MyNetwork
from ImageUtils import parse_record, preprocess_test, visualize
import tqdm

"""This script defines the training, validation and testing process.
"""

class MyModel(object):

    def __init__(self, configs):
        self.configs = configs
        self.network = MyNetwork(configs)

    def model_setup(self):
        lr = self.configs['learning_rate']
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr,momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        # pass

    def train(self, x_train, y_train, configs, x_valid=None, y_valid=None):
        self.model = self.network(None, True)
        self.model_setup()
        
        # Determine how many batches in an epoch
        batch_size = configs['batch_size']
        max_epoch = configs['max_epochs']
        num_samples = x_train.shape[0]
        num_batches = num_samples // batch_size

        print('### Training... ###')
        for epoch in range(1, max_epoch+1):
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            # Set the learning rate for this epoch
            # Usage example: divide the initial learning rate by 10 after several epochs
            self.scheduler.step()
            print('Epoch-{0} lr: {1}'.format(epoch, self.optimizer.param_groups[0]['lr']))
            if epoch % 20 == 0:print()
            loss = 0

            for i in range(num_batches):
                
                # Construct the current batch.
                # Don't forget to use "parse_record" to perform data preprocessing.
                # Don't forget L2 weight decay
                curr_x_batch = [parse_record(x, True) for x in curr_x_train[i * batch_size: (i + 1) * batch_size]]
                curr_y_batch = curr_y_train[i * batch_size: (i + 1) * batch_size]
                curr_x_batch_tensor = torch.tensor(curr_x_batch).float().cuda()
                curr_y_batch_tensor = torch.tensor(curr_y_batch).float().cuda()

                self.model = self.network.network.cuda()
                
                self.optimizer.zero_grad()
                outputs = self.model(curr_x_batch_tensor)
                loss = self.criterion(outputs, curr_y_batch_tensor.long())
                loss.backward()
                self.optimizer.step()

                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)
            
            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, duration))
            
            if epoch % configs['save_interval'] == 0:
            
                self.save(epoch)

    def evaluate(self, x, y, checkpoint_num_list):
        model_dir = self.configs['model_dir']
        resnet_version = self.configs['resnet_version']
        self.network.network.eval()
        print('### Test or Validation ###')
        for checkpoint_num in checkpoint_num_list:
            if resnet_version == 1:
                res = 'standard_res'
            else:
                res = 'bottleneck_res'
            checkpointfile = os.path.join(model_dir, 'model-%s-%d.ckpt'%(res, checkpoint_num))
            self.load(checkpointfile)

            preds = []
            for i in tqdm.tqdm(range(x.shape[0])):
                curr_x = x[i]
                curr_x = torch.tensor(parse_record(curr_x, False)).float().cuda()
                # self.network = self.network.cpu()
                predict_output = self.network.network(curr_x.view(1, 3, 32, 32))
                predict = int(torch.max(predict_output.data, 1)[1])
                # print("Predict: ", predict)
                preds.append(predict)
            y = torch.tensor(y)
            preds = torch.tensor(preds)
            print(y.shape, preds.shape)

            print('Test accuracy: {:.4f}'.format(torch.sum(preds==y)/y.shape[0]))

    def predict_prob(self, x, configs):
        checkpoint_name = configs['checkpoint_name']
        private_model = configs['private_model']
        
        print('### Private Data Test ###')
        model_name = checkpoint_name + '.ckpt'
        best_model_path = os.path.join(private_model, model_name)
        self.load(best_model_path)
        self.network.network.eval()

        predictions = []
        for i in tqdm.tqdm(range(x.shape[0])):
            curr_x = x[i].reshape((32, 32, 3))
            device = 'cuda:0'
            inputs = preprocess_test(curr_x).float().to(device)
            inputs = inputs.view(1, 3, 32, 32)
            output = self.network.network(inputs)
            predictions.append(output.cpu().detach().numpy())
        # converting the result of predictions into probabilities
        predictions = np.array(predictions)
        predictions = predictions.reshape((predictions.shape[0], predictions.shape[1]*predictions.shape[2]))
        predictions_exp = np.exp(predictions)
        pred_exp_sum = predictions_exp.sum(axis=1)
        predictions_proba = (predictions_exp.T/pred_exp_sum).T

        return np.array(predictions_proba)
        
    def save(self, epoch):
        model_dir = self.configs['model_dir']
        resnet_version = self.configs['resnet_version']
        if resnet_version == 1:
            res = 'standard_res'
        else:
            res = 'bottleneck_res'
        checkpoint_path = os.path.join(model_dir, 'model-%s-%d.ckpt'%(res, epoch))
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.network.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")
       
    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cuda:0")
        self.network.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))