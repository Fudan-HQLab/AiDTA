import random
import numpy as np
import pickle
import time
from cnn_net import PolicyValueNet
from config import CONFIG
from game import State,game_end,self_play


# Define the entire training pipeline
class TrainPipeline:

    def __init__(self, init_model=None):
        # Training parameters
        self.learn_rate = 1e-3
        self.lr_multiplier = 1  # Adaptive learning rate adjustment based on KL divergence
        self.temp = 1
        self.batch_size = CONFIG['batch_size']
        self.epochs = CONFIG['epochs']
        self.kl_targ = CONFIG['kl_targ']
        self.check_freq = 100  # Frequency to save the model
        self.game_batch_num = CONFIG['game_batch_num']  # Number of training updates

        if init_model:
            try:
                self.policy_value_net = PolicyValueNet(model_file=init_model)
                print("Successfully loaded the last final model")

            except:
                # Start training from scratch
                print('Model path does not exist, starting from scratch')
                self.policy_value_net = PolicyValueNet()

        else:
            # Start training from scratch
            print('Model path does not exist, starting from scratch')
            self.policy_value_net = PolicyValueNet()

    def policy_update(self):
        """Update the policy value network"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        print("mini_batch", mini_batch)
        sequence_batch = [data[0] for data in mini_batch]
        structure_batch = [data[1] for data in mini_batch]
        mcts_probs_batch = [data[2] for data in mini_batch]
        mcts_probs_batch = np.array(mcts_probs_batch).astype('float64')
        score_batch = [data[3] for data in mini_batch]
        score_batch = np.array(score_batch).astype('float64')

        combined_matrix_batch = []
        for sequence, structure in zip(sequence_batch, structure_batch):
            combined_matrix = State(sequence, structure)
            combined_matrix = np.squeeze(combined_matrix, axis=0)
            combined_matrix_batch.append(combined_matrix)

        # Old policy, old value function
        old_probs, old_value = self.policy_value_net.policy_value(combined_matrix_batch)

        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                combined_matrix_batch,
                mcts_probs_batch,
                score_batch,
                self.learn_rate * self.lr_multiplier)

            # New policy, new value function
            new_probs, new_value = self.policy_value_net.policy_value(combined_matrix_batch)

            kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))

            if kl > self.kl_targ * 4:  # If KL divergence is too large, terminate early
                break

        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5

        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{}"
               ).format(kl, self.lr_multiplier, loss, entropy))
        return loss, entropy

    def run(self):

        """Start training"""
        try:
            for i in range(self.game_batch_num):
                time.sleep(10)  # Update the model every 10 seconds
                while True:
                    try:
                        with open(CONFIG['train_data_buffer_path'], 'rb') as data_dict:
                            data_file = pickle.load(data_dict)
                            print("Data loaded successfully")
                            # print("data_file", data_file)
                            self.data_buffer = data_file['data_buffer']
                            self.iters = data_file['iters']
                            del data_file
                        print('Data loaded')
                        break

                    except:
                        time.sleep(5)
                print('Step {}'.format(self.iters))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                # Save the model
                self.policy_value_net.save_model(CONFIG['pytorch_model_path'])
        except KeyboardInterrupt:
            print("Training interrupted")

training_pipeline = TrainPipeline(init_model='current_policy.pkl')

training_pipeline.run()






