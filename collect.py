from collections import deque
import copy
import os
import pickle
import time
from game import availabel,State,game_end,self_play
from cnn_net import PolicyValueNet
from mcts import MCTSPlayer
from config import CONFIG

# list1 = ["AAA","CCC&GGG"]
# list2 = ["...","(((&)))"]
# fragment
list1 = ['GGC&GCC', 'CGT&ACG', 'TAT&ATA', 'CAC&GTG', 'AAG&CTT', 'TACA&TGTA', 'ATCG&CGAT', 'AATA&TATT', 'TAAA&TTTA', 'GTGG&CCAC', 'GGGTG&CACCC', 'CCAGC&GCTGG', 'CGGTG&CACCG', 'AGGTG&CACCT', 'TCAGG&CCTGA', 'TAAATG&CATTTA', 'AAAAGG&CCTTTT', 'TAAAGG&CCTTTA', 'TAAGTG&CACTTA', 'TAAAGT&ACTTTA', 'AGG', 'CTG', 'GGG', 'GAA', 'GAC', 'GAGA', 'GCCA', 'ATTT', 'TCTG', 'GTTG', 'TTAGT', 'TTAGA', 'TTGGT', 'TTGGA', 'TTTGC', 'TTTTAA', 'TCTTTG', 'TTTTAC', 'TACGTC', 'TTCTGG']
# structure of fragment
list2 = ['(((&)))', '(((&)))', '(((&)))', '(((&)))', '(((&)))', '((((&))))', '((((&))))', '((((&))))', '((((&))))', '((((&))))', '(((((&)))))', '(((((&)))))', '(((((&)))))', '(((((&)))))', '(((((&)))))', '((((((&))))))', '((((((&))))))', '((((((&))))))', '((((((&))))))', '((((((&))))))', '...', '...', '...', '...', '...', '....', '....', '....', '....', '....', '.....', '.....', '.....', '.....', '.....', '......', '......', '......', '......', '......']


# Define the entire sequence concatenation process
class CollectPipeline:

    def __init__(self,init_model=None):
        self.sequence = ""
        self.structure = ""

        self.temp = 1
        self.n_playout = CONFIG['play_out'] # The number of moves per simulation
        self.c_puct = CONFIG['c_puct']
        self.buffer_size = CONFIG['buffer_size']
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iters = 0

    # Load the model from the body
    def load_model(self,model_path = CONFIG['pytorch_model_path']):
        try:
            self.policy_value_net = PolicyValueNet(model_file=model_path)

        except:
            self.policy_value_net = PolicyValueNet()

        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)


    def collect_selfplay_data(self,n_games = 1):
        for i in range(n_games):
            self.load_model()

            play_data = self_play(self.sequence,self.structure,self.mcts_player)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)

            if os.path.exists(CONFIG['train_data_buffer_path']):
                while True:
                    try:
                        with open(CONFIG['train_data_buffer_path'],'rb') as data_dict:
                            data_file = pickle.load(data_dict)
                            self.data_buffer = data_file['data_buffer']
                            self.iters = data_file['iters']
                            del data_file
                            self.iters +=1
                            self.data_buffer.extend(play_data)


                        break
                    except :
                        time.sleep(30)

            else:
                self.data_buffer.extend(play_data)
                self.iters +=1
            data_dict = {'data_buffer':self.data_buffer,'iters':self.iters}
            with open(CONFIG['train_data_buffer_path'],'wb') as data_file:
                pickle.dump(data_dict,data_file)
        return self.iters

    def run(self):
        try:
            k = 0
            while True:
                k += 1
                print("self play times",k)
                iters = self.collect_selfplay_data()
                print('batch i : {},episode_len : {}'.format(iters,self.episode_len))
        except KeyboardInterrupt:
            print('quit')

collecting_pipeline = CollectPipeline(init_model='current_policy.pkl')
collecting_pipeline.run()




