CONFIG = {
    'dirichlet': 0.2,       # Dirichlet noise parameter: Chess, 0.3; Shogi (Japanese chess), 0.15; Go, 0.03
    'play_out': 100,       # Number of simulations per move
    'c_puct': 10,            # Weight of U (exploration term)
    'buffer_size': 10000,  # Size of experience replay buffer
    'pytorch_model_path': 'current_policy.pkl',    # PyTorch model path
    'train_data_buffer_path': 'train_data_buffer.pkl',   # Path to data container
    'batch_size': 512,      # Number of train steps per update
    'kl_targ': 0.02,        # KL divergence target
    'epochs': 5,            # Number of epochs per update
    'game_batch_num': 3000, # Number of training updates
    'use_frame': 'pytorch', # Use pytorch
    'train_update_interval': 600,   # Model update interval time
    'use_redis': False,     # Data storage method
    'list1':['A', 'T', 'C', 'G', 'AGG', 'GAA', 'GAC', 'GGG', 'CTG', 'ATTT', 'GAGA', 'GCCA', 'GTTG', 'TCTG', 'TTAGA', 'TTAGT', 'TTGGA', 'TTGGT', 'TTTGC', 'TACGTC', 'TCTTTG', 'TTCTGG', 'TTTTAA', 'TTTTAC', 'AAG&CTT', 'GGC&GCC', 'CAC&GTG', 'CGT&ACG', 'TAT&ATA', 'AATA&TATT', 'ATCG&CGAT', 'GTGG&CCAC', 'TAAA&TTTA', 'TACA&TGTA', 'AGGTG&CACCT', 'GGGTG&CACCC', 'CGGTG&CACCG', 'CCAGC&GCTGG', 'TCAGG&CCTGA', 'AACATT&AATGTT', 'GACATT&AATGTC', 'GGGGCA&TGCCCC', 'CTGGCA&TGCCAG', 'TATCTG&CAGATA'],
    'list2':['.', '.', '.', '.', '...', '...', '...', '...', '...', '....', '....', '....', '....', '....', '.....', '.....', '.....', '.....', '.....', '......', '......', '......', '......', '......', '(((&)))', '(((&)))', '(((&)))', '(((&)))', '(((&)))', '((((&))))', '((((&))))', '((((&))))', '((((&))))', '((((&))))', '(((((&)))))', '(((((&)))))', '(((((&)))))', '(((((&)))))', '(((((&)))))', '((((((&))))))', '((((((&))))))', '((((((&))))))', '((((((&))))))', '((((((&))))))'],
    'min_len': 50, # Min length of complete sequence
    'max_len': 61, # Max length of complete sequence 49+12 =61
    'policy_num': 2200 # 44*50 = 2200
}
# copy the list1 and list2 from list.txt in 0_hdock_data after runing '2_result_get.py'
