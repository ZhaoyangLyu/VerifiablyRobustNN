import json
import os
import pdb

if __name__ == '__main__':
    folder1 = '../exp_configs/mnist'
    folder2 = '../exp_configs/mnist/ibp'
    file1_list = os.listdir(folder1)
    file2_list = os.listdir(folder2)

    file1_list = [f for f in file1_list if '.json' in f]
    file2_list = [f for f in file2_list if f in file1_list]
    
    for config_file in file2_list:
        print('Comparing file', config_file)
        with open(os.path.join(folder1, config_file)) as f:
            data = f.read()
        config1 = json.loads(data)
        if 'attack_params' in config1.keys():
            del config1['attack_params']
        with open(os.path.join(folder2, config_file)) as f:
            data = f.read()
        config2 = json.loads(data)

        print(config1==config2)
        # pdb.set_trace()
        



