import json
import os
import pdb


def mnist_crown_ibp_to_ibp_configs(input_folder, output_folder):
    file_list = os.listdir(input_folder)
    os.makedirs(output_folder, exist_ok=True)
    for config_file in file_list:
        with open(os.path.join(input_folder, config_file)) as f:
            data = f.read()
        config = json.loads(data)
        # pdb.set_trace()
        config['models_path'] = config['models_path'].replace('crown-', '')
        config['training_params']['multi_gpu'] = False
        config['training_params']['device_ids'] = []
        config['training_params']['method_params']['bound_type'] = 'interval'
        if 'adaptive-lb' in config['training_params']['method_params']['bound_opts'].keys():
            del config['training_params']['method_params']['bound_opts']['adaptive-lb']

        config['training_params']['after_crown_or_lbp_settings']['multi_gpu'] = False
        # print('The configuration is:')
        # print(json.dumps(config, indent=4))
        new_file_name = config_file.replace('crown-', '')
        print('%s has been transformed and saved to %s' % (config_file, new_file_name))
        with open(os.path.join(output_folder, new_file_name), 'w') as f:
            json.dump(config, f, indent=4)

if __name__ == '__main__':
    input_folder = '../exp_configs/mnist/crown-ibp'
    output_folder = '../exp_configs/mnist/ibp'
    mnist_crown_ibp_to_ibp_configs(input_folder, output_folder)
    



