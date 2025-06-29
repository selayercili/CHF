"""
1) Import in train and test data
2) Loop through the models
3) Train the models
4) Save the model weights in a standard location ./weights/{model_name}/{epoch}.pth

from model_1 import model_1
from model_2 import model_2

model_names = [model_1, model_2]
config_args = yaml.safe_load(configs/model_config.yaml)

for model_name in model_names:
    model = model_1(train, test, config_args[model_name])
    model.load_weights(f"./weights/{model_name}/{epoch}.pth")
    model.test()
"""
