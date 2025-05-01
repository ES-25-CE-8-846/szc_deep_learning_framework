import argparse
import yaml
import importlib
import models
import training

models.filter_estimator.FilterEstimatorModel

parser = argparse.ArgumentParser()
parser.add_argument("config_path")
args = parser.parse_args()


def get_class_or_func(path):
    module_name, func_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)

if __name__ == "__main__":
    config_path = args.config_path

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(config)

    model_class = get_class_or_func(config['training_run']['model'])
    loss_function = get_class_or_func(config['training_run']['loss_function'])




