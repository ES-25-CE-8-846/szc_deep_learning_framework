# Framework for Sound Zone Control using Deep Learning

This framework is designed to be a highly modular system where loss functions and parameters can be easily modified through a training configuration file.

## General Structure

The framework is organized into three main directories: `training`, `testing`, and `models`, each containing the relevant code for that stage of the workflow. The main scripts that users interact with are located in the root directory.

### Training

To train a model, you need to provide a configuration file. This file contains all necessary training and testing parameters. During training, a testing configuration is automatically generated and saved in the experiment directory along with the model checkpoints.

**Training example:**

```bash
python train.py configs/some_config.yaml
```

**Testing (Quantitative)**

To evaluate the model quantitatively, use the test_model.py script and point it to the experiment directory:
```bash
python test_model.py exp/some_experiment/  
```

**Testing (Qualitative)**

For qualitative analysis, use the qualitative_model_analysis.py script. This script launches a terminal-based user interface (TUI) for exploring model outputs:
```bash
python qualitative_model_analysis.py exp/some_experiment/  
```
where a tui is exposed. 
