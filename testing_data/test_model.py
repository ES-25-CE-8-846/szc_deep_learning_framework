import torch
from models import impulse_estimator

if __name__ == "__main__":

    test_tensor = torch.rand(2, 20000, 3)

    model = impulse_estimator.ImpulseEstimatorModel(
        input_channels=2,
    )
    print(model.forward(test_tensor).size())
