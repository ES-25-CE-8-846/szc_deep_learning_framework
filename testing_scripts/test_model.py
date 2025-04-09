import torch

from models.impulse_estimator import ImpulseEstimatorModel


if __name__ == "__main__":

    test_tensor = torch.rand(2, 20000, 3)

    model = ImpulseEstimatorModel(
        in_seq_len=20000, input_channels=2, n_impulse=25, out_impulse_len=20000
    )

    model.forward(test_tensor)
