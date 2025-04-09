import torch 
from models import impulse_estimator


if __name__ == "__main__":

    test_tensor = torch.rand(2, 20000, 3)

    model = impulse_estimator.ImpulseEstimatorModel(
        in_seq_len=20000, input_channels=2, n_impulse=25, out_impulse_len=20000, output_shape=(3,100)
    )

    print(model.forward(test_tensor).size())
