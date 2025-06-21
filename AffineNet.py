import torch
import torch.nn as nn
import torch.nn.functional as F


class AffineNet(nn.Module):
    def __init__(self, image_size=(128, 128)):
        super(AffineNet, self).__init__()
        self.image_size = image_size

        # Feature extraction layers
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(2, 64, kernel_size=3, padding=1),  # 2 channels for concatenated single-channel images
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Output: 64 x H/2 x W/2

            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Output: 128 x H/4 x W/4

            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Output: 256 x H/8 x W/8

            # Fourth conv block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # Output: 512 x 4 x 4
        )

        # Regression head for affine parameters
        self.regressor = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 6)  # 6 affine transformation parameters
        )

        # Initialize the final layer to predict identity transformation
        nn.init.zeros_(self.regressor[-1].weight)
        nn.init.constant_(self.regressor[-1].bias, 0)
        # Set bias for identity transformation: [1, 0, 0, 0, 1, 0]
        self.regressor[-1].bias.data[0] = 1.0  # scale_x
        self.regressor[-1].bias.data[4] = 1.0  # scale_y

    def forward(self, img1, img2):
        """
        Args:
            img1: torch.Tensor of shape (batch_size, 1, H, W) - reference image
            img2: torch.Tensor of shape (batch_size, 1, H, W) - transformed image

        Returns:
            affine_params: torch.Tensor of shape (batch_size, 6)
                          [a, b, tx, c, d, ty] where transformation matrix is:
                          [[a, b, tx],
                           [c, d, ty]]
        """
        # Concatenate images along channel dimension
        x = torch.cat([img1, img2], dim=1)  # Shape: (batch_size, 2, H, W)

        # Extract features
        features = self.features(x)  # Shape: (batch_size, 512, 4, 4)

        # Flatten for regression
        features = features.view(features.size(0), -1)  # Shape: (batch_size, 512*4*4)

        # Predict affine parameters
        affine_params = self.regressor(features)  # Shape: (batch_size, 6)

        return affine_params

    def params_to_matrix(self, params):
        """
        Convert 6 parameters to 2x3 affine transformation matrix

        Args:
            params: torch.Tensor of shape (batch_size, 6)

        Returns:
            matrix: torch.Tensor of shape (batch_size, 2, 3)
        """
        batch_size = params.size(0)
        matrix = torch.zeros(batch_size, 2, 3, device=params.device)

        matrix[:, 0, 0] = params[:, 0]  # a (scale/rotate x)
        matrix[:, 0, 1] = params[:, 1]  # b (shear/rotate)
        matrix[:, 0, 2] = params[:, 2]  # tx (translation x)
        matrix[:, 1, 0] = params[:, 3]  # c (shear/rotate)
        matrix[:, 1, 1] = params[:, 4]  # d (scale/rotate y)
        matrix[:, 1, 2] = params[:, 5]  # ty (translation y)

        return matrix
