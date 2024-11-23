"""
Utils module for the Leap Labs Task.
"""

from typing import Dict, Union

import matplotlib.pyplot as plt
import torch


def show_images(output_dict: Dict[str, Union[torch.Tensor, str, float]]) -> None:
    """
    Display the original and the adversarially perturbed images, along with the
    predicted classes and the confidence scores. The keys are as follows:
        "adversarial_image" (torch.Tensor): The adversarial image tensor (3xHxW).
        "original_image" (torch.Tensor): The original image tensor (3xHxW).
        "original_class" (str): The original class name.
        "original_confidence" (float): The model's confidence for the original class.
        "target_class" (str): The target class name.
        "target_confidence" (float): The model's confidence for the target class.

    Args:
    output_dict: Dict[str, Union[torch.Tensor, str, float]]: The dictionary predicted
        by the `__call__` method of the `AdversarialGenerator` class.
    """

    original_image = output_dict["original_image"].numpy().transpose(1, 2, 0)
    adversarial_image = output_dict["adversarial_image"].numpy().transpose(1, 2, 0)
    original_class = output_dict["original_class"]
    original_confidence = output_dict["original_confidence"]
    target_class = output_dict["target_class"]
    target_confidence = output_dict["target_confidence"]

    _, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(original_image)
    ax[0].set_title(
        (
            f"Original Image\n Model Prediction: {original_class}\n "
            f"Confidence: {original_confidence:.2f}"
        )
    )
    ax[0].axis("off")
    ax[1].imshow(adversarial_image)
    ax[1].set_title(
        (
            f"Adversarial Image\n Model Prediction: {target_class}\n "
            f"Confidence: {target_confidence:.2f}"
        )
    )
    ax[1].axis("off")

    # Display the difference between the original and adversarial images
    diff_image = adversarial_image.astype(float) - original_image.astype(float)
    diff_image = diff_image.sum(axis=2)
    ax[2].imshow(diff_image.astype(int), cmap="jet")
    ax[2].set_title("Difference Image")
    ax[2].axis("off")
    # show colorbar
    plt.colorbar(
        ax[2].imshow(diff_image.astype(int), cmap="jet"),
        ax=ax[2],
        orientation="horizontal",
    )
    plt.show()
