"""
Tests for the main module.
"""

import torch
import pytest
from leap_labs_task.main import AdversarialGenerator

# pylint: disable=protected-access

PRETRAINED_MODEL = "resnet18"
IMAGE_PATH = "tests/sample_image.JPEG"
adversarial_generator = AdversarialGenerator(PRETRAINED_MODEL)
EPSILON = 0.01
MAX_ITER = 100
TARGET_CLASS = 1
DESIRED_COFIDENCE = 0.99


def test_preprocess_image():
    """
    Test the `_preprocess_image` function.
    """
    preprocessed_image = adversarial_generator._preprocess_image(IMAGE_PATH)
    assert isinstance(
        preprocessed_image, torch.Tensor
    ), "The output is not a torch.Tensor."
    assert preprocessed_image.shape == (
        1,
        3,
        224,
        224,
    ), f"Expected shape: (1, 3, 224, 224), but got {preprocessed_image.shape}."
    assert (
        preprocessed_image.dtype == torch.float32
    ), f"Expected dtype: torch.float32, but got {preprocessed_image.dtype}."


def test_get_prediction():
    """
    Test the `_get_prediction` function.
    """
    # Test the function with a random image tensor
    preprocessed_image = adversarial_generator._preprocess_image(IMAGE_PATH)
    prediction = adversarial_generator._get_prediction(preprocessed_image)
    assert isinstance(prediction, tuple), "The output is not a tuple."
    assert len(prediction) == 2, f"Expected length: 2, but got {len(prediction)}."
    assert isinstance(prediction[0], int), "The predicted class is not a string."
    assert isinstance(prediction[1], float), "The confidence score is not a float."
    assert 0 <= prediction[0] <= 999, "The predicted class is not between 0 and 999."
    assert 0 <= prediction[1] <= 1, "The confidence score is not between 0 and 1."


def test_check_input():
    """
    Test the `_check_input` function.
    """
    # Test the function with an incorrect image path
    with pytest.raises(FileNotFoundError):
        adversarial_generator._check_input(
            "tests/incorrect_image.JPEG",
            TARGET_CLASS,
            EPSILON,
            DESIRED_COFIDENCE,
            MAX_ITER,
        )

    # Test the function with an incorrect target class
    with pytest.raises(ValueError):
        adversarial_generator._check_input(
            IMAGE_PATH, 1000, EPSILON, DESIRED_COFIDENCE, MAX_ITER
        )

    # Test the function with an incorrect epsilon
    with pytest.raises(ValueError):
        adversarial_generator._check_input(
            IMAGE_PATH, TARGET_CLASS, -0.01, DESIRED_COFIDENCE, MAX_ITER
        )

    # Test the function with an incorrect desired confidence
    with pytest.raises(ValueError):
        adversarial_generator._check_input(
            IMAGE_PATH, TARGET_CLASS, EPSILON, 1.01, MAX_ITER
        )

    # Test the function with an incorrect max iterations
    with pytest.raises(ValueError):
        adversarial_generator._check_input(
            IMAGE_PATH, TARGET_CLASS, EPSILON, DESIRED_COFIDENCE, 0
        )


def test_generate_adversarial_image():
    """
    Test the `generate_adversarial_image` function.
    """
    epsilon = 0.01
    max_iter = 100
    target_class = 1
    desired_confidence = 0.99
    output_dict = adversarial_generator.generate_adversarial_image(
        IMAGE_PATH, target_class, epsilon, desired_confidence, max_iter
    )
    assert isinstance(output_dict, dict), "The output is not a dictionary."
    assert "adversarial_image" in output_dict, "The key 'adversarial_image' is missing."
    assert "original_image" in output_dict, "The key 'original_image' is missing."
    assert "original_class" in output_dict, "The key 'original_class' is missing."
    assert (
        "original_confidence" in output_dict
    ), "The key 'original_confidence' is missing."
    assert "target_class" in output_dict, "The key 'target_class' is missing."
    assert "target_confidence" in output_dict, "The key 'target_confidence' is missing."
    assert isinstance(
        output_dict["adversarial_image"], torch.Tensor
    ), "The adversarial image is not a torch.Tensor."
    assert isinstance(
        output_dict["original_image"], torch.Tensor
    ), "The original image is not a torch.Tensor."
    assert isinstance(
        output_dict["original_class"], str
    ), "The original class is not a string."
    assert isinstance(
        output_dict["original_confidence"], float
    ), "The original confidence is not a float."
    assert isinstance(
        output_dict["target_class"], str
    ), "The target class is not a string."
    assert isinstance(
        output_dict["target_confidence"], float
    ), "The target confidence is not a float."
    assert (
        0 <= output_dict["original_confidence"] <= 1
    ), "The original confidence is not between 0 and 1."
    assert (
        0 <= output_dict["target_confidence"] <= 1
    ), "The target confidence is not between 0 and 1."
