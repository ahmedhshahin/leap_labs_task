"""
The main script for generating adversarial examples. We will use torchvision pretrained
models on ImageNet dataset.
"""

import logging
import os
from typing import Dict, Tuple, Union

import torch
from torchvision import models
from torchvision.io import read_image

# pylint: disable=logging-fstring-interpolation
logging.getLogger("PIL").setLevel(logging.WARNING)


class AdversarialGenerator:
    """
    A class to generate adversarial images using pretrained models from torchvision.models

    Args:
        pretrained_model (str): The name of the pretrained model from torchvision.models
        verbose (bool): Whether to print confidence wrt target class at each iteration.
            Default is False.
    """

    def __init__(self, pretrained_model: str, verbose: bool = False) -> None:
        if not pretrained_model:
            raise ValueError(
                f"Please provide a pretrained model name from torchvision.models.\
                             Choose from {models.list_models(module=models)}"
            )

        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logging.info(
            f"Initializing AdversarialGenerator with model: {pretrained_model}"
        )

        available_weights = torch.hub.load(
            "pytorch/vision", "get_model_weights", name=pretrained_model
        )
        available_weights = list(available_weights)
        # if there are multiple weights, we will use the first one
        # TODO: choose the latest one or allow user to choose
        if not available_weights:
            logging.error(
                f"No weights available for {pretrained_model}."
                "Please choose another model."
            )
        if len(available_weights) > 1:
            logging.warning(
                f"Multiple weights available for {pretrained_model}."
                f"Using {available_weights[0]}"
            )

        self.model = models.__dict__[pretrained_model](weights=available_weights[0])
        self.weight = available_weights[0]
        self.preprocess = self.weight.transforms(antialias=True)
        self.categories = self.weight.meta["categories"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        self.model.to(self.device)
        self.model.eval()
        self.verbose = verbose

    def _get_prediction(self, image: torch.Tensor) -> Tuple[int, float]:
        """
        Get the model's prediction for the image and the confidence for the prediction
        (the softmax probability).

        Args:
            image (torch.Tensor): The image tensor (3xHxW).

        Returns:
            Tuple[int, float]: The predicted class index and the confidence
                (probability).
        """
        prob = self.model(image).squeeze(0).softmax(dim=0)
        confidence, predicted_class = prob.max(dim=0)
        return predicted_class.item(), confidence.item()

    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Read and reprocess the image to be compatible with the model.

        Args:
            image_path (str): The path to the image file.

        Returns:
            torch.Tensor: The preprocessed image tensor.
        """
        image = read_image(image_path)
        return self.preprocess(image).unsqueeze(0).to(self.device)

    def _denormalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Denormalize the image tensor back to the original scale.

        Args:
            image (torch.Tensor): The image tensor (3xHxW).

        Returns:
            torch.Tensor: The denormalized image tensor.
        """
        mean = torch.Tensor(self.preprocess.mean).view(3, 1, 1)
        std = torch.Tensor(self.preprocess.std).view(3, 1, 1)
        return (255 * (image * std + mean)).clamp(0, 255).to(torch.uint8)

    def _check_input(
        self,
        image_path: str,
        target_class: Union[str, int],
        epsilon: float,
        desired_confidence: float,
        max_iter: int,
    ) -> None:
        """
        Check the input arguments for the generate_adversarial_image function. Raise
        ValueError if the input is invalid.
        """
        # check image_path
        if not isinstance(image_path, str):
            raise ValueError("image_path must be a string")
        if not image_path.lower().endswith((".jpg", ".jpeg", ".png")):
            raise ValueError("image_path must be a path to a .jpg, .jpeg, or .png file")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")

        # check target_class
        if not isinstance(target_class, (str, int)):
            raise ValueError(
                "target_class must be a string or an integer between 0 and 999"
            )
        if isinstance(target_class, int):
            if not 0 <= target_class < 1000:
                raise ValueError("target_class must be an integer between 0 and 999")
        else:
            if target_class not in self.categories:
                raise ValueError(
                    "target_class must be one of the classes in the ImageNet dataset.\
                                 See https://github.com/EliSchwartz/imagenet-sample-images"
                )

        # check epsilon
        if not isinstance(epsilon, float):
            raise ValueError("epsilon must be a float")
        if epsilon <= 0:
            raise ValueError("epsilon must be greater than 0")

        # check desired_confidence
        if not isinstance(desired_confidence, float):
            raise ValueError("desired_confidence must be a float")
        if not 0 <= desired_confidence <= 1:
            raise ValueError("desired_confidence must be between 0 and 1")

        # check max_iter
        if not isinstance(max_iter, int):
            raise ValueError("max_iter must be an integer")
        if max_iter <= 0:
            raise ValueError("max_iter must be greater than 0")

    def generate_adversarial_image(
        self,
        image_path: str,
        target_class: Union[str, int],
        epsilon: float = 0.01,
        desired_confidence: float = 0.9,
        max_iter: int = 100,
    ) -> Dict[str, Union[torch.Tensor, str, float]]:
        """
        Generate an adversarial image so that the model predicts the target class,
        instead of the original class. The image should be as close to the original
        image as possible to the human eye.

        Args:
            image_path (str): The path to the image file. The image should be in RGB.
            target_class (str or int): The target class name or index. If it is an integer,
                it should be between 0 and 999. If it is a string, it should be one of the
                classes in the ImageNet dataset. See
                https://github.com/EliSchwartz/imagenet-sample-images
            epsilon (float): the image $x$ is updates as
                $x = x + epsilon * \frac{d log f(x)}{dx}$, where $f(x)$ is the model's
                predicted probability for the target class. So $epsilon$ is the step
                size for the gradient ascent (think of it as the learning rate). Default
                is 0.01.
            desired_confidence (float): The desired confidence of the model for the
                target class, i.e. $f(x)$, iterations will stop when the confidence is
                greater than this value. Default is 0.9 (90%). In other words, the model
                should be at least 90% confident that the image is of the target class.
            max_iter (int): The maximum number of iterations to run the gradient ascent.
                Default is 100. If the desired confidence is not reached in this number
                of iterations, the function will return the current image.

        Returns:
            Dict: A dictionary containing the following keys:
                "adversarial_image" (torch.Tensor): The adversarial image tensor (3xHxW).
                "original_image" (torch.Tensor): The original image tensor (3xHxW).
                "original_class" (str): The original class name.
                "original_confidence" (float): The model's confidence for the original class.
                "target_class" (str): The target class name.
                "target_confidence" (float): The model's confidence for the target class.
        """
        self._check_input(
            image_path, target_class, epsilon, desired_confidence, max_iter
        )

        # preprocess the image
        orgn_image = self._preprocess_image(image_path)
        gen_image = orgn_image.clone()

        # get the original prediction
        original_class, original_confidence = self._get_prediction(orgn_image)

        # get the target class index
        if isinstance(target_class, int):
            target_class_index = target_class
        else:
            target_class_index = self.categories.index(target_class)

        # don't compute gradients for the model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # run the gradient ascent
        score = 0.0
        for i in range(max_iter):
            if score > desired_confidence:
                logging.info(
                    f"Desired confidence of {desired_confidence} reached in {i} iterations"
                )
                break

            gen_image.requires_grad_()
            prob = self.model(gen_image).squeeze(0).softmax(dim=0)
            score = prob[target_class_index].item()

            # update the gen_image
            prob[target_class_index].log().backward()  # log p(y=c|x)
            gen_image = (gen_image + gen_image.grad * epsilon).detach()

            if self.verbose:
                logging.info(
                    f"Iteration {i}: probability for {target_class} is {score}"
                )

        # get the final prediction
        target_class, target_confidence = self._get_prediction(gen_image)

        gen_image = self._denormalize_image(gen_image.squeeze(0).cpu())
        orgn_image = self._denormalize_image(orgn_image.squeeze(0).cpu())

        return {
            "adversarial_image": gen_image,
            "original_image": orgn_image,
            "original_class": self.categories[original_class],
            "original_confidence": original_confidence,
            "target_class": self.categories[target_class],
            "target_confidence": target_confidence,
        }

    def __call__(
        self,
        image_path: str,
        target_class: str,
        epsilon: float = 0.01,
        desired_confidence: float = 0.9,
        max_iter: int = 100,
    ) -> Dict[str, Union[torch.Tensor, str, float]]:
        """
        A wrapper function to call generate_adversarial_image function.
        """
        return self.generate_adversarial_image(
            image_path, target_class, epsilon, desired_confidence, max_iter
        )
