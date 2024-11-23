## Adversarial Generation Task
### Problem: Adversarial Noise
The task involves developing a program that manipulates images by adding adversarial noise. This noise is designed to trick an image classification model into misclassifying the altered image as a specified target class, regardless of the original content.

You may select any pre-trained image classification model for this task. A model from the torchvision library is recommended, but not mandatory.
The core challenge is to effectively introduce noise into the image in such a way that the model misclassifies it as the desired target class, without making the noise perceptible to a casual human viewer.

Input:
The user will provide an image and specify a target class.

Output:
The program should output an image that has been altered with adversarial noise. The altered image should be classified by the model as the target class, irrespective of the original image's content. The altered image should not be obviously different to the original.

### My Approach
My approach depends on using gradient ascent to maximize the probability of the target class. The gradient ascent is applied to the input image, and the gradient is calculated with respect to the target class. The gradient ascent is applied iteratively to the input image until the model classifies the image as the target class. The update formula is as follows:
$$
\hat{x} = x + \epsilon * \frac{d}{dx} \log p(y_{target}|x)
$$
where:
- $\hat{x}$ is the updated image
- $x$ is the original image
- $\epsilon$ is the step size
- $p(y_{target}|x)$ is the probability of the target class given the image (softmax output of the model)
- $\frac{d}{dx} \log p(y_{target}|x)$ is the gradient of the log probability of the target class with respect to the image

The user will provide an image and specify a target class and optionally specify the number of iterations, the step size, and the threshold for the probability of the target class. The program will output an image that has been altered with adversarial noise. The altered image should be classified by the model as the target class, irrespective of the original image's content. The altered image should not be obviously different from the original.

### Sample output
The example below shows the original image, the target class, and the altered image. The altered image is classified as the target class by the model with high confidence.
![Example](https://raw.githubusercontent.com/ahmedhshahin/leap_labs_task/main/example.png)

