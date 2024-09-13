# Cat and Dog Sketch Classifier

This is a machine learning model trained to differentiate between sketches of cats and dogs. It was built as part of a learning project to understand how AI models work and how to train them.

## Model Details
- **[Hugging Face Model](https://huggingface.co/ljt019/cat_dog_classifier_cnn)**

- **Model Type**: Convolutional Neural Network (CNN)
- **Training Data**: Quick, Draw! dataset (cat and dog sketches only)
- **License**: MIT License
- **Supported Tasks**: Image Classification

## Usage

To use this model, you can follow these steps:

1. **Load the Model**:
    ```python
    import torch
    from model import SimpleCNN

    # Load the model
    model = SimpleCNN()
    model.load_state_dict(torch.load('cat_dog_classifier.bin'))
    model.eval()
    ```

2. **Predict an Image**:
    ```python
    from PIL import Image
    import numpy as np
    import torch

    def predict_image(model, image):
        # Preprocess the image
        if isinstance(image, Image.Image):
            image = image.resize((28, 28)).convert('L')
            image = np.array(image).astype('float32') / 255.0
        elif isinstance(image, np.ndarray):
            if image.shape != (28, 28):
                image = Image.fromarray(image).resize((28, 28)).convert('L')
                image = np.array(image).astype('float32') / 255.0
        else:
            raise ValueError("Image must be a PIL Image or NumPy array.")
        image = image.reshape(1, 1, 28, 28)
        image_tensor = torch.tensor(image).to(device)
        # Get prediction
        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output.data, 1)
        return 'cat' if predicted.item() == 0 else 'dog'

    # Example usage
    image = Image.open('path/to/your/image.png')
    prediction = predict_image(model, image)
    print(prediction)
    ```

## Training the Model

To train the model yourself, use the provided `train_cat_dog_classifier.py` script.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
