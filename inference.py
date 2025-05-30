import argparse
from classes.Network import Network
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Run inference on an image.")
    parser.add_argument(
        "--image-path", type=str, required=True, help="Path to the input image"
    )
    args = parser.parse_args()

    # Prepare the image for inference
    # Load the image and convert to grayscale, resize to 28x28 pixels
    # and convert to numpy array
    image = Image.open(args.image_path).convert("L")
    image = image.resize((28, 28))
    image = np.array(image)

    network = Network()

    # Load the model from the json file
    network.load_model("./model/model_structure_epoch_5_accuracy__95.json")

    # Run and return
    return network.inference(image)


if __name__ == "__main__":
    main()
