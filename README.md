# docai

**Version:** 0.0.1  
**License:** [MIT](LICENSE)

---

## Overview

**docai** is a document processing system designed to fine-tune a pretrained LayoutLMv3 model on your custom dataset. LayoutLMv3 is a transformer-based model built for structured document understanding, capable of leveraging both textual content and layout information to perform tasks like token classification on documents (e.g., invoices, forms, claims). The system uses OCR via pytesseract to extract text and spatial data from images and formats custom JSON annotations for training and inference.

---

## Features

- **Custom Fine-Tuning:**  
  Fine-tune a pretrained LayoutLMv3 model on your own annotated dataset to adapt it to your specific document processing tasks.

- **OCR Integration:**  
  Automatically extract text and bounding boxes from images using pytesseract.

- **Custom Data Loader:**  
  Leverage a custom data loader that transforms JSON annotations into training-ready examples.

- **Training & Evaluation Pipeline:**  
  Easily train, evaluate, and save the best performing model with a configurable training loop.

- **Inference Pipeline:**  
  Run inference on new documents to visualize extracted bounding boxes, labels, and probabilities.

- **Modular & Extensible:**  
  The code is structured in a modular way to allow easy customization and extension of functionality.

- **Command Line Interface (CLI):**  
  Use Typer for a flexible CLI to configure and run training, evaluation, and inference.

---

## Installation

### Prerequisites

- **Python:** Version 3.10 (recommended)
- **Package Manager:** pip (or conda if you prefer a virtual environment)

### Setting Up Your Environment

1. Clone the Repository:

    ```bash
    git clone https://github.com/dmdaksh/docai.git
    cd docai

2. (Optional) Create a Conda Environment:

    Use the provided Makefile command to set up a conda environment:

    ```bash
    make create_environment
    ```

    Then activate the environment:

    ```bash
    conda activate docai
    ```

3. Install Dependencies:
    Upgrade pip and install required packages:

    ```bash
    make requirements
    ```

4. Code Formatting and Linting (Optional):
    Format Code:

    ```bash
    make format
    ```

    Lint Code:

    ```bash
    make lint
    ```

## Dataset Preparation

Your training data should be provided in a JSON file. Each entry in the JSON should include:

- **file_name:** The path to the image file.
- **annotations:** A list of annotation dictionaries for the document. Each annotation should contain:
  - **text:** The token or word extracted from the document.
  - **box:** The bounding box coordinates in `[x1, y1, x2, y2]` format.
  - **label:** The classification label for the token.

The utility function `train_data_format` in `docai/utils/utils.py` converts the raw JSON data into the format required for training. Ensure that your JSON conforms to this structure.

## Training the Model

The training process fine-tunes the pretrained LayoutLMv3 model on your custom dataset. The training script (`docai/training/main.py`) uses Typer to enable configurable training parameters via the command line.

### Running the Training Script

You can run the training with default settings:

```bash
python -m docai.training.main
```

To customize training parameters, specify options such as the number of epochs, batch size, learning rate, training JSON file path, and model save path:

```bash
python -m docai.training.main \
  --epochs 5 \
  --batch_size 4 \
  --learning_rate 3e-5 \
  --training_json path/to/your_training_data.json \
  --model_save_path path/to/best_model.bin
```

#### Key Parameters

- `--epochs`: Number of training epochs.
- `--batch_size`: Batch size for training.
- `--learning_rate`: Learning rate for the AdamW optimizer.
- `--training_json`: Path to your JSON file containing training annotations.
- `--model_save_path`: Path where the best model weights will be saved.

During training, the script will:

1. Load and preprocess your custom dataset.
2. Fine-tune the pretrained LayoutLMv3 model using the provided training loop.
3. Evaluate the model after each epoch.
4. Save the best model based on training loss and periodic checkpoints.
5. Save the training loss history to a NumPy file (`loss_list.npy`).

## Inference

After training, use the inference pipeline to process new documents. The inference code in `docai/inference/inference.py` loads the fine-tuned model, processes an input image, and displays the image with overlaid bounding boxes, predicted labels, and probabilities.

### Running Inference

Example command to run inference:

```bash
python -m docai.inference.inference --image_path path/to/image.png --model_path path/to/best_model.bin
```

## Contributing

Contributions are welcome! Please follow these guidelines:

- **Fork the Repository:**
  - Create your feature branch from `main`.

- **Coding Standards:**
  - Adhere to PEP8 guidelines. Use `make lint` and `make format` to ensure code consistency.

- **Commit Messages:**
  - Write clear, descriptive commit messages.

- **Pull Request:**
  - Open a pull request detailing your changes. For major changes, open an issue first to discuss your ideas.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- **LayoutLMv3:**
  - A transformer-based model for structured document understanding. More details at [Hugging Face](https://huggingface.co).

- **pytesseract:**
  - An OCR tool used for text extraction. Visit [pytesseract on PyPI](https://pypi.org/project/pytesseract/) for more information.

- **Open-Source Community:**
  - Thanks to all contributors and maintainers of the libraries and tools used in this project.
