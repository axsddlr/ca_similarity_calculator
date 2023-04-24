# Cross-Attention Similarity Calculator

This script calculates the cross-attention similarity between two or more models (in PyTorch format or SafeTensors format). The models' cross-attention weights are compared to find the similarity between them.

## Requirements

- Python 3.7 or higher
- PyTorch 1.7 or higher
- SafeTensors library

Install the required libraries using the following command:

```bash
pip install torch safetensors
```

## Usage

Run the script using the following command:

```bash
python calc.py <base_model_path> <model_path1> [<model_path2> ...]
```

- `<base_model_path>`: The path to the base model file (either .pt or .safetensors format)
- `<model_path1>`: The path to the first model file to compare with the base model (either .pt or .safetensors format)
- `<model_path2>` (optional): Additional model file paths to compare with the base model (either .pt or .safetensors format)

## Example

Assuming you have a base model in model_base.pt, and two other models model_1.pt and model_2.safetensors, run the script as follows:

```bash
python calc.py model_base.pt model_1.pt model_2.safetensors
```

## Output

The script will output the similarity between the base model and the other models as a percentage. Higher values indicate greater similarity between the models.

```bash
base: model_base.pt [12345678]

model_1.pt [abcdef12] - 96.27%
model_2.safetensors [3456789a] - 98.46%
```

## License
