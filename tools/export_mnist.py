import os               # Create directories, handle file paths
import struct           # Write data such as integers in the specified binary format.

import numpy as np      # Handling arrays and converting data types
import torch            # PyTorch Tensor Computation and Automatic Differentiation
import torch.nn as nn   # Constructing neural network layers and loss functions
from torch.utils.data import DataLoader # Read training data in batches.
from torchvision import datasets, transforms # Convert images into the format required by the neural network.


def save_tensors(path, tensors):
    """tensors: dict[name -> np.ndarray], Write it out in MINI format:"""
    with open(path, "wb") as f:
        f.write(b"MINI")
        f.write(struct.pack("<II", 1, len(tensors)))
        for name, arr in tensors.items():
            arr = np.ascontiguousarray(arr, dtype="<f4") # Write the metadata and data for each tensor.
            nb = name.encode("utf-8")
            f.write(struct.pack("<I", len(nb)))     # write the byte length of the name,
            f.write(nb)                             # write the name
            f.write(struct.pack("<I", arr.ndim))    # Number of array dimensions
            f.write(struct.pack(f"<{arr.ndim}I", *arr.shape)) # The length of each dimension.
            f.write(arr.tobytes())                  # The actual floating-point data of the array.


torch.manual_seed(0)    # PyTorch random seed.
tf = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to PyTorch tensor, and scale pixel values to [0, 1].
    transforms.Normalize((0.1307,), (0.3081,)),   # MNIST standard mean/variance
])

# Load the training and test sets respectively; 
# if the data is not available locally, attempt to download it to the `data` directory.
train_set = datasets.MNIST("data", train=True, download=True, transform=tf)
test_set = datasets.MNIST("data", train=False, download=True, transform=tf)

# Building a neural network
# --- Define a simple MLP model ---
model = nn.Sequential(
    nn.Flatten(),   # Input image 28 × 28 = 784, flatten into a one-dimensional vector
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 10),
)

# ---- train neural network ----
# Adam: Update model parameters based on gradients.
# CrossEntropyLoss:	Measures the discrepancy between predictions and ground-truth labels
# DataLoader: Provides 128 images and their labels at a time
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
loader = DataLoader(train_set, batch_size=128, shuffle=True)
for epoch in range(5):  # Train for 5 epochs
    model.train()
    for x, y in loader:
        opt.zero_grad()     # Clear old gradients
        loss_fn(model(x), y).backward() # Perform backward propagation and calculate the loss.
        opt.step()          # Update weights

# ---- Evaluate ----
model.eval()    # Switch the model to evaluation mode.
with torch.no_grad():   # Disable gradient calculation
    xs = torch.stack([test_set[i][0] for i in range(len(test_set))])    # Stack all the test images
    ys = torch.tensor([test_set[i][1] for i in range(len(test_set))])
    acc = (model(xs).argmax(1) == ys).float().mean().item() # Calculate the accuracy of the model on the test set.
print(f"PyTorch test accuracy: {acc:.4f}")

# ---- Export weights ----
# The shape of PyTorch's `Linear.weight` is `[out, in]`, 
# whereas our C++ convention is `[in, out]`, so a transpose is required.
os.makedirs("models", exist_ok=True)
weights = {}
linears = [m for m in model if isinstance(m, nn.Linear)]    # Get all the linear layers in the model
for i, layer in enumerate(linears, start=1):    # Enumerate the linear layers starting from 1
    weights[f"fc{i}.weight"] = layer.weight.detach().numpy().T  # Transpose the weight matrix to match the C++ convention
    weights[f"fc{i}.bias"] = layer.bias.detach().numpy()    # Get the bias vector of the layer
save_tensors("models/mnist_mlp.bin", weights)

# ---- Export reference data: first 100 test samples' inputs, labels, and PyTorch logits ----
N = 100
with torch.no_grad():
    x = xs[:N].reshape(N, -1)       #[N, 784], row-major order, consistent with C++
    logits = model(xs[:N])          # [N, 10], PyTorch logits
save_tensors("models/mnist_ref.bin", {
    "input": x.numpy(),             
    "labels": ys[:N].numpy().astype(np.float32),    # numerical labels
    "logits": logits.numpy(),       # PyTorch logits
})
print("written models/mnist_mlp.bin, models/mnist_ref.bin")
