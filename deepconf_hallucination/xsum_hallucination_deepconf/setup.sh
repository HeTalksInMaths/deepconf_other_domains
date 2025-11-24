#!/bin/bash

echo "======================================"
echo "XSum Hallucination DeepConf Setup"
echo "======================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p results
mkdir -p data
mkdir -p notebooks
echo "Directories created."

# Download XSum test set (small sample)
echo ""
echo "Downloading XSum test set sample..."
python3 << EOF
try:
    from datasets import load_dataset
    print("Downloading XSum test split (first 10 examples)...")
    dataset = load_dataset('xsum', split='test[:10]', cache_dir='./data')
    print(f"Downloaded {len(dataset)} examples successfully!")
except Exception as e:
    print(f"Warning: Could not download dataset: {e}")
    print("Dataset will be downloaded automatically when you run experiments.")
EOF

# Create __init__ files
echo ""
echo "Creating package structure..."
touch src/__init__.py

echo ""
echo "======================================"
echo "Setup complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. (Optional) Download Google hallucination annotations:"
echo "   git clone https://github.com/google-research-datasets/xsum_hallucination_annotations.git data/hallucination_annotations"
echo ""
echo "3. Run a quick test:"
echo "   python src/run_experiments.py"
echo ""
echo "4. Read the documentation:"
echo "   - INSTANTIATION.md for experimental design"
echo "   - README.md for usage guide"
echo ""
echo "Happy experimenting!"
