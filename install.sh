#!/bin/bash

# JCTA GraphGym Installation Script
# This script installs the local torch_geometric version and other dependencies

echo "🚀 Installing JCTA GraphGym with local torch_geometric..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9+ first."
    exit 1
fi

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install pip first."
    exit 1
fi

# Create virtual environment (optional)
read -p "🤔 Do you want to create a virtual environment? (y/n): " create_venv
if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv graphgym-venv
    source graphgym-venv/bin/activate
    echo "✅ Virtual environment created and activated."
fi

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (CPU version by default)
echo "🔥 Installing PyTorch..."
pip install torch>=2.0.0

# Ask for CUDA support
read -p "🤔 Do you want to install PyTorch with CUDA support? (y/n): " cuda_support
if [[ $cuda_support == "y" || $cuda_support == "Y" ]]; then
    echo "🎯 Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

# Install local torch_geometric
echo "📦 Installing local torch_geometric..."
if [ -d "pytorch_geometric" ]; then
    pip install -e ./pytorch_geometric
    echo "✅ Local torch_geometric installed successfully."
else
    echo "❌ pytorch_geometric directory not found!"
    exit 1
fi

# Install other dependencies
echo "📦 Installing other dependencies..."
pip install -r requirements.txt

# Install PyTorch Lightning and related packages with compatible versions
echo "📦 Installing PyTorch Lightning and related packages..."
pip install pytorch-lightning>=2.0.0,<3.0.0
pip install torchmetrics>=1.0.0,<2.0.0
pip install torchvision>=0.15.0

# Install optional PyTorch Geometric extensions
read -p "🤔 Do you want to install optional PyTorch Geometric extensions? (y/n): " install_extensions
if [[ $install_extensions == "y" || $install_extensions == "Y" ]]; then
    echo "📦 Installing PyTorch Geometric extensions..."
    
    # Detect PyTorch version
    pytorch_version=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "🔍 Detected PyTorch version: $pytorch_version"
        
        # Extract major.minor version
        version_parts=(${pytorch_version//./ })
        pytorch_major_minor="${version_parts[0]}.${version_parts[1]}"
        
        echo "📦 Installing extensions for PyTorch $pytorch_major_minor..."
        
        # Install based on PyTorch version
        if [[ $pytorch_major_minor == "2.7" ]]; then
            if [[ $cuda_support == "y" || $cuda_support == "Y" ]]; then
                pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
            else
                pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cpu.html
            fi
        elif [[ $pytorch_major_minor == "2.6" ]]; then
            if [[ $cuda_support == "y" || $cuda_support == "Y" ]]; then
                pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu118.html
            else
                pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
            fi
        else
            echo "⚠️  PyTorch version $pytorch_major_minor detected. Please install extensions manually."
            echo "   Visit: https://data.pyg.org/whl/ for available versions"
        fi
    else
        echo "❌ Could not detect PyTorch version. Please install extensions manually."
    fi
fi

# Test installation
echo "🧪 Testing installation..."
python3 -c "
import torch
import torch_geometric
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ PyTorch Geometric version: {torch_geometric.__version__}')
print('✅ Installation successful!')
"

echo ""
echo "🎉 Installation completed!"
echo ""
echo "📚 Next steps:"
echo "1. Navigate to the GraphGym directory: cd pytorch_geometric/graphgym"
echo "2. Run experiments:"
echo "   - GCN: python main.py --cfg configs/pyg/gcn_node.yaml"
echo "   - GraphSAGE: python main.py --cfg configs/pyg/graphsage_node.yaml"
echo "   - GAT: python main.py --cfg configs/pyg/gat_node.yaml"
echo ""
echo "📖 For more information, see the README.md file." 