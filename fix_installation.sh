#!/bin/bash

# Fix PyTorch Lightning Compatibility Issues
echo "🔧 Fixing PyTorch Lightning compatibility issues..."

# Uninstall problematic packages
echo "🗑️  Uninstalling problematic packages..."
pip uninstall -y pytorch-lightning torchmetrics torchvision

# Install compatible versions
echo "📦 Installing compatible versions..."
pip install pytorch-lightning>=2.0.0,<3.0.0
pip install torchmetrics>=1.0.0,<2.0.0
pip install torchvision>=0.15.0

# Reinstall local torch_geometric to ensure it's properly linked
echo "📦 Reinstalling local torch_geometric..."
pip install -e ./pytorch_geometric

echo "✅ Installation fixed!"
echo ""
echo "🧪 Testing the fix..."
python3 -c "
import torch
import torch_geometric
import pytorch_lightning
import torchmetrics
import torchvision
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ PyTorch Geometric: {torch_geometric.__version__}')
print(f'✅ PyTorch Lightning: {pytorch_lightning.__version__}')
print(f'✅ TorchMetrics: {torchmetrics.__version__}')
print(f'✅ TorchVision: {torchvision.__version__}')
print('✅ All packages imported successfully!')
"

echo ""
echo "🎉 Fix completed! You can now run your experiments." 