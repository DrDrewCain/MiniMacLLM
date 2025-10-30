# GitHub Release Guide

This guide explains how to prepare and release your trained model on GitHub.

---

## Pre-Release Checklist

### ✅ What You Should Have

- [x] Trained model: `checkpoints/wikitext_medium/final/model.pt` (485MB)
- [x] Tokenizer: `data/tokenizers/wikitext_8k/` (vocab, merges, config)
- [x] Training scripts: `scripts/pretrain.py`, `scripts/continual_learn.py`, etc.
- [x] Documentation: README.md, MODEL_CARD.md, etc.
- [x] Tests: Passing test suite (122/122 tests)
- [x] License: MIT license file

### ⚠️ Important Decisions

**Should you commit the model file (485MB)?**

**Option A: Include Model in Repo** (Recommended for small models)
- ✅ Easy to download and use
- ✅ Version controlled
- ❌ Large repo size (485MB+)
- ❌ Slow clone times

**Option B: GitHub Release Asset**
- ✅ Smaller repo size
- ✅ Separate downloads
- ✅ Better for large files
- ❌ Extra step to download

**Option C: Git LFS** (Best for models)
- ✅ Version controlled
- ✅ Efficient storage
- ✅ Handles large files well
- ❌ Requires Git LFS setup

---

## Option 1: Release with Git LFS (Recommended)

### Step 1: Install Git LFS

```bash
# Mac
brew install git-lfs

# Linux
sudo apt-get install git-lfs

# Initialize
git lfs install
```

### Step 2: Track Model Files

```bash
# Track all model checkpoints
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.ckpt"

# Track tokenizer files (if large)
git lfs track "*.json" --lockable

# Add .gitattributes
git add .gitattributes
```

### Step 3: Add Files and Commit

```bash
# Add everything
git add .

# Commit
git commit -m "feat: Add trained 127M WikiText model

- Trained for 5 epochs on WikiText-2
- Final loss: 5.32
- Full continual learning system
- BPE tokenizer (8K vocab)
- Complete training and inference scripts

Model details:
- Parameters: 127,224,576
- Architecture: GQA, RoPE, SwiGLU, RMSNorm
- Training time: 2.5 hours on M3 Max
- Demonstrates continual learning without forgetting
"

# Push (may take a while for 485MB)
git push origin main
```

---

## Option 2: GitHub Release (No LFS)

### Step 1: Exclude Model from Git

```bash
# Add to .gitignore
echo "checkpoints/**/*.pt" >> .gitignore
echo "checkpoints/**/*.pth" >> .gitignore
echo "!checkpoints/README.md" >> .gitignore

# Commit without model
git add .
git commit -m "chore: Add code and documentation (model in release)"
git push origin main
```

### Step 2: Create GitHub Release

```bash
# Create release archive
cd checkpoints/wikitext_medium/final
tar -czf ~/wikitext_medium_v1.0.0.tar.gz model.pt
cd ~/Custom_ML_Agent

# Or zip
zip -r ~/wikitext_medium_v1.0.0.zip checkpoints/wikitext_medium/final/model.pt
```

### Step 3: Upload via GitHub UI

1. Go to your repo: `https://github.com/YOUR_USERNAME/MiniMacLLM`
2. Click "Releases" → "Create a new release"
3. Tag: `v1.0.0`
4. Title: `MiniMacLLM v1.0.0 - Initial Release`
5. Description:

```markdown
# MiniMacLLM v1.0.0

First release of the continual learning LLM system!

## What's Included

- 127M parameter base model (trained on WikiText-2)
- BPE tokenizer (8K vocabulary)
- Complete continual learning system (LoRA + Replay + EWC)
- Training scripts
- Inference scripts
- Documentation

## Model Stats

- **Parameters**: 127,224,576
- **Training Loss**: 5.32
- **Training Time**: 2.5 hours (M3 Max)
- **Model Size**: 485 MB
- **Architecture**: Modern LLM (GQA, RoPE, SwiGLU)

## Quick Start

```bash
# Download release
wget https://github.com/YOUR_USERNAME/MiniMacLLM/releases/download/v1.0.0/wikitext_medium_v1.0.0.tar.gz
tar -xzf wikitext_medium_v1.0.0.tar.gz

# Install dependencies
pip install -r requirements.txt

# Generate text
python scripts/generate.py \
  --model model.pt \
  --tokenizer data/tokenizers/wikitext_8k \
  --prompt "The future of AI"
```

## Files in This Release

- `wikitext_medium_v1.0.0.tar.gz` - Model checkpoint (485MB)
- Source code (zip)
- Source code (tar.gz)

See [MODEL_CARD.md](https://github.com/YOUR_USERNAME/MiniMacLLM/blob/main/MODEL_CARD.md) for full details.
```

6. Attach `wikitext_medium_v1.0.0.tar.gz` (485MB)
7. Click "Publish release"

---

## Option 3: External Hosting

Host the model elsewhere and link from GitHub:

### Google Drive / Dropbox

```bash
# Upload to Google Drive, get shareable link
# Add to README.md:
```

**Download Pre-trained Model:**
- [WikiText-2 Model (485MB)](https://drive.google.com/your-link-here)
- [Tokenizer (820KB)](https://drive.google.com/your-tokenizer-link)

```bash
# Or use gdown for Google Drive
pip install gdown
gdown https://drive.google.com/uc?id=YOUR_FILE_ID
```

---

## Preparing the Repository

### Step 1: Clean Up

```bash
# Remove unnecessary files
rm -rf __pycache__
rm -rf .pytest_cache
rm -rf *.pyc
find . -name ".DS_Store" -delete

# Update .gitignore
cat >> .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Model checkpoints (if not using LFS)
# checkpoints/**/*.pt
# checkpoints/**/*.pth

# Data
data/raw/*.txt
!data/raw/.gitkeep

# Temporary
*.log
.cache/
EOF
```

### Step 2: Update Documentation

Make sure these files are ready:

```bash
# Check all docs exist
ls -l README.md
ls -l MODEL_CARD.md
ls -l LICENSE
ls -l requirements.txt
ls -l docs/CONTINUAL_LEARNING_DEMO.md
ls -l docs/QUICK_START.md
```

### Step 3: Add Download Instructions

Update README.md with download instructions:

```markdown
## Download Pre-trained Model

### Via Git LFS (if using LFS)
```bash
git clone https://github.com/YOUR_USERNAME/MiniMacLLM.git
cd MiniMacLLM
# Model automatically downloaded with LFS
```

### Via GitHub Release
```bash
# Clone code
git clone https://github.com/YOUR_USERNAME/MiniMacLLM.git
cd MiniMacLLM

# Download model
wget https://github.com/YOUR_USERNAME/MiniMacLLM/releases/download/v1.0.0/wikitext_medium_v1.0.0.tar.gz
tar -xzf wikitext_medium_v1.0.0.tar.gz -C checkpoints/wikitext_medium/final/
```
```

---

## Final Steps

### 1. Test the Release

```bash
# Clone fresh copy
cd /tmp
git clone https://github.com/YOUR_USERNAME/MiniMacLLM.git
cd MiniMacLLM

# Install deps
pip install -r requirements.txt

# Test generation
python scripts/generate.py \
  --model checkpoints/wikitext_medium/final/model.pt \
  --tokenizer data/tokenizers/wikitext_8k \
  --prompt "Test" \
  --max_tokens 20
```

### 2. Create Announcement

Post on:
- Reddit: r/MachineLearning, r/LocalLLaMA
- Twitter/X: Share your achievement
- LinkedIn: Professional network
- Hacker News: Show HN

Example post:

```
🚀 MiniMacLLM: A 127M Continual Learning LLM

I built a language model that can learn new domains in real-time
without forgetting what it already knows!

Key features:
- 127M parameters (efficient for Apple Silicon)
- Continual learning (LoRA + Experience Replay + EWC)
- Zero catastrophic forgetting
- Add new domains with only 2.2% parameter increase

GitHub: https://github.com/YOUR_USERNAME/MiniMacLLM

Built in Python, trained on M3 Max, fully open source (MIT).

Feedback welcome!
```

### 3. Add Badges to README

```markdown
![GitHub release](https://img.shields.io/github/v/release/YOUR_USERNAME/MiniMacLLM)
![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/MiniMacLLM)
![License](https://img.shields.io/github/license/YOUR_USERNAME/MiniMacLLM)
![Tests](https://img.shields.io/badge/tests-122%2F122%20passing-brightgreen)
```

---

## Recommended: Use Git LFS

For the best experience, use **Git LFS**:

```bash
# Setup (one time)
brew install git-lfs
git lfs install

# Track model files
git lfs track "*.pt"
git lfs track "*.pth"
git add .gitattributes

# Normal git workflow
git add checkpoints/wikitext_medium/final/model.pt
git commit -m "Add trained model"
git push origin main
```

Users can then:

```bash
# Clone with LFS (model downloads automatically)
git lfs clone https://github.com/YOUR_USERNAME/MiniMacLLM.git
```

---

## Summary

**Best approach**: Git LFS + GitHub Release

1. Use Git LFS for version control
2. Create GitHub release for easy download link
3. Add clear documentation for users
4. Test the download/install process
5. Share with the community!

Your continual learning model is ready to share with the world! 🎉
