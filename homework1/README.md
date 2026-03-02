# Deep Learning Training and Evaluation - Four New Tasks

## Overview
This project applies deep learning training and evaluation to four new tasks using custom datasets and features. All implementations follow the `pytorch_task_v1` protocol and are implemented in PyTorch.

## Requirements
- PyTorch
- Python 3.8+
- NumPy
- Pandas

## Tasks

### Task 1: [Task Name]
- **Dataset**: [Dataset description]
- **Model**: [Model architecture]
- **Features**: [Key features]

### Task 2: [Task Name]
- **Dataset**: [Dataset description]
- **Model**: [Model architecture]
- **Features**: [Key features]

### Task 3: [Task Name]
- **Dataset**: [Dataset description]
- **Model**: [Model architecture]
- **Features**: [Key features]

### Task 4: [Task Name]
- **Dataset**: [Dataset description]
- **Model**: [Model architecture]
- **Features**: [Key features]

## Protocol
All tasks implement the `pytorch_task_v1` protocol with:
- Standardized data loading
- Unified model training pipeline
- Self-verifiable exit codes via `sys.exit(exit_code)`

## Usage
```bash
cd MLtasks/<task_id>
python task.py
```

## Exit Codes
- `0`: Success
- `1`: Error
- `2`: Invalid input
