# AI Manufacturing Defect Detection & Analytics Platform

This project is an AI Manufacturing Defect Detection & Analytics Platform developed for an internship application at General Motors.

## Project Structure

- `app/`: Core application logic, API endpoints, and backend services.
- `dashboard/`: User interface and data visualization dashboards.
- `models/`: Machine learning model architectures, training, and evaluation scripts.
- `notebooks/`: Jupyter notebooks for exploratory data analysis (EDA) and prototyping.
- `data/`: Data directory containing raw and processed data.
  - `raw/`: Immutable, original data files (e.g., images of components, sensor logs).
  - `processed/`: Cleaned and preprocessed data ready for modeling.
- `tests/`: Unit and integration tests.
- `docs/`: Project documentation.
- `scripts/`: Utility scripts for data fetching, deployment, database migrations, etc.

## Setup Instructions

### 1. Prerequisites

Ensure you have Python 3.8+ installed on your system.

### 2. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to isolate project dependencies.

#### On Windows:
```powershell
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\activate
```

#### On macOS/Linux:
```bash
# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

### 3. Install Dependencies

Once the virtual environment is activated, install the required packages:

```bash
pip install -r requirements.txt
```
