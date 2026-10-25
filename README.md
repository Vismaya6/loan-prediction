# Loan Approval Prediction — Machine Learning Pipeline

An end-to-end Machine Learning system developed collaboratively by a 4-member team to predict loan approval decisions, evaluate classifier performance, and provide decision tree and feature-level interpretability.

---

## 👥 Team Responsibilities & Architecture

The system is organized into modular pipeline phases corresponding to each member's contributions:

1. **Member 1 (`member1_data_preprocessing.py`) — Data Preprocessing**:
   - Cleans column names and object column whitespace.
   - Maps `loan_status` ("Approved": 1, "Rejected": 0).
   - One-hot encodes categorical attributes (`education`, `self_employed`).
   - Performs stratified 80/20 train/test split (`random_state=42`).
   - Normalizes numerical features with `StandardScaler` for distance/gradient models.

2. **Member 2 (`member2_model_training.py`) — Model Training**:
   - Trains three distinct supervised classifiers:
     - **Logistic Regression** (scaled data, `max_iter=1000`, `random_state=42`)
     - **K-Nearest Neighbors (KNN)** (scaled data, `n_neighbors=5`)
     - **Decision Tree** (unscaled data, `max_depth=5`, `random_state=42`)
   - Measures and benchmarks training durations for each model.

3. **Member 3 (`member3_model_evaluation.py`) — Model Evaluation**:
   - Evaluates predictions using Accuracy, Precision, Recall, and F1 Score.
   - Computes confusion matrices for false approvals vs. false rejections.
   - Renders and exports comparative bar charts (`accuracy_comparison.png`, `f1_score_comparison.png`).

4. **Member 4 (`member4_interpretability.py`) — Interpretability & Benchmarking**:
   - Extracts and ranks Logistic Regression feature coefficients.
   - Measures Decision Tree complexity metrics (depth, node count, leaf count).
   - Generates high-resolution tree visualization (`decision_tree_visualization.png`).
   - Compiles the final consolidated summary table comparing Accuracy, F1 Score, Interpretability, Complexity, and Training Time.

---

## 📁 Project Structure

```text
loan-prediction/
├── .gitignore                      # Python and environment ignore rules
├── CONTRIBUTING.md                 # Team git workflow and standards
├── README.md                       # Complete project documentation
├── requirements.txt                # Version-pinned dependencies
├── data/
│   └── README.md                   # Dataset schema and setup guidelines
├── reports/                        # Auto-generated visualization artifacts
│   ├── accuracy_comparison.png
│   ├── f1_score_comparison.png
│   └── decision_tree_visualization.png
├── tests/                          # Automated unit and integration tests
│   ├── __init__.py
│   ├── test_pipeline.py            # End-to-end pipeline tests
│   └── test_preprocessing.py       # Preprocessing unit tests
├── main.py                         # Unified pipeline runner
├── member1_data_preprocessing.py   # Member 1 module
├── member2_model_training.py       # Member 2 module
├── member3_model_evaluation.py     # Member 3 module
└── member4_interpretability.py     # Member 4 module
```

---

## 🚀 Quick Start & Setup

### 1. Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/Vismaya6/loan-prediction.git
cd loan-prediction
pip install -r requirements.txt
```

### 2. Dataset Setup
Place the loan approval dataset CSV into either the project root or the `data/` directory.
Supported filenames:
- `loan_approval_dataset (1).csv`
- `loan_approval_dataset.csv`

*(If the dataset is omitted, the pipeline raises an informative error guiding you to place the file.)*

---

## ⚙️ Running the Pipeline

### Option A: Run the Unified Pipeline (Recommended)
Executes all 4 phases sequentially and saves all report figures to `reports/`:

```bash
python main.py
```

You can also specify a custom path to your dataset:
```bash
python main.py path/to/your_dataset.csv
```

### Option B: Run Individual Member Modules
Each module is self-contained and can be executed independently:

```bash
# Phase 1: Data inspection and preprocessing
python member1_data_preprocessing.py

# Phase 2: Model training and prediction
python member2_model_training.py

# Phase 3: Performance evaluation and confusion matrices
python member3_model_evaluation.py

# Phase 4: Model interpretability and final comparison
python member4_interpretability.py
```

---

## 🧪 Running Automated Tests

The project includes unit and integration tests covering data integrity, model outputs, and end-to-end execution:

```bash
pytest -v tests/
```

---

## 📊 Empirical Evaluation Results

Evaluated on the full 4,269-applicant dataset (stratified 80/20 train/test split with 854 test instances):

| Model | Accuracy | Precision | Recall | F1 Score | Complexity | Interpretability | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (depth=5)** | **96.84%** | **98.28%** | **96.61%** | **0.9744** | Medium | High | ~0.0065s |
| **Logistic Regression** | 91.33% | 92.08% | 94.16% | 0.9311 | Low | High | ~0.0124s |
| **KNN (k=5)** | 89.46% | 90.61% | 92.66% | 0.9162 | Low | Low | ~0.0065s |

### Confusion Matrix Breakdown (854 Test Samples)
- **Decision Tree**:
  - True Rejected ($TN$): 314 | False Approved ($FP$): 9
  - False Rejected ($FN$): 18 | True Approved ($TP$): 513
  - *Risk Analysis*: Minimizes credit default risk with only 9 false approvals ($1.05\%$).
- **Logistic Regression**:
  - True Rejected ($TN$): 280 | False Approved ($FP$): 43
  - False Rejected ($FN$): 31 | True Approved ($TP$): 500
- **KNN**:
  - True Rejected ($TN$): 272 | False Approved ($FP$): 51
  - False Rejected ($FN$): 39 | True Approved ($TP$): 492

### Key Feature Determinants (Logistic Regression Coefficients)
- **`cibil_score` (+4.091)**: The single strongest positive driver of loan approval.
- **`loan_amount` (+1.321)**: Positively correlates with approved applicants in this portfolio.
- **`income_annum` (-1.595)** & **`loan_term` (-0.859)**: Longer terms negatively affect approval probability.

---

## ⚖️ Ethical & Practical Considerations

In automated loan underwriting:
* **False Approvals (Type I Error)**: Approving a borrower at risk of default directly impacts lending solvency and credit risk.
* **False Rejections (Type II Error)**: Rejecting a creditworthy applicant results in lost revenue and may unfairly exclude individuals from access to credit.
* **Fairness & Interpretability**: Black-box models (such as deep KNN embeddings) can obscure the reasoning behind rejections. Combining high-interpretability models (Logistic Regression and shallow Decision Trees) enables loan officers to audit decision factors transparently against regulatory compliance standards.