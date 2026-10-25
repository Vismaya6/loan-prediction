# Dataset Directory

Place your loan approval dataset in this directory or in the project root.

### Supported Filenames:
- `loan_approval_dataset (1).csv`
- `loan_approval_dataset.csv`

### Expected Schema:
The dataset should contain the following columns:
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `loan_id` | Integer | Unique identifier for loan application |
| `no_of_dependents` | Integer | Number of dependents |
| `education` | Categorical | `Graduate` or `Not Graduate` |
| `self_employed` | Categorical | `Yes` or `No` |
| `income_annum` | Numeric | Annual applicant income |
| `loan_amount` | Numeric | Requested loan amount |
| `loan_term` | Integer | Term of loan in years |
| `cibil_score` | Numeric | Credit score (300 - 900) |
| `residential_assets_value` | Numeric | Total value of residential assets |
| `commercial_assets_value` | Numeric | Total value of commercial assets |
| `luxury_assets_value` | Numeric | Total value of luxury assets |
| `bank_asset_value` | Numeric | Liquid bank assets |
| `loan_status` | Categorical | `Approved` or `Rejected` |

*Note: Never commit private, proprietary, or sensitive customer financial data to public version control.*
