# Contributing Guidelines

Thank you for contributing to the Loan Approval Prediction project!

## Collaboration Workflow

1. **Feature Branching**:
   - Always branch off the latest `main`:
     ```bash
     git checkout main
     git pull origin main
     git checkout -b feature/<feature-name>
     ```
   - Keep branch names descriptive (e.g. `feature/member-X-<task>`).

2. **Commit Conventions**:
   - Follow standard conventional commits:
     - `feat: <description>` for new functional features
     - `fix: <description>` for bug or conflict fixes
     - `test: <description>` for automated tests
     - `docs: <description>` for documentation improvements

3. **Pull Request Protocol**:
   - Push your feature branch:
     ```bash
     git push -u origin feature/<feature-name>
     ```
   - Open a PR into `main` using the Pull Request template.
   - Ensure all automated tests (`pytest`) pass before requesting a review.
   - Do not force-push (`--force`) to shared branches.

4. **Verification Before Submitting**:
   - Always run the pipeline locally before pushing:
     ```bash
     python main.py
     ```
