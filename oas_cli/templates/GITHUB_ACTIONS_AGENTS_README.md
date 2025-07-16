# GitHub Actions Error Analysis Multi-Agent System

This directory contains a complete multi-agent system for analyzing GitHub Actions errors and providing developer-friendly explanations and fixes.

## Overview

The system consists of three main components:

1. **Error Collector Agent** (`github-actions-error-collector.yaml`) - Extracts and structures error data from raw GitHub Actions logs
2. **Error Analyzer Agent** (`github-actions-error-analyzer.yaml`) - Analyzes structured errors and provides actionable fixes
3. **Python Workflow Script** (`../github_actions_error_workflow.py`) - Orchestrates the entire analysis process using DACP

## Agent Components

### 1. GitHub Actions Error Collector Agent
- **Purpose**: Processes raw GitHub Actions logs and extracts structured error information
- **Key Tasks**:
  - `collect_errors`: Extracts error summaries, categorizes errors, and identifies affected files
  - `extract_build_info`: Extracts build system specific information (npm, maven, docker, etc.)
- **Output**: Structured error data with severity levels, error types, and context

### 2. GitHub Actions Error Analyzer Agent
- **Purpose**: Analyzes structured error data and provides developer-friendly guidance
- **Key Tasks**:
  - `analyze_error`: Provides root cause analysis and recommended fixes
  - `generate_pr_comment`: Creates formatted markdown comments for GitHub PRs
  - `suggest_workflow_improvements`: Recommends workflow optimizations
- **Output**: Detailed analysis, actionable fixes, and developer messages

### 3. Python Workflow Script
- **Purpose**: Orchestrates the complete error analysis process using DACP
- **Key Functions**:
  - `analyze_github_actions_failure()`: Full end-to-end analysis
  - `load_agent_class()`: Dynamically loads generated agent classes
  - `print_analysis_summary()`: Human-readable results display

## Usage Examples

### Single Error Analysis

```yaml
# Input to the workflow
input:
  job_name: "build-and-test"
  workflow_name: "CI"
  repository: "example/my-react-app"
  branch: "feature/add-new-component"
  commit_sha: "a1b2c3d4e5f6789012345678901234567890abcd"
  pr_number: 42
  raw_logs: |
    npm ERR! code ERESOLVE
    npm ERR! ERESOLVE unable to resolve dependency tree
    npm ERR! Could not resolve dependency:
    npm ERR! peer react@"^18.0.0" from @testing-library/react@13.4.0
    ...
```

### Expected Output Flow

1. **Collector Agent Output**:
   ```json
   {
     "error_summary": {
       "primary_error": "npm ERR! ERESOLVE unable to resolve dependency tree",
       "error_type": "dependency_error",
       "severity": "high",
       "affected_files": ["package.json"]
     }
   }
   ```

2. **Analyzer Agent Output**:
   ```json
   {
     "analysis_result": {
       "root_cause": "React version conflict between main dependency (v17) and testing library peer dependency (v18)",
       "error_category": "dependency_problem",
       "confidence_level": "high"
     },
     "recommended_fixes": [
       {
         "fix_title": "Update React to v18",
         "fix_description": "Update React and React-DOM to version 18 to match testing library requirements",
         "fix_type": "immediate",
         "estimated_effort": "low"
       }
     ]
   }
   ```

3. **Generated PR Comment**:
   ```markdown
   ## 🚨 GitHub Actions Build Failure Analysis

   **Job:** build-and-test
   **Error:** React version conflict

   ### Root Cause
   The build failed due to a dependency conflict between React v17 and @testing-library/react v13 which requires React v18.

   ### Recommended Fix
   1. Update React to v18: `npm install react@^18.0.0 react-dom@^18.0.0`
   2. Update related dependencies if needed

   ### Impact
   This prevents the build from completing and blocks the PR from being merged.
   ```

## Sample Data

The `github-actions-sample-data.json` file contains realistic examples of common GitHub Actions failures:

- **NPM Build Failure**: Dependency conflicts
- **Python Test Failure**: Syntax errors causing test failures
- **Docker Build Failure**: Missing base images
- **Timeout Failure**: Long-running tests exceeding time limits
- **Linting Failure**: ESLint errors in TypeScript code

## Supported Error Types

The system can analyze various types of GitHub Actions failures:

- `build_failure`: Compilation or build process errors
- `test_failure`: Test execution failures
- `dependency_error`: Package dependency issues
- `syntax_error`: Code syntax problems
- `runtime_error`: Runtime execution errors
- `timeout`: Job timeout issues
- `permission_error`: Access and permission problems
- `network_error`: Network connectivity issues
- `configuration_error`: Workflow configuration problems
- `deployment_error`: Deployment process failures
- `linting_error`: Code quality and formatting issues

## Integration with GitHub Actions

To use this system in your GitHub Actions workflow:

1. **Collect Job Logs**: Capture the raw logs from failed jobs
2. **Invoke Collector**: Process logs through the collector agent
3. **Analyze Errors**: Send structured data to the analyzer agent
4. **Post Results**: Optionally post analysis as PR comments

Example GitHub Actions integration:
```yaml
- name: Analyze Build Failure
  if: failure()
  run: |
    # Capture logs and send to multi-agent system
    # Create input data and run the Python workflow
    python github_actions_error_workflow.py \
      --job-name "${{ github.job }}" \
      --workflow-name "${{ github.workflow }}" \
      --repository "${{ github.repository }}" \
      --pr-number "${{ github.event.pull_request.number }}" \
      --logs-file "build.log"
```

## Benefits

1. **Automated Error Analysis**: No need to manually investigate every failure
2. **Developer-Friendly Explanations**: Technical errors translated into actionable guidance
3. **Consistent Analysis**: Standardized approach to error categorization
4. **Proactive Fixes**: Specific recommendations with code examples
5. **Pattern Recognition**: Identify recurring issues across multiple failures
6. **Time Savings**: Reduce time spent debugging GitHub Actions failures

## Getting Started

### Step 1: Generate the Agents
```bash
# Generate the collector agent
oas init --spec oas_cli/templates/github-actions-error-collector.yaml --output ./github-actions-collector

# Generate the analyzer agent
oas init --spec oas_cli/templates/github-actions-error-analyzer.yaml --output ./github-actions-analyzer
```

### Step 2: Install Dependencies
```bash
# Install dependencies for each agent
cd github-actions-collector && pip install -r requirements.txt && cd ..
cd github-actions-analyzer && pip install -r requirements.txt && cd ..
```

### Step 3: Set Up API Keys
```bash
# Copy .env.example to .env and set your API keys in each agent directory
cp github-actions-collector/.env.example github-actions-collector/.env
cp github-actions-analyzer/.env.example github-actions-analyzer/.env

# Edit the .env files to add your OpenAI or Anthropic API keys
```

### Step 4: Run the Workflow
```bash
# Run the multi-agent workflow with sample data
python github_actions_error_workflow.py
```

## Next Steps

1. **Test with Sample Data**: The workflow includes built-in sample data for testing
2. **Customize for Your Stack**: Modify error types and analysis for your specific technology stack
3. **Integrate with GitHub**: Set up automated posting of analysis results to PRs
4. **Add Real Data**: Replace sample data with actual GitHub Actions logs
5. **Extend Analysis**: Add more error types and analysis patterns

## File Structure

```
oas_cli/templates/
├── github-actions-error-collector.yaml      # Collector agent specification
├── github-actions-error-analyzer.yaml       # Analyzer agent specification
├── github-actions-sample-data.json          # Sample test data
└── GITHUB_ACTIONS_AGENTS_README.md          # This documentation

github_actions_error_workflow.py             # Multi-agent workflow script (root directory)
```

This multi-agent system provides a comprehensive solution for GitHub Actions error analysis, helping developers quickly understand and fix build failures without having to dig through complex logs in the GitHub UI.
