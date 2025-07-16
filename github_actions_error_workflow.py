#!/usr/bin/env python3
"""
GitHub Actions Error Analysis Multi-Agent Workflow

This script demonstrates how to use the GitHub Actions error collector and analyzer agents
together to provide comprehensive error analysis and developer-friendly explanations.

Usage:
    python github_actions_error_workflow.py [--debug] [--verbose]

Options:
    --debug     Enable debug mode to see raw LLM outputs and intermediate results
    --verbose   Enable verbose DACP logging to see prompts and responses

Prerequisites:
    1. Generate the agents from YAML specs:
       oas init --spec oas_cli/templates/github-actions-error-collector.yaml --output ./oas_cli/templates/temp/collector
       oas init --spec oas_cli/templates/github-actions-error-analyzer.yaml --output ./oas_cli/templates/temp/analyzer

    2. Install dependencies in each agent directory:
       cd oas_cli/templates/temp/collector && pip install -r requirements.txt && cd ../../..
       cd oas_cli/templates/temp/analyzer && pip install -r requirements.txt && cd ../../..

    3. Set up .env files with your API keys in each directory:
       # Create .env files in each agent directory
       echo "OPENAI_API_KEY=your_openai_api_key_here" > oas_cli/templates/temp/collector/.env
       echo "OPENAI_API_KEY=your_openai_api_key_here" > oas_cli/templates/temp/analyzer/.env
"""

import json
import importlib.util
from pathlib import Path
from typing import Dict, Any


def load_agent_class(agent_dir: str, class_name: str):
    """Load an agent class from a directory."""
    agent_path = Path(agent_dir) / "agent.py"
    if not agent_path.exists():
        raise FileNotFoundError(f"Agent file not found at {agent_path}")

    spec = importlib.util.spec_from_file_location("agent_module", agent_path)
    if spec is None:
        raise ImportError(f"Could not find spec for agent module at {agent_path}")
    agent_module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Spec for agent module at {agent_path} has no loader")
    spec.loader.exec_module(agent_module)

    return getattr(agent_module, class_name)


def load_sample_data() -> Dict[str, Any]:
    """Load sample GitHub Actions failure data."""
    sample_data_path = Path("oas_cli/templates/github-actions-sample-data.json")

    if sample_data_path.exists():
        with open(sample_data_path, "r") as f:
            return json.load(f)
    else:
        # Fallback sample data if file doesn't exist
        return {
            "sample_failures": [
                {
                    "name": "npm_build_failure",
                    "description": "Common npm build failure due to missing dependency",
                    "data": {
                        "job_name": "build-and-test",
                        "workflow_name": "CI",
                        "repository": "example/my-react-app",
                        "branch": "feature/add-new-component",
                        "commit_sha": "a1b2c3d4e5f6789012345678901234567890abcd",
                        "pr_number": 42,
                        "job_step": "Install dependencies",
                        "build_system": "npm",
                        "raw_logs": """2024-01-15T10:30:00.000Z ##[group]Run npm ci
2024-01-15T10:30:00.123Z npm WARN deprecated react-scripts@5.0.1: This package is deprecated.
2024-01-15T10:30:05.456Z npm ERR! code ERESOLVE
2024-01-15T10:30:05.456Z npm ERR! ERESOLVE unable to resolve dependency tree
2024-01-15T10:30:05.456Z npm ERR!
2024-01-15T10:30:05.456Z npm ERR! While resolving: my-react-app@1.0.0
2024-01-15T10:30:05.456Z npm ERR! Found: react@17.0.2
2024-01-15T10:30:05.456Z npm ERR! node_modules/react
2024-01-15T10:30:05.456Z npm ERR!   react@"^17.0.2" from the root project
2024-01-15T10:30:05.456Z npm ERR!
2024-01-15T10:30:05.456Z npm ERR! Could not resolve dependency:
2024-01-15T10:30:05.456Z npm ERR! peer react@"^18.0.0" from @testing-library/react@13.4.0
2024-01-15T10:30:05.456Z npm ERR! node_modules/@testing-library/react
2024-01-15T10:30:05.456Z npm ERR!   @testing-library/react@"^13.4.0" from the root project
2024-01-15T10:30:05.456Z npm ERR!
2024-01-15T10:30:05.456Z npm ERR! Fix the upstream dependency conflict, or retry
2024-01-15T10:30:05.456Z npm ERR! this command with --force, or --legacy-peer-deps
2024-01-15T10:30:05.456Z npm ERR! to accept an incorrect (and potentially broken) dependency resolution.
2024-01-15T10:30:05.456Z ##[error]Process completed with exit code 1.""",
                    },
                }
            ]
        }


def analyze_github_actions_failure(
    collector_agent,
    analyzer_agent,
    orchestrator,
    failure_data: Dict[str, Any],
    debug_mode: bool = False,
) -> Dict[str, Any]:
    """
    Complete GitHub Actions error analysis workflow.

    Args:
        collector_agent: The error collector agent instance
        analyzer_agent: The error analyzer agent instance
        orchestrator: DACP orchestrator
        failure_data: GitHub Actions failure data

    Returns:
        Complete analysis results including PR comment if applicable
    """

    print(f"\n🔍 Analyzing GitHub Actions failure: {failure_data['job_name']}")
    print(f"📁 Repository: {failure_data['repository']}")
    print(f"🔀 Workflow: {failure_data['workflow_name']}")
    if failure_data.get("pr_number"):
        print(f"🔃 PR: #{failure_data['pr_number']}")

    # Step 1: Collect and structure error data
    print("\n📊 Step 1: Collecting and structuring error data...")

    # Only pass the parameters that collect_errors expects
    collection_input = {
        "task": "collect_errors",
        "job_name": failure_data["job_name"],
        "workflow_name": failure_data["workflow_name"],
        "raw_logs": failure_data["raw_logs"],
        "repository": failure_data["repository"],
    }

    # Add optional parameters if they exist
    if "job_step" in failure_data:
        collection_input["job_step"] = failure_data["job_step"]
    if "branch" in failure_data:
        collection_input["branch"] = failure_data["branch"]
    if "commit_sha" in failure_data:
        collection_input["commit_sha"] = failure_data["commit_sha"]
    if "pr_number" in failure_data:
        collection_input["pr_number"] = failure_data["pr_number"]

    if debug_mode:
        print("\n🔍 DEBUG: Collection input data:")
        print(json.dumps(collection_input, indent=2, default=str))

    collection_result = orchestrator.send_message(
        "github-actions-collector", collection_input
    )

    if debug_mode:
        print("\n🔍 DEBUG: Raw collection result:")
        print(json.dumps(collection_result, indent=2, default=str))

    if "error" in collection_result:
        print(f"❌ Error collection failed: {collection_result['error']}")
        return {"error": "Failed to collect error data"}

    print("✅ Error data collected successfully")
    print(
        f"   - Error type: {collection_result.get('error_summary', {}).get('error_type', 'unknown')}"
    )
    print(
        f"   - Severity: {collection_result.get('error_summary', {}).get('severity', 'unknown')}"
    )

    # Step 2: Extract build information
    print("\n🔧 Step 2: Extracting build information...")

    build_result = orchestrator.send_message(
        "github-actions-collector",
        {
            "task": "extract_build_info",
            "raw_logs": failure_data["raw_logs"],
            "build_system": failure_data.get("build_system", "other"),
        },
    )

    if debug_mode:
        print("\n🔍 DEBUG: Raw build info result:")
        print(json.dumps(build_result, indent=2, default=str))

    if "error" in build_result:
        print(f"⚠️  Build info extraction failed: {build_result['error']}")
        build_result = {"build_commands": [], "dependencies": []}
    else:
        print("✅ Build information extracted")
        print(f"   - Commands: {len(build_result.get('build_commands', []))}")
        print(f"   - Dependencies: {len(build_result.get('dependencies', []))}")

    # Step 3: Analyze errors and provide fixes
    print("\n🧠 Step 3: Analyzing errors and generating fixes...")

    analysis_input = {
        "task": "analyze_error",
        "error_summary": collection_result.get("error_summary", {}),
        "job_context": collection_result.get("job_context", {}),
        "log_statistics": collection_result.get("log_statistics", {}),
        "additional_context": {
            "build_commands": build_result.get("build_commands", []),
            "dependencies": build_result.get("dependencies", []),
            "environment_info": build_result.get("environment_info", {}),
        },
    }

    analysis_result = orchestrator.send_message(
        "github-actions-analyzer", analysis_input
    )

    if debug_mode:
        print("\n🔍 DEBUG: Raw analysis result:")
        print(json.dumps(analysis_result, indent=2, default=str))

    if "error" in analysis_result:
        print(f"❌ Error analysis failed: {analysis_result['error']}")
        return {"error": "Failed to analyze error"}

    print("✅ Error analysis completed")
    print(
        f"   - Root cause: {analysis_result.get('analysis_result', {}).get('root_cause', 'Unknown')[:100]}..."
    )
    print(
        f"   - Confidence: {analysis_result.get('analysis_result', {}).get('confidence_level', 'unknown')}"
    )
    print(f"   - Fixes suggested: {len(analysis_result.get('recommended_fixes', []))}")

    # Step 4: Generate PR comment if applicable
    pr_comment_result = None
    if failure_data.get("pr_number"):
        print("\n💬 Step 4: Generating PR comment...")

        pr_comment_input = {
            "task": "generate_pr_comment",
            "analysis_result": analysis_result.get("analysis_result", {}),
            "recommended_fixes": analysis_result.get("recommended_fixes", []),
            "developer_message": analysis_result.get("developer_message", {}),
            "pr_number": failure_data["pr_number"],
            "repository": failure_data["repository"],
            "job_name": failure_data["job_name"],
            "workflow_name": failure_data["workflow_name"],
        }

        pr_comment_result = orchestrator.send_message(
            "github-actions-analyzer", pr_comment_input
        )

        if debug_mode:
            print("\n🔍 DEBUG: Raw PR comment result:")
            print(json.dumps(pr_comment_result, indent=2, default=str))

        if "error" in pr_comment_result:
            print(f"⚠️  PR comment generation failed: {pr_comment_result['error']}")
            pr_comment_result = None
        else:
            print("✅ PR comment generated")

    return {
        "error_collection": collection_result,
        "build_information": build_result,
        "error_analysis": analysis_result,
        "pr_comment": pr_comment_result,
        "workflow_summary": {
            "total_processing_steps": 4 if failure_data.get("pr_number") else 3,
            "agents_used": ["github-actions-collector", "github-actions-analyzer"],
            "confidence_score": analysis_result.get("analysis_result", {}).get(
                "confidence_level", "unknown"
            ),
            "success": True,
        },
    }


def print_analysis_summary(results: Dict[str, Any]):
    """Print a human-readable summary of the analysis results."""

    print("\n" + "=" * 60)
    print("📋 GITHUB ACTIONS ERROR ANALYSIS SUMMARY")
    print("=" * 60)

    if "error" in results:
        print(f"❌ Analysis failed: {results['error']}")
        return

    # Error Summary
    error_summary = results.get("error_collection", {}).get("error_summary", {})
    print("\n🚨 ERROR DETAILS:")
    print(f"   Type: {error_summary.get('error_type', 'unknown')}")
    print(f"   Severity: {error_summary.get('severity', 'unknown')}")
    print(f"   Primary Error: {error_summary.get('primary_error', 'Not specified')}")

    # Analysis Results
    analysis = results.get("error_analysis", {}).get("analysis_result", {})
    print("\n🧠 ANALYSIS:")
    print(f"   Root Cause: {analysis.get('root_cause', 'Not determined')}")
    print(f"   Category: {analysis.get('error_category', 'unknown')}")
    print(f"   Confidence: {analysis.get('confidence_level', 'unknown')}")

    # Recommended Fixes
    fixes = results.get("error_analysis", {}).get("recommended_fixes", [])
    print(f"\n🔧 RECOMMENDED FIXES ({len(fixes)}):")
    for i, fix in enumerate(fixes[:3], 1):  # Show first 3 fixes
        print(f"   {i}. {fix.get('fix_title', 'Untitled fix')}")
        print(f"      Effort: {fix.get('estimated_effort', 'unknown')}")
        print(f"      Type: {fix.get('fix_type', 'unknown')}")

    if len(fixes) > 3:
        print(f"   ... and {len(fixes) - 3} more fixes")

    # Developer Message
    dev_message = results.get("error_analysis", {}).get("developer_message", {})
    if dev_message.get("summary"):
        print("\n👨‍💻 DEVELOPER SUMMARY:")
        print(f"   {dev_message['summary']}")

    # PR Comment
    if results.get("pr_comment"):
        print("\n💬 PR COMMENT GENERATED:")
        comment = results["pr_comment"].get("pr_comment", "")
        print(f"   Length: {len(comment)} characters")
        print(
            f"   Type: {results['pr_comment'].get('comment_metadata', {}).get('comment_type', 'unknown')}"
        )

    print("\n" + "=" * 60)


def main():
    """Main workflow execution."""

    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(
        description="GitHub Actions Error Analysis Multi-Agent Workflow"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode to see raw LLM outputs"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    print("🚀 GitHub Actions Error Analysis Multi-Agent Workflow")
    print("=" * 60)

    # Check if agent directories exist
    collector_dir = "./oas_cli/templates/temp/collector"
    analyzer_dir = "./oas_cli/templates/temp/analyzer"

    if not Path(collector_dir).exists() or not Path(analyzer_dir).exists():
        print("❌ Agent directories not found!")
        print("\nPlease generate the agents first:")
        print(
            f"   oas init --spec oas_cli/templates/github-actions-error-collector.yaml --output {collector_dir}"
        )
        print(
            f"   oas init --spec oas_cli/templates/github-actions-error-analyzer.yaml --output {analyzer_dir}"
        )
        print("\nThen install dependencies in each directory:")
        print(
            f"   cd {collector_dir} && pip install -r requirements.txt && cd ../../.."
        )
        print(f"   cd {analyzer_dir} && pip install -r requirements.txt && cd ../../..")
        return

    try:
        # Load environment variables from agent directories
        print("🔑 Loading environment variables...")
        from dotenv import load_dotenv
        import os

        # Load .env files from each agent directory
        collector_env_path = Path(collector_dir) / ".env"
        analyzer_env_path = Path(analyzer_dir) / ".env"

        if collector_env_path.exists():
            load_dotenv(collector_env_path)
            print(f"✅ Loaded environment from {collector_env_path}")
        else:
            print(f"⚠️  No .env file found at {collector_env_path}")

        if analyzer_env_path.exists():
            load_dotenv(analyzer_env_path)
            print(f"✅ Loaded environment from {analyzer_env_path}")
        else:
            print(f"⚠️  No .env file found at {analyzer_env_path}")

        # Check if API key is available
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("❌ No API key found in environment variables")
            print("Please ensure your .env files contain OPENAI_API_KEY=your_key_here or ANTHROPIC_API_KEY=your_key_here")
            return

        print("✅ API key found in environment")

        # Load the agent classes (names based on the YAML agent.name fields)
        print("📥 Loading agent classes...")
        GithubActionsErrorCollectorAgent = load_agent_class(
            collector_dir, "GithubActionsErrorCollectorAgent"
        )
        GithubActionsErrorAnalyzerAgent = load_agent_class(
            analyzer_dir, "GithubActionsErrorAnalyzerAgent"
        )

        # Set up DACP orchestrator with optional verbose logging
        from dacp.orchestrator import Orchestrator

        orchestrator = Orchestrator()

        if args.verbose:
            print("🔊 Enabling verbose DACP logging...")
            # Set DACP logging to DEBUG level
            import logging

            logging.getLogger("dacp").setLevel(logging.DEBUG)

        # Initialize agents
        print("🤖 Initializing agents...")
        collector_agent = GithubActionsErrorCollectorAgent(
            "github-actions-collector", orchestrator
        )
        analyzer_agent = GithubActionsErrorAnalyzerAgent(
            "github-actions-analyzer", orchestrator
        )

        # Load sample data
        print("📋 Loading sample failure data...")
        sample_data = load_sample_data()
        failure_example = sample_data["sample_failures"][0]["data"]  # Use first example

        # Run the complete analysis workflow
        results = analyze_github_actions_failure(
            collector_agent,
            analyzer_agent,
            orchestrator,
            failure_example,
            debug_mode=args.debug,
        )

        # Print results summary
        print_analysis_summary(results)

        # Optionally save detailed results to file
        output_file = "github_actions_analysis_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: {output_file}")

    except ImportError as e:
        print(f"❌ Failed to import agent classes: {e}")
        print(
            "Make sure the agents have been generated and their dependencies installed."
        )
    except Exception as e:
        print(f"❌ Workflow execution failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
