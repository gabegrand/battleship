#!/usr/bin/env python3
"""
Enhanced script to check synchronization issues between checkpoints and results files
in the spotter benchmark experiment with detailed analysis and recommendations.
"""
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple


def load_checkpoint_questions(checkpoint_dir: str) -> Set[Tuple[str, int]]:
    """Load completed questions from checkpoint file."""
    completed_file = os.path.join(checkpoint_dir, "completed_questions.json")

    try:
        with open(completed_file, "r") as f:
            completed = json.load(f)
        return {(item["round_id"], item["question_id"]) for item in completed}
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def load_results_questions(results_file: str) -> Set[Tuple[str, int]]:
    """Load questions from results file."""
    try:
        with open(results_file, "r") as f:
            results = json.load(f)
        return {(result["round_id"], result["question_id"]) for result in results}
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def parse_config_name(config_name: str) -> Dict[str, str]:
    """Parse configuration name to extract components."""
    parts = config_name.split("_")

    llm = parts[0] if parts else "unknown"
    spotter_type = parts[1] if len(parts) > 1 else "unknown"
    use_cot = "cot" in parts

    return {
        "llm": llm,
        "spotter_type": spotter_type,
        "use_cot": use_cot,
        "config_name": config_name,
    }


def analyze_experiment_sync(experiment_dir: str) -> Dict:
    """Analyze synchronization issues in an experiment directory with enhanced details."""
    experiment_path = Path(experiment_dir)
    checkpoints_dir = experiment_path / "checkpoints"
    results_dir = experiment_path / "results"

    analysis = {
        "total_configs": 0,
        "sync_issues": [],
        "summary": {},
        "patterns": {
            "by_llm": defaultdict(list),
            "by_spotter": defaultdict(list),
            "by_cot": defaultdict(list),
            "severity": defaultdict(list),
        },
        "recommendations": [],
        "statistics": {},
    }

    if not checkpoints_dir.exists() or not results_dir.exists():
        print(f"Missing directories in {experiment_dir}")
        return analysis

    # Get all checkpoint configurations
    checkpoint_configs = {d.name for d in checkpoints_dir.iterdir() if d.is_dir()}
    results_files = {f.stem for f in results_dir.iterdir() if f.suffix == ".json"}

    analysis["total_configs"] = len(checkpoint_configs)

    total_checkpoint_questions = 0
    total_results_questions = 0
    total_missing_questions = 0
    configs_with_issues = 0
    configs_missing_files = 0

    for config_name in checkpoint_configs:
        checkpoint_path = checkpoints_dir / config_name
        results_file = results_dir / f"{config_name}.json"

        # Parse config details
        config_details = parse_config_name(config_name)

        # Load questions from both sources
        checkpoint_questions = load_checkpoint_questions(str(checkpoint_path))
        results_questions = (
            load_results_questions(str(results_file))
            if results_file.exists()
            else set()
        )

        # Calculate differences
        checkpoint_only = checkpoint_questions - results_questions
        results_only = results_questions - checkpoint_questions
        synchronized = checkpoint_questions.intersection(results_questions)

        # Calculate sync percentage
        if checkpoint_questions:
            sync_percentage = len(synchronized) / len(checkpoint_questions) * 100
        else:
            sync_percentage = 100.0 if not results_questions else 0.0

        # Determine severity
        missing_count = len(checkpoint_only)
        if missing_count == 0:
            severity = "none"
        elif missing_count < 50:
            severity = "low"
        elif missing_count < 200:
            severity = "medium"
        elif missing_count < 500:
            severity = "high"
        else:
            severity = "critical"

        config_info = {
            "config_name": config_name,
            "llm": config_details["llm"],
            "spotter_type": config_details["spotter_type"],
            "use_cot": config_details["use_cot"],
            "checkpoint_count": len(checkpoint_questions),
            "results_count": len(results_questions),
            "synchronized_count": len(synchronized),
            "checkpoint_only_count": len(checkpoint_only),
            "results_only_count": len(results_only),
            "sync_percentage": round(sync_percentage, 2),
            "severity": severity,
            "has_sync_issue": len(checkpoint_only) > 0 or len(results_only) > 0,
            "missing_results_file": not results_file.exists(),
        }

        # Update counters
        total_checkpoint_questions += len(checkpoint_questions)
        total_results_questions += len(results_questions)
        total_missing_questions += len(checkpoint_only)

        if config_info["has_sync_issue"] or config_info["missing_results_file"]:
            configs_with_issues += 1
            config_info["checkpoint_only_sample"] = list(checkpoint_only)[
                :10
            ]  # More samples
            config_info["results_only_sample"] = list(results_only)[:10]
            analysis["sync_issues"].append(config_info)

            # Categorize by patterns
            analysis["patterns"]["by_llm"][config_details["llm"]].append(config_info)
            analysis["patterns"]["by_spotter"][config_details["spotter_type"]].append(
                config_info
            )
            analysis["patterns"]["by_cot"][config_details["use_cot"]].append(
                config_info
            )
            analysis["patterns"]["severity"][severity].append(config_info)

        if config_info["missing_results_file"]:
            configs_missing_files += 1

        analysis["summary"][config_name] = config_info

    # Calculate overall statistics
    analysis["statistics"] = {
        "total_checkpoint_questions": total_checkpoint_questions,
        "total_results_questions": total_results_questions,
        "total_missing_questions": total_missing_questions,
        "configs_with_issues": configs_with_issues,
        "configs_missing_files": configs_missing_files,
        "overall_sync_percentage": round(
            total_results_questions / total_checkpoint_questions * 100, 2
        )
        if total_checkpoint_questions > 0
        else 0,
        "configs_fully_synced": analysis["total_configs"] - configs_with_issues,
    }

    # Generate recommendations
    analysis["recommendations"] = generate_recommendations(analysis)

    return analysis


def generate_recommendations(analysis: Dict) -> List[Dict]:
    """Generate actionable recommendations based on the analysis."""
    recommendations = []

    # Pattern-based recommendations
    llm_issues = analysis["patterns"]["by_llm"]
    spotter_issues = analysis["patterns"]["by_spotter"]
    cot_issues = analysis["patterns"]["by_cot"]
    severity_issues = analysis["patterns"]["severity"]

    # LLM-specific issues
    if llm_issues:
        most_problematic_llm = max(llm_issues.items(), key=lambda x: len(x[1]))
        recommendations.append(
            {
                "type": "llm_pattern",
                "priority": "high",
                "description": f"LLM '{most_problematic_llm[0]}' has {len(most_problematic_llm[1])} configurations with sync issues",
                "affected_configs": [c["config_name"] for c in most_problematic_llm[1]],
                "action": "Focus recovery efforts on this LLM first",
            }
        )

    # Spotter-specific issues
    if spotter_issues:
        most_problematic_spotter = max(spotter_issues.items(), key=lambda x: len(x[1]))
        recommendations.append(
            {
                "type": "spotter_pattern",
                "priority": "medium",
                "description": f"Spotter type '{most_problematic_spotter[0]}' has {len(most_problematic_spotter[1])} configurations with sync issues",
                "affected_configs": [
                    c["config_name"] for c in most_problematic_spotter[1]
                ],
                "action": "Review spotter implementation for potential issues",
            }
        )

    # Critical severity issues
    if "critical" in severity_issues:
        critical_configs = severity_issues["critical"]
        recommendations.append(
            {
                "type": "critical_severity",
                "priority": "urgent",
                "description": f"{len(critical_configs)} configurations have critical sync issues (>500 missing results)",
                "affected_configs": [c["config_name"] for c in critical_configs],
                "action": "These configurations likely need to be re-run completely",
            }
        )

    # Missing files
    missing_file_configs = [
        c for c in analysis["sync_issues"] if c["missing_results_file"]
    ]
    if missing_file_configs:
        recommendations.append(
            {
                "type": "missing_files",
                "priority": "high",
                "description": f"{len(missing_file_configs)} configurations are missing results files entirely",
                "affected_configs": [c["config_name"] for c in missing_file_configs],
                "action": "These configurations need to be re-run from scratch",
            }
        )

    # Recovery feasibility
    recoverable_configs = [
        c
        for c in analysis["sync_issues"]
        if not c["missing_results_file"] and c["checkpoint_only_count"] < 500
    ]
    if recoverable_configs:
        recommendations.append(
            {
                "type": "recoverable",
                "priority": "medium",
                "description": f"{len(recoverable_configs)} configurations might be recoverable through resumption",
                "affected_configs": [c["config_name"] for c in recoverable_configs],
                "action": "Try resuming these configurations with fixed checkpointing",
            }
        )

    return recommendations


def print_enhanced_analysis_report(analysis: Dict):
    """Print a comprehensive, formatted report of the analysis."""
    stats = analysis["statistics"]

    print(f"\n{'='*80}")
    print(f"{'ENHANCED SYNC ANALYSIS REPORT':^80}")
    print(f"{'='*80}")

    print(f"\n📊 OVERALL STATISTICS")
    print(f"  Total configurations: {analysis['total_configs']}")
    print(f"  Configurations with sync issues: {stats['configs_with_issues']}")
    print(f"  Configurations fully synchronized: {stats['configs_fully_synced']}")
    print(f"  Configurations missing result files: {stats['configs_missing_files']}")
    print(f"  Overall sync percentage: {stats['overall_sync_percentage']}%")
    print(f"  Total checkpoint questions: {stats['total_checkpoint_questions']:,}")
    print(f"  Total results questions: {stats['total_results_questions']:,}")
    print(f"  Total missing questions: {stats['total_missing_questions']:,}")

    if analysis["sync_issues"]:
        print(f"\n🔍 DETAILED SYNC ISSUES")
        print(f"{'Config Name':<50} {'Sync %':<8} {'Missing':<8} {'Severity':<10}")
        print(f"{'-'*80}")

        # Sort by severity and missing count
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}
        sorted_issues = sorted(
            analysis["sync_issues"],
            key=lambda x: (
                severity_order.get(x["severity"], 0),
                x["checkpoint_only_count"],
            ),
            reverse=True,
        )

        for issue in sorted_issues:
            print(
                f"{issue['config_name']:<50} {issue['sync_percentage']:>6.1f}% {issue['checkpoint_only_count']:>7} {issue['severity']:>9}"
            )

    # Pattern analysis
    print(f"\n🔍 PATTERN ANALYSIS")

    llm_issues = analysis["patterns"]["by_llm"]
    if llm_issues:
        print(f"\n  Issues by LLM:")
        for llm, configs in sorted(
            llm_issues.items(), key=lambda x: len(x[1]), reverse=True
        ):
            total_missing = sum(c["checkpoint_only_count"] for c in configs)
            print(
                f"    {llm}: {len(configs)} configs, {total_missing} missing questions"
            )

    spotter_issues = analysis["patterns"]["by_spotter"]
    if spotter_issues:
        print(f"\n  Issues by Spotter Type:")
        for spotter, configs in sorted(
            spotter_issues.items(), key=lambda x: len(x[1]), reverse=True
        ):
            total_missing = sum(c["checkpoint_only_count"] for c in configs)
            print(
                f"    {spotter}: {len(configs)} configs, {total_missing} missing questions"
            )

    cot_issues = analysis["patterns"]["by_cot"]
    if cot_issues:
        print(f"\n  Issues by CoT Usage:")
        for cot_status, configs in cot_issues.items():
            total_missing = sum(c["checkpoint_only_count"] for c in configs)
            cot_label = "with CoT" if cot_status else "without CoT"
            print(
                f"    {cot_label}: {len(configs)} configs, {total_missing} missing questions"
            )

    severity_issues = analysis["patterns"]["severity"]
    if severity_issues:
        print(f"\n  Issues by Severity:")
        for severity in ["critical", "high", "medium", "low"]:
            if severity in severity_issues:
                configs = severity_issues[severity]
                total_missing = sum(c["checkpoint_only_count"] for c in configs)
                print(
                    f"    {severity.title()}: {len(configs)} configs, {total_missing} missing questions"
                )

    # Recommendations
    if analysis["recommendations"]:
        print(f"\n💡 RECOMMENDATIONS")
        priority_order = {"urgent": 3, "high": 2, "medium": 1, "low": 0}
        sorted_recommendations = sorted(
            analysis["recommendations"],
            key=lambda x: priority_order.get(x["priority"], 0),
            reverse=True,
        )

        for i, rec in enumerate(sorted_recommendations, 1):
            priority_emoji = {
                "urgent": "🚨",
                "high": "⚠️",
                "medium": "📋",
                "low": "💡",
            }.get(rec["priority"], "📋")
            print(
                f"\n  {i}. {priority_emoji} {rec['priority'].upper()}: {rec['description']}"
            )
            print(f"     Action: {rec['action']}")
            if len(rec["affected_configs"]) <= 5:
                print(f"     Affected: {', '.join(rec['affected_configs'])}")
            else:
                print(f"     Affected: {len(rec['affected_configs'])} configurations")

    print(f"\n{'='*80}")


def export_detailed_analysis(
    analysis: Dict, output_file: str = "detailed_sync_analysis.json"
):
    """Export detailed analysis with actionable data."""

    # Create export structure
    export_data = {
        "metadata": {
            "analysis_timestamp": __import__("time").time(),
            "total_configs": analysis["total_configs"],
            "configs_with_issues": len(analysis["sync_issues"]),
        },
        "statistics": analysis["statistics"],
        "sync_issues": analysis["sync_issues"],
        "patterns": {
            "by_llm": {
                llm: len(configs)
                for llm, configs in analysis["patterns"]["by_llm"].items()
            },
            "by_spotter": {
                spotter: len(configs)
                for spotter, configs in analysis["patterns"]["by_spotter"].items()
            },
            "by_cot": {
                str(cot): len(configs)
                for cot, configs in analysis["patterns"]["by_cot"].items()
            },
            "by_severity": {
                severity: len(configs)
                for severity, configs in analysis["patterns"]["severity"].items()
            },
        },
        "recommendations": analysis["recommendations"],
        "actionable_lists": {
            "needs_complete_rerun": [
                c["config_name"]
                for c in analysis["sync_issues"]
                if c["missing_results_file"]
            ],
            "critical_issues": [
                c["config_name"]
                for c in analysis["sync_issues"]
                if c["severity"] == "critical"
            ],
            "high_issues": [
                c["config_name"]
                for c in analysis["sync_issues"]
                if c["severity"] == "high"
            ],
            "medium_issues": [
                c["config_name"]
                for c in analysis["sync_issues"]
                if c["severity"] == "medium"
            ],
            "low_issues": [
                c["config_name"]
                for c in analysis["sync_issues"]
                if c["severity"] == "low"
            ],
            "fully_synced": [
                name
                for name, info in analysis["summary"].items()
                if not info["has_sync_issue"]
            ],
        },
    }

    with open(output_file, "w") as f:
        json.dump(export_data, f, indent=2, default=str)

    return export_data


def main():
    experiment_dir = "/Users/grandg/mit/battleship_project/battleship/experiments/collaborative/spotter_benchmarks/run_2025_07_11_18_32_51"

    print(f"Analyzing experiment: {experiment_dir}")
    analysis = analyze_experiment_sync(experiment_dir)
    print_enhanced_analysis_report(analysis)

    # Export detailed analysis
    export_data = export_detailed_analysis(analysis)
    print(f"\nDetailed analysis exported to: detailed_sync_analysis.json")

    # Save original format for backward compatibility
    with open("sync_analysis_report.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"Original format saved to: sync_analysis_report.json")


if __name__ == "__main__":
    main()
