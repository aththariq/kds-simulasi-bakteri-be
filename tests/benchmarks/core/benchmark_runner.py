"""
Benchmark runner for orchestrating benchmark execution, reporting, and CI/CD integration.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
import importlib
import inspect

from .benchmark_utils import BenchmarkResult, BenchmarkStorage
from .benchmark_decorators import benchmark
from .performance_budgets import BudgetValidator, BudgetViolation


@dataclass
class BenchmarkSuite:
    """A collection of related benchmark tests."""
    name: str
    benchmarks: Dict[str, Callable] = field(default_factory=dict)
    description: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    setup: Optional[Callable] = None
    teardown: Optional[Callable] = None
    
    def add_benchmark(self, func: Callable, name: Optional[str] = None):
        """Add a benchmark function to this suite."""
        benchmark_name = name or func.__name__
        self.benchmarks[benchmark_name] = func
    
    def run_setup(self):
        """Run suite setup if defined."""
        if self.setup:
            if asyncio.iscoroutinefunction(self.setup):
                return asyncio.run(self.setup())
            else:
                return self.setup()
    
    def run_teardown(self):
        """Run suite teardown if defined."""
        if self.teardown:
            if asyncio.iscoroutinefunction(self.teardown):
                return asyncio.run(self.teardown())
            else:
                return self.teardown()


@dataclass
class BenchmarkReport:
    """Complete benchmark execution report."""
    suite_name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    violations: List[BudgetViolation] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'suite_name': self.suite_name,
            'results': [r.to_dict() for r in self.results],
            'violations': [v.to_dict() for v in self.violations],
            'summary': self.summary,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp
        }
    
    def to_json(self) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def save_to_file(self, filepath: Union[str, Path]):
        """Save report to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write(self.to_json())


class BenchmarkRunner:
    """Main benchmark runner for executing and managing benchmark tests."""
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        budgets_file: Optional[str] = None,
        enable_budgets: bool = True
    ):
        self.storage = BenchmarkStorage(storage_path or "benchmark_results")
        self.budget_validator = BudgetValidator(budgets_file) if enable_budgets else None
        self.suites: Dict[str, BenchmarkSuite] = {}
        self.default_suite = BenchmarkSuite("default")
        
    def add_suite(self, suite: BenchmarkSuite):
        """Add a benchmark suite."""
        self.suites[suite.name] = suite
    
    def create_suite(self, name: str, description: Optional[str] = None) -> BenchmarkSuite:
        """Create and add a new benchmark suite."""
        suite = BenchmarkSuite(name=name, description=description)
        self.add_suite(suite)
        return suite
    
    def add_benchmark(
        self, 
        func: Callable, 
        suite_name: str = "default",
        benchmark_name: Optional[str] = None
    ):
        """Add a benchmark to a suite."""
        if suite_name not in self.suites:
            if suite_name == "default":
                self.suites["default"] = self.default_suite
            else:
                self.create_suite(suite_name)
        
        self.suites[suite_name].add_benchmark(func, benchmark_name)
    
    def discover_benchmarks(self, module_path: str) -> int:
        """Discover benchmark functions from a module."""
        try:
            module = importlib.import_module(module_path)
            discovered_count = 0
            
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj) and hasattr(obj, '__wrapped__'):
                    # Check if it's a benchmarked function
                    if hasattr(obj, '_benchmark_config'):
                        self.add_benchmark(obj)
                        discovered_count += 1
                elif name.startswith('benchmark_') and inspect.isfunction(obj):
                    # Auto-wrap functions with benchmark_ prefix
                    benchmarked_func = benchmark()(obj)
                    self.add_benchmark(benchmarked_func)
                    discovered_count += 1
            
            return discovered_count
            
        except ImportError as e:
            print(f"Warning: Could not import module {module_path}: {e}")
            return 0
    
    def run_benchmark(self, func: Callable, *args, **kwargs) -> BenchmarkResult:
        """Run a single benchmark function."""
        try:
            if asyncio.iscoroutinefunction(func):
                return asyncio.run(func(*args, **kwargs))
            else:
                return func(*args, **kwargs)
        except Exception as e:
            # Create a failed result
            from .benchmark_utils import BenchmarkConfig
            config = BenchmarkConfig(name=func.__name__)
            result = BenchmarkResult(name=func.__name__, config=config)
            result.errors.append(str(e))
            result.success_rate = 0.0
            return result
    
    def run_suite(
        self, 
        suite_name: str,
        benchmark_filter: Optional[str] = None,
        validate_budgets: bool = True
    ) -> BenchmarkReport:
        """Run all benchmarks in a suite."""
        if suite_name not in self.suites:
            raise ValueError(f"Suite '{suite_name}' not found")
        
        suite = self.suites[suite_name]
        report = BenchmarkReport(suite_name=suite_name)
        start_time = time.perf_counter()
        
        try:
            # Run suite setup
            suite.run_setup()
            
            # Filter benchmarks if specified
            benchmarks_to_run = suite.benchmarks
            if benchmark_filter:
                benchmarks_to_run = {
                    name: func for name, func in suite.benchmarks.items()
                    if benchmark_filter in name
                }
            
            # Execute benchmarks
            for benchmark_name, benchmark_func in benchmarks_to_run.items():
                print(f"Running benchmark: {benchmark_name}")
                
                try:
                    result = self.run_benchmark(benchmark_func)
                    report.results.append(result)
                    
                    # Validate against budgets
                    if validate_budgets and self.budget_validator:
                        violations = self.budget_validator.validate_result(result)
                        report.violations.extend(violations)
                        
                        if violations:
                            print(f"  ⚠️  Budget violations detected: {len(violations)}")
                            for violation in violations:
                                print(f"     {violation.formatted_message}")
                        else:
                            print(f"  ✅ No budget violations")
                    
                    print(f"  Completed in {result.avg_execution_time:.4f}s")
                    
                except Exception as e:
                    print(f"  ❌ Failed: {str(e)}")
            
        finally:
            # Run suite teardown
            suite.run_teardown()
            
            # Calculate summary
            report.execution_time = time.perf_counter() - start_time
            report.summary = self._generate_summary(report)
        
        return report
    
    def run_all_suites(
        self,
        benchmark_filter: Optional[str] = None,
        validate_budgets: bool = True
    ) -> Dict[str, BenchmarkReport]:
        """Run all benchmark suites."""
        reports = {}
        
        for suite_name in self.suites:
            print(f"\n🚀 Running benchmark suite: {suite_name}")
            print("=" * 50)
            
            report = self.run_suite(suite_name, benchmark_filter, validate_budgets)
            reports[suite_name] = report
            
            # Print suite summary
            print(f"\n📊 Suite Summary:")
            print(f"  Benchmarks run: {len(report.results)}")
            print(f"  Total time: {report.execution_time:.2f}s")
            print(f"  Budget violations: {len(report.violations)}")
            
            if report.violations:
                critical_count = sum(1 for v in report.violations if v.threshold.severity.value == 'critical')
                high_count = sum(1 for v in report.violations if v.threshold.severity.value == 'high')
                if critical_count > 0:
                    print(f"  🚨 Critical violations: {critical_count}")
                if high_count > 0:
                    print(f"  ⚠️  High severity violations: {high_count}")
        
        return reports
    
    def compare_with_baseline(
        self, 
        suite_name: str,
        baseline_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compare current results with baseline."""
        current_report = self.run_suite(suite_name, validate_budgets=False)
        
        if baseline_file and Path(baseline_file).exists():
            with open(baseline_file, 'r') as f:
                baseline_data = json.load(f)
            baseline_results = {
                r['name']: r for r in baseline_data.get('results', [])
            }
        else:
            # Use storage to find baseline results
            baseline_results = {}
            for result in current_report.results:
                baseline = self.storage.get_latest_result(result.name)
                if baseline and baseline.timestamp != result.timestamp:
                    baseline_results[result.name] = baseline.to_dict()
        
        comparisons = {}
        for result in current_report.results:
            if result.name in baseline_results:
                baseline_data = baseline_results[result.name]
                baseline_result = BenchmarkResult.from_dict(baseline_data)
                
                comparison = self.storage.compare_results(result, baseline_result)
                comparisons[result.name] = comparison
        
        return {
            'current_report': current_report.to_dict(),
            'comparisons': comparisons,
            'summary': self._generate_comparison_summary(comparisons)
        }
    
    def generate_ci_report(
        self,
        output_file: str = "benchmark_ci_report.json",
        fail_on_violations: bool = True
    ) -> bool:
        """Generate CI/CD compatible report and return success status."""
        reports = self.run_all_suites()
        
        # Aggregate all results and violations
        all_results = []
        all_violations = []
        
        for report in reports.values():
            all_results.extend(report.results)
            all_violations.extend(report.violations)
        
        # Determine CI status
        success = True
        if fail_on_violations and all_violations:
            critical_violations = [v for v in all_violations if v.threshold.severity.value == 'critical']
            high_violations = [v for v in all_violations if v.threshold.severity.value == 'high']
            
            if critical_violations or high_violations:
                success = False
        
        # Generate CI report
        ci_report = {
            'success': success,
            'timestamp': time.time(),
            'summary': {
                'total_benchmarks': len(all_results),
                'total_violations': len(all_violations),
                'critical_violations': len([v for v in all_violations if v.threshold.severity.value == 'critical']),
                'high_violations': len([v for v in all_violations if v.threshold.severity.value == 'high']),
                'average_execution_time': sum(r.avg_execution_time for r in all_results) / len(all_results) if all_results else 0,
                'total_execution_time': sum(r.execution_time for r in reports.values())
            },
            'suites': {name: report.to_dict() for name, report in reports.items()},
            'violations': [v.to_dict() for v in all_violations]
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(ci_report, f, indent=2)
        
        return success
    
    def _generate_summary(self, report: BenchmarkReport) -> Dict[str, Any]:
        """Generate summary statistics for a report."""
        if not report.results:
            return {}
        
        execution_times = [r.avg_execution_time for r in report.results]
        memory_usage = [r.avg_memory_usage for r in report.results if r.memory_usage]
        success_rates = [r.success_rate for r in report.results]
        
        return {
            'total_benchmarks': len(report.results),
            'successful_benchmarks': sum(1 for r in report.results if r.success_rate > 0.5),
            'average_execution_time': sum(execution_times) / len(execution_times),
            'total_execution_time': sum(execution_times),
            'average_memory_usage': sum(memory_usage) / len(memory_usage) if memory_usage else 0,
            'average_success_rate': sum(success_rates) / len(success_rates),
            'budget_violations': len(report.violations),
            'critical_violations': len([v for v in report.violations if v.threshold.severity.value == 'critical'])
        }
    
    def _generate_comparison_summary(self, comparisons: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of baseline comparisons."""
        if not comparisons:
            return {}
        
        improvements = 0
        regressions = 0
        
        for comparison in comparisons.values():
            if comparison['execution_time']['improvement']:
                improvements += 1
            else:
                regressions += 1
        
        return {
            'total_comparisons': len(comparisons),
            'improvements': improvements,
            'regressions': regressions,
            'improvement_rate': improvements / len(comparisons) if comparisons else 0
        }
    
    def create_default_budgets(self):
        """Create default performance budgets."""
        if self.budget_validator:
            self.budget_validator.create_default_budgets()
    
    def list_benchmarks(self) -> Dict[str, List[str]]:
        """List all available benchmarks by suite."""
        return {
            suite_name: list(suite.benchmarks.keys())
            for suite_name, suite in self.suites.items()
        } 