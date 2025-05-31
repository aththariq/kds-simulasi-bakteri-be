"""
Performance budgets system for establishing performance thresholds and automated validation.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from enum import Enum
from .benchmark_utils import BenchmarkResult


class BudgetMetric(Enum):
    """Performance metrics that can be budgeted."""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    OPERATIONS_PER_SECOND = "operations_per_second"
    SUCCESS_RATE = "success_rate"
    ERROR_COUNT = "error_count"


class BudgetOperator(Enum):
    """Comparison operators for budget validation."""
    LESS_THAN = "less_than"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    GREATER_THAN = "greater_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    EQUAL = "equal"
    NOT_EQUAL = "not_equal"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BudgetThreshold:
    """A single performance threshold within a budget."""
    metric: BudgetMetric
    operator: BudgetOperator
    value: float
    severity: AlertSeverity = AlertSeverity.MEDIUM
    message: Optional[str] = None
    
    def validate(self, result: BenchmarkResult) -> Optional['BudgetViolation']:
        """Validate this threshold against a benchmark result."""
        actual_value = self._extract_metric_value(result)
        
        if actual_value is None:
            return None
        
        violated = self._check_violation(actual_value)
        
        if violated:
            return BudgetViolation(
                threshold=self,
                actual_value=actual_value,
                benchmark_name=result.name,
                timestamp=time.time()
            )
        
        return None
    
    def _extract_metric_value(self, result: BenchmarkResult) -> Optional[float]:
        """Extract the metric value from a benchmark result."""
        metric_extractors = {
            BudgetMetric.EXECUTION_TIME: lambda r: r.avg_execution_time,
            BudgetMetric.MEMORY_USAGE: lambda r: r.avg_memory_usage,
            BudgetMetric.OPERATIONS_PER_SECOND: lambda r: r.operations_per_second(),
            BudgetMetric.SUCCESS_RATE: lambda r: r.success_rate,
            BudgetMetric.ERROR_COUNT: lambda r: float(len(r.errors))
        }
        
        extractor = metric_extractors.get(self.metric)
        if extractor:
            try:
                return extractor(result)
            except Exception:
                return None
        
        return None
    
    def _check_violation(self, actual_value: float) -> bool:
        """Check if the actual value violates this threshold."""
        comparisons = {
            BudgetOperator.LESS_THAN: lambda a, t: a >= t,
            BudgetOperator.LESS_THAN_OR_EQUAL: lambda a, t: a > t,
            BudgetOperator.GREATER_THAN: lambda a, t: a <= t,
            BudgetOperator.GREATER_THAN_OR_EQUAL: lambda a, t: a < t,
            BudgetOperator.EQUAL: lambda a, t: abs(a - t) > 1e-6,
            BudgetOperator.NOT_EQUAL: lambda a, t: abs(a - t) <= 1e-6
        }
        
        comparison_func = comparisons.get(self.operator)
        if comparison_func:
            return comparison_func(actual_value, self.value)
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert threshold to dictionary."""
        return {
            'metric': self.metric.value,
            'operator': self.operator.value,
            'value': self.value,
            'severity': self.severity.value,
            'message': self.message
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BudgetThreshold':
        """Create threshold from dictionary."""
        return cls(
            metric=BudgetMetric(data['metric']),
            operator=BudgetOperator(data['operator']),
            value=data['value'],
            severity=AlertSeverity(data.get('severity', 'medium')),
            message=data.get('message')
        )


@dataclass
class BudgetViolation:
    """Represents a performance budget violation."""
    threshold: BudgetThreshold
    actual_value: float
    benchmark_name: str
    timestamp: float
    
    @property
    def formatted_message(self) -> str:
        """Get formatted violation message."""
        if self.threshold.message:
            return self.threshold.message
        
        return (
            f"{self.benchmark_name}: {self.threshold.metric.value} "
            f"({self.actual_value:.4f}) violates threshold "
            f"({self.threshold.operator.value} {self.threshold.value})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert violation to dictionary."""
        return {
            'threshold': self.threshold.to_dict(),
            'actual_value': self.actual_value,
            'benchmark_name': self.benchmark_name,
            'timestamp': self.timestamp,
            'formatted_message': self.formatted_message
        }


@dataclass
class PerformanceBudget:
    """A collection of performance thresholds for a benchmark."""
    name: str
    thresholds: List[BudgetThreshold] = field(default_factory=list)
    description: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    enabled: bool = True
    
    def add_threshold(
        self,
        metric: Union[BudgetMetric, str],
        operator: Union[BudgetOperator, str],
        value: float,
        severity: Union[AlertSeverity, str] = AlertSeverity.MEDIUM,
        message: Optional[str] = None
    ):
        """Add a threshold to this budget."""
        if isinstance(metric, str):
            metric = BudgetMetric(metric)
        if isinstance(operator, str):
            operator = BudgetOperator(operator)
        if isinstance(severity, str):
            severity = AlertSeverity(severity)
        
        threshold = BudgetThreshold(
            metric=metric,
            operator=operator,
            value=value,
            severity=severity,
            message=message
        )
        self.thresholds.append(threshold)
    
    def validate(self, result: BenchmarkResult) -> List[BudgetViolation]:
        """Validate all thresholds against a benchmark result."""
        if not self.enabled:
            return []
        
        violations = []
        for threshold in self.thresholds:
            violation = threshold.validate(result)
            if violation:
                violations.append(violation)
        
        return violations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert budget to dictionary."""
        return {
            'name': self.name,
            'thresholds': [t.to_dict() for t in self.thresholds],
            'description': self.description,
            'categories': self.categories,
            'enabled': self.enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceBudget':
        """Create budget from dictionary."""
        budget = cls(
            name=data['name'],
            description=data.get('description'),
            categories=data.get('categories', []),
            enabled=data.get('enabled', True)
        )
        
        for threshold_data in data.get('thresholds', []):
            budget.thresholds.append(BudgetThreshold.from_dict(threshold_data))
        
        return budget


class BudgetValidator:
    """Validates benchmark results against performance budgets."""
    
    def __init__(self, budgets_file: Optional[str] = None):
        self.budgets_file = Path(budgets_file) if budgets_file else Path("performance_budgets.json")
        self.budgets: Dict[str, PerformanceBudget] = {}
        self.load_budgets()
    
    def load_budgets(self):
        """Load budgets from configuration file."""
        if self.budgets_file.exists():
            try:
                with open(self.budgets_file, 'r') as f:
                    data = json.load(f)
                
                self.budgets.clear()
                for budget_data in data.get('budgets', []):
                    budget = PerformanceBudget.from_dict(budget_data)
                    self.budgets[budget.name] = budget
                    
            except Exception as e:
                print(f"Warning: Could not load budgets from {self.budgets_file}: {e}")
    
    def save_budgets(self):
        """Save budgets to configuration file."""
        try:
            self.budgets_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'budgets': [budget.to_dict() for budget in self.budgets.values()],
                'updated': time.time()
            }
            
            with open(self.budgets_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save budgets to {self.budgets_file}: {e}")
    
    def add_budget(self, budget: PerformanceBudget):
        """Add a budget to the validator."""
        self.budgets[budget.name] = budget
        self.save_budgets()
    
    def remove_budget(self, name: str):
        """Remove a budget from the validator."""
        if name in self.budgets:
            del self.budgets[name]
            self.save_budgets()
    
    def get_budget(self, name: str) -> Optional[PerformanceBudget]:
        """Get a budget by name."""
        return self.budgets.get(name)
    
    def validate_result(self, result: BenchmarkResult) -> List[BudgetViolation]:
        """Validate a benchmark result against all applicable budgets."""
        violations = []
        
        # Check for exact name match
        if result.name in self.budgets:
            budget = self.budgets[result.name]
            violations.extend(budget.validate(result))
        
        # Check for category matches
        for budget in self.budgets.values():
            if budget.name != result.name and budget.categories:
                if any(cat in result.config.categories for cat in budget.categories):
                    violations.extend(budget.validate(result))
        
        return violations
    
    def create_default_budgets(self):
        """Create default performance budgets for common operations."""
        
        # Population operations budget
        population_budget = PerformanceBudget(
            name="population_operations",
            description="Performance budget for population-related operations",
            categories=["population", "bacteria"]
        )
        population_budget.add_threshold(
            BudgetMetric.EXECUTION_TIME, BudgetOperator.LESS_THAN, 1.0,
            AlertSeverity.HIGH, "Population operations should complete within 1 second"
        )
        population_budget.add_threshold(
            BudgetMetric.MEMORY_USAGE, BudgetOperator.LESS_THAN, 100.0,
            AlertSeverity.MEDIUM, "Population operations should use less than 100MB"
        )
        
        # Spatial operations budget
        spatial_budget = PerformanceBudget(
            name="spatial_operations",
            description="Performance budget for spatial grid operations",
            categories=["spatial", "grid"]
        )
        spatial_budget.add_threshold(
            BudgetMetric.EXECUTION_TIME, BudgetOperator.LESS_THAN, 0.1,
            AlertSeverity.HIGH, "Spatial operations should complete within 100ms"
        )
        spatial_budget.add_threshold(
            BudgetMetric.OPERATIONS_PER_SECOND, BudgetOperator.GREATER_THAN, 1000.0,
            AlertSeverity.MEDIUM, "Spatial operations should achieve >1000 ops/sec"
        )
        
        # WebSocket operations budget
        websocket_budget = PerformanceBudget(
            name="websocket_operations",
            description="Performance budget for WebSocket communication",
            categories=["websocket", "communication"]
        )
        websocket_budget.add_threshold(
            BudgetMetric.EXECUTION_TIME, BudgetOperator.LESS_THAN, 0.05,
            AlertSeverity.HIGH, "WebSocket operations should complete within 50ms"
        )
        websocket_budget.add_threshold(
            BudgetMetric.SUCCESS_RATE, BudgetOperator.GREATER_THAN_OR_EQUAL, 0.99,
            AlertSeverity.CRITICAL, "WebSocket operations should have >99% success rate"
        )
        
        # Rendering operations budget
        rendering_budget = PerformanceBudget(
            name="rendering_operations",
            description="Performance budget for visualization rendering",
            categories=["rendering", "visualization"]
        )
        rendering_budget.add_threshold(
            BudgetMetric.EXECUTION_TIME, BudgetOperator.LESS_THAN, 0.016,  # 60fps
            AlertSeverity.HIGH, "Rendering should maintain 60fps (16ms per frame)"
        )
        rendering_budget.add_threshold(
            BudgetMetric.MEMORY_USAGE, BudgetOperator.LESS_THAN, 50.0,
            AlertSeverity.MEDIUM, "Rendering should use less than 50MB"
        )
        
        # Memory operations budget
        memory_budget = PerformanceBudget(
            name="memory_operations",
            description="Performance budget for memory management operations",
            categories=["memory", "cleanup"]
        )
        memory_budget.add_threshold(
            BudgetMetric.EXECUTION_TIME, BudgetOperator.LESS_THAN, 0.1,
            AlertSeverity.MEDIUM, "Memory operations should complete within 100ms"
        )
        memory_budget.add_threshold(
            BudgetMetric.SUCCESS_RATE, BudgetOperator.EQUAL, 1.0,
            AlertSeverity.HIGH, "Memory operations should have 100% success rate"
        )
        
        # Add all budgets
        for budget in [population_budget, spatial_budget, websocket_budget, rendering_budget, memory_budget]:
            self.add_budget(budget)
    
    def get_violations_summary(self, violations: List[BudgetViolation]) -> Dict[str, Any]:
        """Get a summary of budget violations."""
        if not violations:
            return {"status": "passed", "violation_count": 0}
        
        severity_counts = {}
        for violation in violations:
            severity = violation.threshold.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Determine overall status based on highest severity
        status = "failed"
        if AlertSeverity.CRITICAL.value in severity_counts:
            status = "critical"
        elif AlertSeverity.HIGH.value in severity_counts:
            status = "failed"
        elif AlertSeverity.MEDIUM.value in severity_counts:
            status = "warning"
        
        return {
            "status": status,
            "violation_count": len(violations),
            "severity_counts": severity_counts,
            "violations": [v.to_dict() for v in violations]
        } 