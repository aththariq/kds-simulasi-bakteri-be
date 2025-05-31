# Benchmark Testing Framework

A comprehensive performance testing and regression detection system for the KDS Bacterial Simulation project.

## Overview

This benchmark framework provides automated performance validation for critical system operations, preventing performance regressions and ensuring optimal system performance at scale. The framework is designed to be developer-friendly, CI/CD integrated, and provides detailed insights into system performance characteristics.

## Architecture

### Core Components

1. **BenchmarkRunner** (`core/benchmark_runner.py`)

   - Orchestrates benchmark execution
   - Manages benchmark suites and reporting
   - Provides CI/CD integration capabilities

2. **Benchmark Decorators** (`core/benchmark_decorators.py`)

   - `@benchmark` - Standard benchmark decorator
   - `@performance_test` - Benchmark with performance constraints
   - Auto-execution with resource monitoring

3. **Performance Budgets** (`core/performance_budgets.py`)

   - Define performance thresholds
   - Automated violation detection
   - Severity-based alerting

4. **Utilities** (`core/benchmark_utils.py`)
   - Configuration and result classes
   - Resource monitoring
   - Result storage and comparison

## Quick Start

### Basic Benchmark

```python
from tests.benchmarks.core import benchmark

@benchmark(
    name="my_operation",
    iterations=10,
    categories=["database"],
    tags={"size": "medium"}
)
def benchmark_my_operation():
    # Your operation to benchmark
    result = my_expensive_operation()
    return result
```

### Performance Test with Constraints

```python
from tests.benchmarks.core import performance_test

@performance_test(
    max_execution_time=0.1,    # 100ms max
    max_memory_usage=50.0,     # 50MB max
    min_operations_per_second=1000.0,
    iterations=20
)
def benchmark_high_performance_operation():
    return fast_operation()
```

## Running Benchmarks

### Command Line

```bash
# Run all benchmarks
cd backend
python -m pytest tests/benchmarks/ -v

# Run specific benchmark file
python tests/benchmarks/test_population_benchmarks.py

# Run spatial benchmarks
python tests/benchmarks/test_spatial_benchmarks.py
```

### Programmatically

```python
from tests.benchmarks.core import BenchmarkRunner

# Initialize runner
runner = BenchmarkRunner()

# Run all suites
reports = runner.run_all_suites()

# Generate CI report
success = runner.generate_ci_report("ci_report.json")
```

## Benchmark Suites

### Population Operations (`test_population_benchmarks.py`)

Tests performance of population management operations:

- **Population Creation**: Batch creation of 1K-5K bacteria
- **Indexing Lookups**: O(1) bacteria ID lookups
- **Resistance Filtering**: Set-based resistance filtering
- **Statistics Caching**: Cached statistics calculations
- **Memory Optimization**: Large population memory efficiency

Key metrics:

- Execution time for batch operations
- Memory usage during large populations
- Cache hit rates for statistics

### Spatial Operations (`test_spatial_benchmarks.py`)

Tests spatial grid and movement operations:

- **Grid Creation**: Lazy-loaded spatial grid initialization
- **Coordinate Pooling**: Object pool efficiency
- **Bacteria Placement**: Grid placement operations
- **Movement Operations**: Position updates
- **Neighbor Search**: Spatial proximity queries
- **Memory Management**: Grid memory optimization

Key metrics:

- Grid cell creation efficiency
- Coordinate pool reuse rates
- Neighbor search performance
- Memory cleanup effectiveness

## Performance Budgets

Performance budgets define acceptable thresholds for various metrics:

```json
{
  "population_creation_1k": {
    "execution_time": {
      "threshold": 0.05,
      "operator": "less_than",
      "severity": "high"
    },
    "memory_usage": {
      "threshold": 100.0,
      "operator": "less_than",
      "severity": "medium"
    }
  }
}
```

### Budget Violations

- **Critical**: Performance regression > 50%
- **High**: Performance regression > 25%
- **Medium**: Performance regression > 10%
- **Low**: Performance regression > 5%

## CI/CD Integration

### GitHub Actions

The framework integrates with GitHub Actions for automated performance testing:

- **Pull Request Testing**: Run benchmarks on PR changes
- **Performance Budget Validation**: Fail builds on violations
- **Baseline Comparison**: Compare with main branch performance
- **Automated Reporting**: Comment PR with results

### Workflow Configuration

```yaml
# .github/workflows/benchmark-tests.yml
name: Benchmark Tests
on: [pull_request, push]
jobs:
  benchmark-backend:
    runs-on: ubuntu-latest
    steps:
      - name: Run Benchmarks
        run: python run_benchmarks.py
```

## Configuration

### Environment Variables

- `BENCHMARK_STORAGE_PATH`: Directory for benchmark results
- `BENCHMARK_BUDGETS_FILE`: Performance budgets configuration
- `BENCHMARK_ITERATIONS`: Default iteration count
- `BENCHMARK_TIMEOUT`: Default timeout in seconds

### Performance Budget Configuration

Create `performance_budgets.json`:

```json
{
  "budgets": {
    "benchmark_name": {
      "metric_name": {
        "threshold": 0.1,
        "operator": "less_than",
        "severity": "high"
      }
    }
  },
  "global_settings": {
    "enable_comparison": true,
    "baseline_comparison_threshold": 0.1
  }
}
```

## Best Practices

### Writing Benchmarks

1. **Isolate Operations**: Test single operations, not complex workflows
2. **Use Realistic Data**: Test with production-like data sizes
3. **Minimize Setup**: Keep setup outside benchmark timing
4. **Return Values**: Return meaningful metrics from benchmarks
5. **Handle Exceptions**: Gracefully handle and report errors

### Performance Optimization

1. **Profile Before Optimizing**: Use benchmarks to identify bottlenecks
2. **Set Realistic Budgets**: Based on actual requirements
3. **Monitor Trends**: Track performance over time
4. **Regression Detection**: Set up alerts for performance drops

### Resource Management

1. **Memory Tracking**: Enable memory monitoring for large operations
2. **Cleanup Resources**: Properly dispose of resources after tests
3. **Avoid Side Effects**: Don't let benchmarks affect each other
4. **Pool Resources**: Use object pools for frequently created objects

## Interpreting Results

### Benchmark Reports

Each benchmark generates detailed metrics:

```json
{
  "name": "population_creation_1k",
  "avg_execution_time": 0.045,
  "min_execution_time": 0.042,
  "max_execution_time": 0.048,
  "std_execution_time": 0.002,
  "avg_memory_usage": 12.5,
  "peak_memory_usage": 15.2,
  "success_rate": 1.0,
  "operations_per_second": 22.2
}
```

### Key Metrics

- **Execution Time**: Time to complete operation
- **Memory Usage**: RAM consumption during operation
- **Success Rate**: Percentage of successful runs
- **Operations/Second**: Throughput metric
- **Standard Deviation**: Consistency of performance

### Performance Trends

Monitor these trends over time:

- Average execution time changes
- Memory usage growth
- Success rate degradation
- Performance variance increases

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `PYTHONPATH` includes backend directory
2. **Memory Errors**: Large datasets may require more iterations
3. **Timeout Issues**: Increase timeout for slow operations
4. **CI Failures**: Check performance budget thresholds

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

runner = BenchmarkRunner()
# Detailed execution logs will be shown
```

### Manual Validation

Run individual benchmarks for debugging:

```python
from tests.benchmarks.test_population_benchmarks import PopulationBenchmarks

instance = PopulationBenchmarks()
instance.setup_method()
result = instance.benchmark_population_creation_1k()
print(f"Result: {result}")
```

## Extending the Framework

### Adding New Benchmarks

1. Create new test file in `tests/benchmarks/`
2. Use `@benchmark` or `@performance_test` decorators
3. Follow naming convention: `test_*_benchmarks.py`
4. Add to CI workflow for automation

### Custom Metrics

Extend `BenchmarkResult` for custom metrics:

```python
@dataclass
class CustomBenchmarkResult(BenchmarkResult):
    custom_metric: float = 0.0

    def calculate_custom_metric(self):
        # Your custom calculation
        pass
```

### Integration with Other Tools

- **Grafana**: Export metrics to time-series database
- **Slack/Teams**: Alert on performance regressions
- **Jira**: Create tickets for performance issues
- **APM Tools**: Integrate with application monitoring

## Future Enhancements

1. **Distributed Benchmarking**: Run benchmarks across multiple machines
2. **Historical Analysis**: Long-term performance trend analysis
3. **Automated Optimization**: AI-driven performance tuning suggestions
4. **Real-time Monitoring**: Live performance monitoring in production
5. **Benchmark Recommendations**: Suggest optimizations based on results

---

## Contributing

When adding new benchmarks or modifying existing ones:

1. Follow the existing code patterns
2. Add appropriate performance budgets
3. Update documentation
4. Test locally before submitting PR
5. Monitor CI results for performance impacts

For questions or issues, refer to the project's main documentation or create an issue in the repository.
