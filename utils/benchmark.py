import asyncio
import time
import statistics
import json
import csv
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx
import numpy as np
import matplotlib.pyplot as plt
import psutil
import logging

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Data class for storing benchmark results."""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    median_response_time: float
    percentile_95: float
    percentile_99: float
    requests_per_second: float
    error_rate: float
    memory_usage_start: float
    memory_usage_end: float
    cpu_usage_avg: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class LoadTester:
    """Load testing utility for API endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[BenchmarkResult] = []
    
    async def run_concurrent_requests(
        self,
        endpoint: str,
        method: str = "GET",
        payload: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        concurrent_users: int = 10,
        requests_per_user: int = 10,
        ramp_up_time: float = 0
    ) -> BenchmarkResult:
        """Run concurrent load test on an endpoint."""
        
        test_name = f"{method}_{endpoint.replace('/', '_')}_c{concurrent_users}_r{requests_per_user}"
        logger.info(f"Starting load test: {test_name}")
        
        # Initialize metrics
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        cpu_samples = []
        
        response_times = []
        successful_requests = 0
        failed_requests = 0
        
        async def make_request(session: httpx.AsyncClient, user_id: int, request_id: int) -> float:
            """Make individual request and return response time."""
            nonlocal successful_requests, failed_requests
            
            request_start = time.perf_counter()
            try:
                if method.upper() == "GET":
                    response = await session.get(
                        f"{self.base_url}{endpoint}",
                        headers=headers,
                        timeout=30.0
                    )
                elif method.upper() == "POST":
                    response = await session.post(
                        f"{self.base_url}{endpoint}",
                        json=payload,
                        headers=headers,
                        timeout=30.0
                    )
                elif method.upper() == "PUT":
                    response = await session.put(
                        f"{self.base_url}{endpoint}",
                        json=payload,
                        headers=headers,
                        timeout=30.0
                    )
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                request_time = time.perf_counter() - request_start
                
                if response.status_code < 400:
                    successful_requests += 1
                else:
                    failed_requests += 1
                    logger.warning(f"Request failed with status {response.status_code}")
                
                return request_time
                
            except Exception as e:
                failed_requests += 1
                logger.error(f"Request failed: {str(e)}")
                return time.perf_counter() - request_start
        
        async def user_session(user_id: int) -> List[float]:
            """Simulate a user making multiple requests."""
            user_response_times = []
            
            # Ramp-up delay
            if ramp_up_time > 0:
                await asyncio.sleep(ramp_up_time * user_id / concurrent_users)
            
            async with httpx.AsyncClient() as session:
                for request_id in range(requests_per_user):
                    cpu_samples.append(psutil.cpu_percent())
                    response_time = await make_request(session, user_id, request_id)
                    user_response_times.append(response_time)
                    
                    # Small delay between requests to simulate real usage
                    await asyncio.sleep(0.1)
            
            return user_response_times
        
        # Run concurrent users
        tasks = [user_session(user_id) for user_id in range(concurrent_users)]
        user_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect all response times
        for user_times in user_results:
            if isinstance(user_times, list):
                response_times.extend(user_times)
        
        total_time = time.perf_counter() - start_time
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Calculate statistics
        total_requests = concurrent_users * requests_per_user
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            median_response_time = statistics.median(response_times)
            percentile_95 = np.percentile(response_times, 95)
            percentile_99 = np.percentile(response_times, 99)
        else:
            avg_response_time = min_response_time = max_response_time = 0
            median_response_time = percentile_95 = percentile_99 = 0
        
        requests_per_second = total_requests / total_time if total_time > 0 else 0
        error_rate = (failed_requests / total_requests) * 100 if total_requests > 0 else 0
        cpu_usage_avg = statistics.mean(cpu_samples) if cpu_samples else 0
        
        result = BenchmarkResult(
            test_name=test_name,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_time=total_time,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            median_response_time=median_response_time,
            percentile_95=percentile_95,
            percentile_99=percentile_99,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            memory_usage_start=start_memory,
            memory_usage_end=end_memory,
            cpu_usage_avg=cpu_usage_avg,
            timestamp=datetime.now()
        )
        
        self.results.append(result)
        logger.info(f"Load test completed: {test_name}")
        logger.info(f"RPS: {requests_per_second:.2f}, Error Rate: {error_rate:.2f}%")
        
        return result
    
    def run_simulation_benchmark(
        self,
        simulation_params: Dict[str, Any],
        concurrent_simulations: int = 5,
        api_key: Optional[str] = None
    ) -> BenchmarkResult:
        """Benchmark simulation creation and execution."""
        
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        
        return asyncio.run(
            self.run_concurrent_requests(
                endpoint="/api/simulation/",
                method="POST",
                payload=simulation_params,
                headers=headers,
                concurrent_users=concurrent_simulations,
                requests_per_user=1,
                ramp_up_time=1.0
            )
        )
    
    def generate_benchmark_report(self, output_file: str = "benchmark_report.json"):
        """Generate comprehensive benchmark report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_tests": len(self.results),
            "summary": self._calculate_summary_stats(),
            "detailed_results": [result.to_dict() for result in self.results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Benchmark report saved to {output_file}")
        return report
    
    def _calculate_summary_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics across all tests."""
        if not self.results:
            return {}
        
        total_requests = sum(r.total_requests for r in self.results)
        total_successful = sum(r.successful_requests for r in self.results)
        total_failed = sum(r.failed_requests for r in self.results)
        
        avg_rps = statistics.mean([r.requests_per_second for r in self.results])
        avg_response_time = statistics.mean([r.avg_response_time for r in self.results])
        avg_error_rate = statistics.mean([r.error_rate for r in self.results])
        
        return {
            "total_requests": total_requests,
            "total_successful": total_successful,
            "total_failed": total_failed,
            "overall_success_rate": (total_successful / total_requests) * 100 if total_requests > 0 else 0,
            "average_rps": avg_rps,
            "average_response_time": avg_response_time,
            "average_error_rate": avg_error_rate
        }


class PerformanceBenchmark:
    """Performance benchmarking for simulation algorithms."""
    
    def __init__(self):
        self.algorithm_results: Dict[str, List[Dict[str, Any]]] = {}
    
    def benchmark_simulation_algorithm(
        self,
        algorithm_func: Callable,
        test_cases: List[Dict[str, Any]],
        iterations: int = 10
    ) -> Dict[str, Any]:
        """Benchmark a simulation algorithm with various test cases."""
        
        algorithm_name = algorithm_func.__name__
        results = []
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"Benchmarking {algorithm_name} - Test case {i+1}/{len(test_cases)}")
            
            case_results = []
            for iteration in range(iterations):
                start_time = time.perf_counter()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                try:
                    # Run the algorithm
                    result = algorithm_func(**test_case)
                    
                    end_time = time.perf_counter()
                    end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    
                    case_results.append({
                        'execution_time': end_time - start_time,
                        'memory_used': end_memory - start_memory,
                        'success': True
                    })
                    
                except Exception as e:
                    logger.error(f"Algorithm failed: {str(e)}")
                    case_results.append({
                        'execution_time': 0,
                        'memory_used': 0,
                        'success': False,
                        'error': str(e)
                    })
            
            # Calculate statistics for this test case
            successful_runs = [r for r in case_results if r['success']]
            if successful_runs:
                execution_times = [r['execution_time'] for r in successful_runs]
                memory_usages = [r['memory_used'] for r in successful_runs]
                
                case_summary = {
                    'test_case_id': i,
                    'test_parameters': test_case,
                    'iterations': iterations,
                    'successful_runs': len(successful_runs),
                    'success_rate': len(successful_runs) / iterations * 100,
                    'avg_execution_time': statistics.mean(execution_times),
                    'min_execution_time': min(execution_times),
                    'max_execution_time': max(execution_times),
                    'std_execution_time': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                    'avg_memory_usage': statistics.mean(memory_usages),
                    'max_memory_usage': max(memory_usages) if memory_usages else 0
                }
            else:
                case_summary = {
                    'test_case_id': i,
                    'test_parameters': test_case,
                    'iterations': iterations,
                    'successful_runs': 0,
                    'success_rate': 0,
                    'avg_execution_time': 0,
                    'min_execution_time': 0,
                    'max_execution_time': 0,
                    'std_execution_time': 0,
                    'avg_memory_usage': 0,
                    'max_memory_usage': 0
                }
            
            results.append(case_summary)
        
        # Store results
        if algorithm_name not in self.algorithm_results:
            self.algorithm_results[algorithm_name] = []
        
        algorithm_summary = {
            'algorithm_name': algorithm_name,
            'benchmark_timestamp': datetime.now().isoformat(),
            'test_cases': results,
            'overall_stats': self._calculate_algorithm_stats(results)
        }
        
        self.algorithm_results[algorithm_name].append(algorithm_summary)
        return algorithm_summary
    
    def _calculate_algorithm_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall statistics for an algorithm."""
        if not results:
            return {}
        
        successful_cases = [r for r in results if r['success_rate'] > 0]
        
        if successful_cases:
            avg_execution_times = [r['avg_execution_time'] for r in successful_cases]
            avg_memory_usages = [r['avg_memory_usage'] for r in successful_cases]
            
            return {
                'overall_success_rate': len(successful_cases) / len(results) * 100,
                'avg_execution_time': statistics.mean(avg_execution_times),
                'fastest_case': min(avg_execution_times),
                'slowest_case': max(avg_execution_times),
                'avg_memory_usage': statistics.mean(avg_memory_usages),
                'peak_memory_usage': max([r['max_memory_usage'] for r in successful_cases])
            }
        
        return {'overall_success_rate': 0}
    
    def export_benchmark_results(self, filepath: str = "algorithm_benchmarks.json"):
        """Export all algorithm benchmark results."""
        with open(filepath, 'w') as f:
            json.dump(self.algorithm_results, f, indent=2, default=str)
        
        logger.info(f"Algorithm benchmark results exported to {filepath}")


class RegressionTester:
    """Performance regression testing utility."""
    
    def __init__(self, baseline_file: Optional[str] = None):
        self.baseline_results: Dict[str, Any] = {}
        self.current_results: Dict[str, Any] = {}
        
        if baseline_file:
            self.load_baseline(baseline_file)
    
    def load_baseline(self, filepath: str):
        """Load baseline performance results."""
        try:
            with open(filepath, 'r') as f:
                self.baseline_results = json.load(f)
            logger.info(f"Loaded baseline results from {filepath}")
        except FileNotFoundError:
            logger.warning(f"Baseline file not found: {filepath}")
    
    def add_current_result(self, test_name: str, metrics: Dict[str, Any]):
        """Add current test results."""
        self.current_results[test_name] = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
    
    def compare_performance(self, tolerance_percent: float = 10.0) -> Dict[str, Any]:
        """Compare current results against baseline."""
        if not self.baseline_results:
            return {"error": "No baseline results loaded"}
        
        comparisons = {}
        regressions = []
        improvements = []
        
        for test_name, current_data in self.current_results.items():
            if test_name in self.baseline_results:
                baseline_metrics = self.baseline_results[test_name]['metrics']
                current_metrics = current_data['metrics']
                
                comparison = self._compare_metrics(
                    baseline_metrics, 
                    current_metrics, 
                    tolerance_percent
                )
                
                comparisons[test_name] = comparison
                
                if comparison['has_regression']:
                    regressions.append(test_name)
                elif comparison['has_improvement']:
                    improvements.append(test_name)
        
        return {
            'total_tests': len(comparisons),
            'regressions': len(regressions),
            'improvements': len(improvements),
            'stable': len(comparisons) - len(regressions) - len(improvements),
            'regression_tests': regressions,
            'improvement_tests': improvements,
            'detailed_comparisons': comparisons
        }
    
    def _compare_metrics(
        self, 
        baseline: Dict[str, Any], 
        current: Dict[str, Any], 
        tolerance: float
    ) -> Dict[str, Any]:
        """Compare individual metrics."""
        key_metrics = ['avg_execution_time', 'avg_memory_usage', 'requests_per_second']
        
        comparison = {
            'has_regression': False,
            'has_improvement': False,
            'metric_changes': {}
        }
        
        for metric in key_metrics:
            if metric in baseline and metric in current:
                baseline_val = baseline[metric]
                current_val = current[metric]
                
                if baseline_val > 0:
                    change_percent = ((current_val - baseline_val) / baseline_val) * 100
                    
                    comparison['metric_changes'][metric] = {
                        'baseline': baseline_val,
                        'current': current_val,
                        'change_percent': change_percent,
                        'is_regression': False,
                        'is_improvement': False
                    }
                    
                    # For execution time and memory usage, increase is regression
                    if metric in ['avg_execution_time', 'avg_memory_usage']:
                        if change_percent > tolerance:
                            comparison['has_regression'] = True
                            comparison['metric_changes'][metric]['is_regression'] = True
                        elif change_percent < -tolerance:
                            comparison['has_improvement'] = True
                            comparison['metric_changes'][metric]['is_improvement'] = True
                    
                    # For RPS, decrease is regression
                    elif metric == 'requests_per_second':
                        if change_percent < -tolerance:
                            comparison['has_regression'] = True
                            comparison['metric_changes'][metric]['is_regression'] = True
                        elif change_percent > tolerance:
                            comparison['has_improvement'] = True
                            comparison['metric_changes'][metric]['is_improvement'] = True
        
        return comparison
    
    def save_current_as_baseline(self, filepath: str):
        """Save current results as new baseline."""
        with open(filepath, 'w') as f:
            json.dump(self.current_results, f, indent=2)
        
        logger.info(f"Current results saved as baseline: {filepath}")


# Global instances
load_tester = LoadTester()
performance_benchmark = PerformanceBenchmark()
regression_tester = RegressionTester() 