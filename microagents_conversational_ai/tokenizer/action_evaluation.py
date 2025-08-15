"""
Action Tokenizer Evaluation Framework

Comprehensive evaluation metrics, benchmarks, and testing utilities
for action tokenization quality assessment.
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import time
import statistics
from collections import defaultdict, Counter
import math

from .action_algebra import ActionToken, ActionSequence, ActionAlgebra, NOOP
from .action_abi import ActionABI
from .action_tokenizer import ActionTokenizer, TokenizationContext, DisambiguationResult


@dataclass
class EvaluationSample:
    """Single evaluation sample with input, expected output, and metadata."""
    input_text: str
    expected_tokens: List[ActionToken]
    context: Optional[TokenizationContext] = None
    difficulty: str = "medium"  # easy, medium, hard
    category: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of evaluating a single sample."""
    sample_id: str
    predicted_tokens: List[ActionToken]
    expected_tokens: List[ActionToken]
    metrics: Dict[str, float]
    execution_time: float
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResults:
    """Results from running a complete benchmark."""
    total_samples: int
    results: List[EvaluationResult]
    aggregate_metrics: Dict[str, float]
    category_metrics: Dict[str, Dict[str, float]]
    difficulty_metrics: Dict[str, Dict[str, float]]
    execution_time: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricCalculator:
    """Calculates various evaluation metrics for tokenization quality."""
    
    @staticmethod
    def exact_match(predicted: List[ActionToken], expected: List[ActionToken]) -> float:
        """Exact sequence match accuracy."""
        if not predicted and not expected:
            return 1.0
        if len(predicted) != len(expected):
            return 0.0
        
        for p, e in zip(predicted, expected):
            if (p.name != e.name or 
                p.type != e.type or 
                p.args != e.args):
                return 0.0
        
        return 1.0
    
    @staticmethod
    def token_level_precision(predicted: List[ActionToken], expected: List[ActionToken]) -> float:
        """Token-level precision (correct predictions / total predictions)."""
        if not predicted:
            return 1.0 if not expected else 0.0
        
        predicted_set = {(t.name, str(t.args)) for t in predicted}
        expected_set = {(t.name, str(t.args)) for t in expected}
        
        true_positives = len(predicted_set & expected_set)
        return true_positives / len(predicted_set)
    
    @staticmethod
    def token_level_recall(predicted: List[ActionToken], expected: List[ActionToken]) -> float:
        """Token-level recall (correct predictions / total expected)."""
        if not expected:
            return 1.0 if not predicted else 0.0
        
        predicted_set = {(t.name, str(t.args)) for t in predicted}
        expected_set = {(t.name, str(t.args)) for t in expected}
        
        true_positives = len(predicted_set & expected_set)
        return true_positives / len(expected_set)
    
    @staticmethod
    def token_level_f1(predicted: List[ActionToken], expected: List[ActionToken]) -> float:
        """Token-level F1 score."""
        precision = MetricCalculator.token_level_precision(predicted, expected)
        recall = MetricCalculator.token_level_recall(predicted, expected)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    @staticmethod
    def sequence_similarity(predicted: List[ActionToken], expected: List[ActionToken]) -> float:
        """Sequence similarity using edit distance."""
        if not predicted and not expected:
            return 1.0
        if not predicted or not expected:
            return 0.0
        
        # Convert to string representations for edit distance
        pred_strs = [f"{t.name}({t.args})" for t in predicted]
        exp_strs = [f"{t.name}({t.args})" for t in expected]
        
        return MetricCalculator._edit_distance_similarity(pred_strs, exp_strs)
    
    @staticmethod
    def _edit_distance_similarity(seq1: List[str], seq2: List[str]) -> float:
        """Calculate edit distance similarity."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        edit_distance = dp[m][n]
        max_len = max(m, n)
        return 1.0 - (edit_distance / max_len) if max_len > 0 else 1.0
    
    @staticmethod
    def argument_accuracy(predicted: List[ActionToken], expected: List[ActionToken]) -> float:
        """Accuracy of argument extraction."""
        if not predicted and not expected:
            return 1.0
        
        # Match tokens by name first
        pred_by_name = {t.name: t for t in predicted}
        exp_by_name = {t.name: t for t in expected}
        
        common_names = set(pred_by_name.keys()) & set(exp_by_name.keys())
        if not common_names:
            return 0.0
        
        correct_args = 0
        total_args = 0
        
        for name in common_names:
            pred_token = pred_by_name[name]
            exp_token = exp_by_name[name]
            
            # Compare each argument
            all_arg_keys = set(pred_token.args.keys()) | set(exp_token.args.keys())
            for key in all_arg_keys:
                total_args += 1
                if (key in pred_token.args and 
                    key in exp_token.args and 
                    pred_token.args[key] == exp_token.args[key]):
                    correct_args += 1
        
        return correct_args / total_args if total_args > 0 else 1.0
    
    @staticmethod
    def semantic_validity(predicted: List[ActionToken], abi: ActionABI) -> float:
        """Check semantic validity of predicted tokens."""
        if not predicted:
            return 1.0
        
        valid_tokens = 0
        for token in predicted:
            if abi.validate_token(token):
                valid_tokens += 1
        
        return valid_tokens / len(predicted)
    
    @staticmethod
    def capability_consistency(predicted: List[ActionToken], 
                              context: TokenizationContext) -> float:
        """Check if predicted tokens are consistent with available capabilities."""
        if not predicted:
            return 1.0
        
        required_caps = set()
        for token in predicted:
            required_caps.update(token.caps)
        
        if not required_caps:
            return 1.0
        
        available_caps = len(required_caps & context.capabilities)
        return available_caps / len(required_caps)


class BenchmarkDataset:
    """Dataset for tokenizer benchmarking."""
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.samples: List[EvaluationSample] = []
        self.metadata = {}
    
    def add_sample(self, sample: EvaluationSample):
        """Add a sample to the dataset."""
        self.samples.append(sample)
    
    def add_samples_from_dict(self, data: List[Dict[str, Any]]):
        """Add samples from dictionary data."""
        for item in data:
            context = None
            if 'context' in item:
                ctx_data = item['context']
                context = TokenizationContext(
                    capabilities=set(ctx_data.get('capabilities', [])),
                    constraints=ctx_data.get('constraints', {}),
                    state=ctx_data.get('state', {})
                )
            
            # Parse expected tokens
            expected_tokens = []
            for token_data in item.get('expected_tokens', []):
                token = ActionToken(
                    id=token_data.get('id', 0),
                    name=token_data['name'],
                    type=token_data.get('type', 'action'),
                    args=token_data.get('args', {}),
                    caps=set(token_data.get('caps', [])),
                    meta=token_data.get('meta', {})
                )
                expected_tokens.append(token)
            
            sample = EvaluationSample(
                input_text=item['input_text'],
                expected_tokens=expected_tokens,
                context=context,
                difficulty=item.get('difficulty', 'medium'),
                category=item.get('category', 'general'),
                metadata=item.get('metadata', {})
            )
            
            self.add_sample(sample)
    
    def get_samples_by_category(self, category: str) -> List[EvaluationSample]:
        """Get samples by category."""
        return [s for s in self.samples if s.category == category]
    
    def get_samples_by_difficulty(self, difficulty: str) -> List[EvaluationSample]:
        """Get samples by difficulty."""
        return [s for s in self.samples if s.difficulty == difficulty]
    
    def save_to_file(self, filepath: str):
        """Save dataset to JSON file."""
        data = {
            'name': self.name,
            'metadata': self.metadata,
            'samples': []
        }
        
        for sample in self.samples:
            sample_data = {
                'input_text': sample.input_text,
                'expected_tokens': [
                    {
                        'id': token.id,
                        'name': token.name,
                        'type': token.type,
                        'args': token.args,
                        'caps': list(token.caps),
                        'meta': token.meta
                    }
                    for token in sample.expected_tokens
                ],
                'difficulty': sample.difficulty,
                'category': sample.category,
                'metadata': sample.metadata
            }
            
            if sample.context:
                sample_data['context'] = {
                    'capabilities': list(sample.context.capabilities),
                    'constraints': sample.context.constraints,
                    'state': sample.context.state
                }
            
            data['samples'].append(sample_data)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'BenchmarkDataset':
        """Load dataset from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        dataset = cls(data['name'])
        dataset.metadata = data.get('metadata', {})
        dataset.add_samples_from_dict(data['samples'])
        
        return dataset


class TokenizerEvaluator:
    """Comprehensive evaluator for action tokenizers."""
    
    def __init__(self, tokenizer: ActionTokenizer, abi: ActionABI):
        self.tokenizer = tokenizer
        self.abi = abi
        self.metric_calculator = MetricCalculator()
    
    def evaluate_sample(self, sample: EvaluationSample, 
                       sample_id: str = None) -> EvaluationResult:
        """Evaluate a single sample."""
        start_time = time.time()
        
        try:
            # Tokenize input
            result = self.tokenizer.tokenize(sample.input_text, sample.context)
            predicted_tokens = result.primary_tokens
            
            # Calculate metrics
            metrics = self._calculate_all_metrics(
                predicted_tokens, sample.expected_tokens, sample.context
            )
            
            execution_time = time.time() - start_time
            
            return EvaluationResult(
                sample_id=sample_id or f"sample_{time.time()}",
                predicted_tokens=predicted_tokens,
                expected_tokens=sample.expected_tokens,
                metrics=metrics,
                execution_time=execution_time,
                errors=[],
                metadata={
                    'confidence': result.confidence,
                    'ambiguity_score': result.ambiguity_score,
                    'alternatives': len(result.alternatives)
                }
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            
            return EvaluationResult(
                sample_id=sample_id or f"sample_{time.time()}",
                predicted_tokens=[],
                expected_tokens=sample.expected_tokens,
                metrics={'error': 1.0},
                execution_time=execution_time,
                errors=[str(e)]
            )
    
    def evaluate_dataset(self, dataset: BenchmarkDataset) -> BenchmarkResults:
        """Evaluate entire dataset."""
        start_time = time.time()
        results = []
        
        for i, sample in enumerate(dataset.samples):
            sample_id = f"{dataset.name}_sample_{i}"
            result = self.evaluate_sample(sample, sample_id)
            results.append(result)
        
        execution_time = time.time() - start_time
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(results)
        
        # Calculate category-specific metrics
        category_metrics = self._calculate_category_metrics(results, dataset)
        
        # Calculate difficulty-specific metrics
        difficulty_metrics = self._calculate_difficulty_metrics(results, dataset)
        
        return BenchmarkResults(
            total_samples=len(dataset.samples),
            results=results,
            aggregate_metrics=aggregate_metrics,
            category_metrics=category_metrics,
            difficulty_metrics=difficulty_metrics,
            execution_time=execution_time,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            metadata={
                'dataset_name': dataset.name,
                'tokenizer_stats': self.tokenizer.get_statistics()
            }
        )
    
    def _calculate_all_metrics(self, predicted: List[ActionToken], 
                              expected: List[ActionToken],
                              context: TokenizationContext = None) -> Dict[str, float]:
        """Calculate all evaluation metrics."""
        metrics = {}
        
        # Core metrics
        metrics['exact_match'] = self.metric_calculator.exact_match(predicted, expected)
        metrics['token_precision'] = self.metric_calculator.token_level_precision(predicted, expected)
        metrics['token_recall'] = self.metric_calculator.token_level_recall(predicted, expected)
        metrics['token_f1'] = self.metric_calculator.token_level_f1(predicted, expected)
        metrics['sequence_similarity'] = self.metric_calculator.sequence_similarity(predicted, expected)
        metrics['argument_accuracy'] = self.metric_calculator.argument_accuracy(predicted, expected)
        
        # Semantic metrics
        metrics['semantic_validity'] = self.metric_calculator.semantic_validity(predicted, self.abi)
        
        if context:
            metrics['capability_consistency'] = self.metric_calculator.capability_consistency(predicted, context)
        
        # Length metrics
        metrics['length_ratio'] = len(predicted) / len(expected) if expected else (0 if not predicted else float('inf'))
        
        return metrics
    
    def _calculate_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Calculate aggregate metrics across all results."""
        if not results:
            return {}
        
        # Get all metric names
        metric_names = set()
        for result in results:
            metric_names.update(result.metrics.keys())
        
        aggregate = {}
        for metric_name in metric_names:
            if metric_name == 'error':
                continue
            
            values = [r.metrics.get(metric_name, 0.0) for r in results 
                     if 'error' not in r.metrics]
            
            if values:
                aggregate[f"{metric_name}_mean"] = statistics.mean(values)
                aggregate[f"{metric_name}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
                aggregate[f"{metric_name}_min"] = min(values)
                aggregate[f"{metric_name}_max"] = max(values)
        
        # Error rate
        error_count = sum(1 for r in results if 'error' in r.metrics)
        aggregate['error_rate'] = error_count / len(results)
        
        # Execution time statistics
        exec_times = [r.execution_time for r in results]
        aggregate['avg_execution_time'] = statistics.mean(exec_times)
        aggregate['total_execution_time'] = sum(exec_times)
        
        return aggregate
    
    def _calculate_category_metrics(self, results: List[EvaluationResult], 
                                   dataset: BenchmarkDataset) -> Dict[str, Dict[str, float]]:
        """Calculate metrics by category."""
        category_results = defaultdict(list)
        
        for i, result in enumerate(results):
            if i < len(dataset.samples):
                category = dataset.samples[i].category
                category_results[category].append(result)
        
        category_metrics = {}
        for category, cat_results in category_results.items():
            category_metrics[category] = self._calculate_aggregate_metrics(cat_results)
        
        return category_metrics
    
    def _calculate_difficulty_metrics(self, results: List[EvaluationResult], 
                                     dataset: BenchmarkDataset) -> Dict[str, Dict[str, float]]:
        """Calculate metrics by difficulty."""
        difficulty_results = defaultdict(list)
        
        for i, result in enumerate(results):
            if i < len(dataset.samples):
                difficulty = dataset.samples[i].difficulty
                difficulty_results[difficulty].append(result)
        
        difficulty_metrics = {}
        for difficulty, diff_results in difficulty_results.items():
            difficulty_metrics[difficulty] = self._calculate_aggregate_metrics(diff_results)
        
        return difficulty_metrics


def create_default_benchmark_dataset() -> BenchmarkDataset:
    """Create a default benchmark dataset for testing."""
    dataset = BenchmarkDataset("default_benchmark")
    
    # Basic action samples
    basic_samples = [
        {
            'input_text': 'click on the submit button',
            'expected_tokens': [
                {
                    'name': 'CLICK',
                    'args': {'target': 'submit button'},
                    'caps': ['ui', 'automation']
                }
            ],
            'category': 'basic_actions',
            'difficulty': 'easy'
        },
        {
            'input_text': 'type "hello world" in the search box',
            'expected_tokens': [
                {
                    'name': 'TYPE',
                    'args': {'text': 'hello world', 'target': 'search box'},
                    'caps': ['ui', 'automation']
                }
            ],
            'category': 'basic_actions',
            'difficulty': 'easy'
        },
        {
            'input_text': 'wait 5 seconds',
            'expected_tokens': [
                {
                    'name': 'WAIT',
                    'args': {'duration': 5.0},
                    'caps': ['timing']
                }
            ],
            'category': 'basic_actions',
            'difficulty': 'easy'
        },
        {
            'input_text': 'navigate to https://example.com',
            'expected_tokens': [
                {
                    'name': 'NAVIGATE',
                    'args': {'url': 'https://example.com'},
                    'caps': ['web', 'navigation']
                }
            ],
            'category': 'basic_actions',
            'difficulty': 'easy'
        },
    ]
    
    # Complex sequence samples
    complex_samples = [
        {
            'input_text': 'click submit and wait 3 seconds',
            'expected_tokens': [
                {
                    'name': 'CLICK',
                    'args': {'target': 'submit'},
                    'caps': ['ui', 'automation']
                },
                {
                    'name': 'WAIT',
                    'args': {'duration': 3.0},
                    'caps': ['timing']
                }
            ],
            'category': 'sequences',
            'difficulty': 'medium'
        },
        {
            'input_text': 'type "username" in login field and click submit button',
            'expected_tokens': [
                {
                    'name': 'TYPE',
                    'args': {'text': 'username', 'target': 'login field'},
                    'caps': ['ui', 'automation']
                },
                {
                    'name': 'CLICK',
                    'args': {'target': 'submit button'},
                    'caps': ['ui', 'automation']
                }
            ],
            'category': 'sequences',
            'difficulty': 'medium'
        }
    ]
    
    # Add context for some samples
    context_data = {
        'capabilities': ['ui', 'automation', 'web', 'navigation', 'timing'],
        'constraints': {'max_sequence_length': 5},
        'state': {}
    }
    
    all_samples = basic_samples + complex_samples
    for sample in all_samples:
        sample['context'] = context_data
    
    dataset.add_samples_from_dict(all_samples)
    
    return dataset


# Example usage
if __name__ == "__main__":
    # Create evaluation setup
    from .action_abi import create_default_abi
    
    abi = create_default_abi()
    tokenizer = ActionTokenizer(abi)
    evaluator = TokenizerEvaluator(tokenizer, abi)
    
    # Create and run benchmark
    dataset = create_default_benchmark_dataset()
    results = evaluator.evaluate_dataset(dataset)
    
    # Print results
    print("Benchmark Results")
    print("=" * 50)
    print(f"Total samples: {results.total_samples}")
    print(f"Execution time: {results.execution_time:.2f}s")
    print(f"Error rate: {results.aggregate_metrics.get('error_rate', 0.0):.2%}")
    
    print("\nAggregate Metrics:")
    for metric, value in results.aggregate_metrics.items():
        if metric.endswith('_mean'):
            print(f"  {metric}: {value:.3f}")
    
    print("\nCategory Metrics:")
    for category, metrics in results.category_metrics.items():
        print(f"  {category}:")
        exact_match_mean = metrics.get('exact_match_mean', 0.0)
        print(f"    Exact match: {exact_match_mean:.3f}")
