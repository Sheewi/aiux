"""
Action Tokenizer Evaluation Framework
Provides comprehensive evaluation metrics for action tokenization quality and performance.
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import statistics
from enum import Enum

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from action_tokenizer import ActionToken, ActionTokenizer, ActionRegistry, ActionType

logger = logging.getLogger(__name__)

# ============================================================================
# 1. EVALUATION METRICS
# ============================================================================

class EvaluationMetric(Enum):
    """Types of evaluation metrics."""
    ACCURACY = "accuracy"
    PRECISION = "precision" 
    RECALL = "recall"
    F1_SCORE = "f1_score"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    CAPABILITY_COVERAGE = "capability_coverage"
    ACTION_DIVERSITY = "action_diversity"
    ERROR_RATE = "error_rate"

@dataclass
class EvaluationResult:
    """Results from tokenizer evaluation."""
    metric: EvaluationMetric
    value: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'metric': self.metric.value,
            'value': self.value,
            'details': self.details,
            'timestamp': self.timestamp
        }

@dataclass
class TestCase:
    """Test case for tokenizer evaluation."""
    input_text: str
    expected_tokens: List[Dict[str, Any]]  # Expected token dictionaries
    capabilities: Optional[Set[str]] = None
    context: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationSuite:
    """Collection of test cases and evaluation configuration."""
    name: str
    test_cases: List[TestCase]
    description: str = ""
    metrics_to_evaluate: List[EvaluationMetric] = field(default_factory=lambda: [
        EvaluationMetric.ACCURACY,
        EvaluationMetric.LATENCY,
        EvaluationMetric.THROUGHPUT
    ])

# ============================================================================
# 2. EVALUATION ENGINE
# ============================================================================

class TokenizerEvaluator:
    """Comprehensive evaluator for action tokenizers."""
    
    def __init__(self, tokenizer: ActionTokenizer):
        self.tokenizer = tokenizer
        self.evaluation_history: List[Dict[str, Any]] = []
        
    def evaluate_suite(self, suite: EvaluationSuite) -> Dict[str, EvaluationResult]:
        """Evaluate tokenizer on a test suite."""
        logger.info(f"Starting evaluation of suite: {suite.name}")
        
        results = {}
        start_time = time.time()
        
        # Run all test cases
        predictions = []
        ground_truths = []
        latencies = []
        
        for i, test_case in enumerate(suite.test_cases):
            case_start = time.time()
            
            # Get tokenizer predictions
            predicted_tokens = self.tokenizer.tokenize(
                test_case.input_text,
                test_case.capabilities,
                test_case.context
            )
            
            case_latency = time.time() - case_start
            latencies.append(case_latency)
            
            predictions.append(predicted_tokens)
            ground_truths.append(test_case.expected_tokens)
            
            if i % 10 == 0:
                logger.debug(f"Processed {i+1}/{len(suite.test_cases)} test cases")
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        for metric in suite.metrics_to_evaluate:
            if metric == EvaluationMetric.ACCURACY:
                results[metric.value] = self._calculate_accuracy(predictions, ground_truths)
            elif metric == EvaluationMetric.PRECISION:
                results[metric.value] = self._calculate_precision(predictions, ground_truths)
            elif metric == EvaluationMetric.RECALL:
                results[metric.value] = self._calculate_recall(predictions, ground_truths)
            elif metric == EvaluationMetric.F1_SCORE:
                results[metric.value] = self._calculate_f1_score(predictions, ground_truths)
            elif metric == EvaluationMetric.LATENCY:
                results[metric.value] = self._calculate_latency_metrics(latencies)
            elif metric == EvaluationMetric.THROUGHPUT:
                results[metric.value] = self._calculate_throughput(len(suite.test_cases), total_time)
            elif metric == EvaluationMetric.SEMANTIC_SIMILARITY:
                results[metric.value] = self._calculate_semantic_similarity(predictions, ground_truths)
            elif metric == EvaluationMetric.CAPABILITY_COVERAGE:
                results[metric.value] = self._calculate_capability_coverage(predictions)
            elif metric == EvaluationMetric.ACTION_DIVERSITY:
                results[metric.value] = self._calculate_action_diversity(predictions)
            elif metric == EvaluationMetric.ERROR_RATE:
                results[metric.value] = self._calculate_error_rate(predictions, ground_truths)
        
        # Store evaluation history
        evaluation_record = {
            'suite_name': suite.name,
            'timestamp': start_time,
            'total_time': total_time,
            'num_test_cases': len(suite.test_cases),
            'results': {k: v.to_dict() for k, v in results.items()}
        }
        self.evaluation_history.append(evaluation_record)
        
        logger.info(f"Completed evaluation of {suite.name} in {total_time:.2f}s")
        return results
    
    def _calculate_accuracy(self, predictions: List[List[ActionToken]], 
                          ground_truths: List[List[Dict[str, Any]]]) -> EvaluationResult:
        """Calculate exact match accuracy."""
        correct = 0
        total = len(predictions)
        
        for pred_tokens, gt_tokens in zip(predictions, ground_truths):
            if self._tokens_match(pred_tokens, gt_tokens):
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return EvaluationResult(
            metric=EvaluationMetric.ACCURACY,
            value=accuracy,
            details={
                'correct': correct,
                'total': total,
                'accuracy_percentage': accuracy * 100
            }
        )
    
    def _calculate_precision(self, predictions: List[List[ActionToken]], 
                           ground_truths: List[List[Dict[str, Any]]]) -> EvaluationResult:
        """Calculate token-level precision."""
        true_positives = 0
        predicted_positives = 0
        
        for pred_tokens, gt_tokens in zip(predictions, ground_truths):
            predicted_positives += len(pred_tokens)
            
            # Count matches
            for pred_token in pred_tokens:
                for gt_token in gt_tokens:
                    if self._token_matches_dict(pred_token, gt_token):
                        true_positives += 1
                        break
        
        precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
        
        return EvaluationResult(
            metric=EvaluationMetric.PRECISION,
            value=precision,
            details={
                'true_positives': true_positives,
                'predicted_positives': predicted_positives,
                'precision_percentage': precision * 100
            }
        )
    
    def _calculate_recall(self, predictions: List[List[ActionToken]], 
                         ground_truths: List[List[Dict[str, Any]]]) -> EvaluationResult:
        """Calculate token-level recall."""
        true_positives = 0
        actual_positives = 0
        
        for pred_tokens, gt_tokens in zip(predictions, ground_truths):
            actual_positives += len(gt_tokens)
            
            # Count matches
            for gt_token in gt_tokens:
                for pred_token in pred_tokens:
                    if self._token_matches_dict(pred_token, gt_token):
                        true_positives += 1
                        break
        
        recall = true_positives / actual_positives if actual_positives > 0 else 0.0
        
        return EvaluationResult(
            metric=EvaluationMetric.RECALL,
            value=recall,
            details={
                'true_positives': true_positives,
                'actual_positives': actual_positives,
                'recall_percentage': recall * 100
            }
        )
    
    def _calculate_f1_score(self, predictions: List[List[ActionToken]], 
                           ground_truths: List[List[Dict[str, Any]]]) -> EvaluationResult:
        """Calculate F1 score."""
        precision_result = self._calculate_precision(predictions, ground_truths)
        recall_result = self._calculate_recall(predictions, ground_truths)
        
        precision = precision_result.value
        recall = recall_result.value
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return EvaluationResult(
            metric=EvaluationMetric.F1_SCORE,
            value=f1,
            details={
                'precision': precision,
                'recall': recall,
                'f1_percentage': f1 * 100
            }
        )
    
    def _calculate_latency_metrics(self, latencies: List[float]) -> EvaluationResult:
        """Calculate latency statistics."""
        if not latencies:
            return EvaluationResult(metric=EvaluationMetric.LATENCY, value=0.0)
        
        mean_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
        p99_latency = sorted(latencies)[int(0.99 * len(latencies))]
        
        return EvaluationResult(
            metric=EvaluationMetric.LATENCY,
            value=mean_latency,
            details={
                'mean_ms': mean_latency * 1000,
                'median_ms': median_latency * 1000,
                'p95_ms': p95_latency * 1000,
                'p99_ms': p99_latency * 1000,
                'min_ms': min(latencies) * 1000,
                'max_ms': max(latencies) * 1000
            }
        )
    
    def _calculate_throughput(self, num_requests: int, total_time: float) -> EvaluationResult:
        """Calculate throughput (requests per second)."""
        throughput = num_requests / total_time if total_time > 0 else 0.0
        
        return EvaluationResult(
            metric=EvaluationMetric.THROUGHPUT,
            value=throughput,
            details={
                'requests_per_second': throughput,
                'total_requests': num_requests,
                'total_time_seconds': total_time
            }
        )
    
    def _calculate_semantic_similarity(self, predictions: List[List[ActionToken]], 
                                     ground_truths: List[List[Dict[str, Any]]]) -> EvaluationResult:
        """Calculate semantic similarity between predicted and expected actions."""
        similarities = []
        
        for pred_tokens, gt_tokens in zip(predictions, ground_truths):
            similarity = self._compute_sequence_similarity(pred_tokens, gt_tokens)
            similarities.append(similarity)
        
        mean_similarity = statistics.mean(similarities) if similarities else 0.0
        
        return EvaluationResult(
            metric=EvaluationMetric.SEMANTIC_SIMILARITY,
            value=mean_similarity,
            details={
                'mean_similarity': mean_similarity,
                'similarity_distribution': {
                    'min': min(similarities) if similarities else 0,
                    'max': max(similarities) if similarities else 0,
                    'std': statistics.stdev(similarities) if len(similarities) > 1 else 0
                }
            }
        )
    
    def _calculate_capability_coverage(self, predictions: List[List[ActionToken]]) -> EvaluationResult:
        """Calculate coverage of different capabilities."""
        all_capabilities = set()
        used_capabilities = set()
        
        # Get all possible capabilities from registry
        for action in self.tokenizer.registry.list_actions():
            all_capabilities.update(action.capabilities)
        
        # Get used capabilities from predictions
        for pred_tokens in predictions:
            for token in pred_tokens:
                used_capabilities.update(token.capabilities)
        
        coverage = len(used_capabilities) / len(all_capabilities) if all_capabilities else 0.0
        
        return EvaluationResult(
            metric=EvaluationMetric.CAPABILITY_COVERAGE,
            value=coverage,
            details={
                'total_capabilities': len(all_capabilities),
                'used_capabilities': len(used_capabilities),
                'coverage_percentage': coverage * 100,
                'unused_capabilities': list(all_capabilities - used_capabilities)
            }
        )
    
    def _calculate_action_diversity(self, predictions: List[List[ActionToken]]) -> EvaluationResult:
        """Calculate diversity of predicted actions."""
        action_counts = Counter()
        total_tokens = 0
        
        for pred_tokens in predictions:
            for token in pred_tokens:
                action_counts[token.name] += 1
                total_tokens += 1
        
        # Calculate entropy as diversity measure
        diversity = 0.0
        if total_tokens > 0:
            for count in action_counts.values():
                prob = count / total_tokens
                if prob > 0:
                    diversity -= prob * (prob ** 0.5)  # Modified entropy
        
        return EvaluationResult(
            metric=EvaluationMetric.ACTION_DIVERSITY,
            value=diversity,
            details={
                'unique_actions': len(action_counts),
                'total_tokens': total_tokens,
                'action_distribution': dict(action_counts),
                'diversity_score': diversity
            }
        )
    
    def _calculate_error_rate(self, predictions: List[List[ActionToken]], 
                            ground_truths: List[List[Dict[str, Any]]]) -> EvaluationResult:
        """Calculate various error rates."""
        total_cases = len(predictions)
        parsing_errors = 0
        validation_errors = 0
        semantic_errors = 0
        
        for pred_tokens, gt_tokens in zip(predictions, ground_truths):
            # Parsing errors (no tokens produced when expected)
            if len(gt_tokens) > 0 and len(pred_tokens) == 0:
                parsing_errors += 1
            
            # Validation errors (invalid token structure)
            for token in pred_tokens:
                if not self._is_valid_token(token):
                    validation_errors += 1
                    break
            
            # Semantic errors (wrong action types)
            if not self._semantic_match(pred_tokens, gt_tokens):
                semantic_errors += 1
        
        overall_error_rate = (parsing_errors + validation_errors + semantic_errors) / (total_cases * 3)
        
        return EvaluationResult(
            metric=EvaluationMetric.ERROR_RATE,
            value=overall_error_rate,
            details={
                'parsing_errors': parsing_errors,
                'validation_errors': validation_errors,
                'semantic_errors': semantic_errors,
                'total_cases': total_cases,
                'parsing_error_rate': parsing_errors / total_cases,
                'validation_error_rate': validation_errors / total_cases,
                'semantic_error_rate': semantic_errors / total_cases
            }
        )
    
    # Helper methods
    def _tokens_match(self, pred_tokens: List[ActionToken], 
                     gt_tokens: List[Dict[str, Any]]) -> bool:
        """Check if predicted tokens exactly match ground truth."""
        if len(pred_tokens) != len(gt_tokens):
            return False
        
        for pred_token, gt_token in zip(pred_tokens, gt_tokens):
            if not self._token_matches_dict(pred_token, gt_token):
                return False
        
        return True
    
    def _token_matches_dict(self, token: ActionToken, gt_dict: Dict[str, Any]) -> bool:
        """Check if token matches ground truth dictionary."""
        return (token.name == gt_dict.get('name') and 
                token.args == gt_dict.get('args', {}))
    
    def _compute_sequence_similarity(self, pred_tokens: List[ActionToken], 
                                   gt_tokens: List[Dict[str, Any]]) -> float:
        """Compute similarity between token sequences."""
        if not pred_tokens and not gt_tokens:
            return 1.0
        if not pred_tokens or not gt_tokens:
            return 0.0
        
        # Simple similarity based on action name overlap
        pred_names = {token.name for token in pred_tokens}
        gt_names = {token.get('name') for token in gt_tokens}
        
        intersection = len(pred_names & gt_names)
        union = len(pred_names | gt_names)
        
        return intersection / union if union > 0 else 0.0
    
    def _is_valid_token(self, token: ActionToken) -> bool:
        """Check if token is structurally valid."""
        try:
            # Check required fields
            if not hasattr(token, 'id') or not hasattr(token, 'name'):
                return False
            if not hasattr(token, 'action_type') or not hasattr(token, 'args'):
                return False
            
            # Check if action exists in registry
            action_def = self.tokenizer.registry.get_action(token.name)
            if not action_def:
                return False
            
            # Validate arguments
            valid, _ = action_def.validate_args(token.args)
            return valid
            
        except Exception:
            return False
    
    def _semantic_match(self, pred_tokens: List[ActionToken], 
                       gt_tokens: List[Dict[str, Any]]) -> bool:
        """Check if token sequences have semantic similarity."""
        if not pred_tokens and not gt_tokens:
            return True
        
        # At least 50% of action types should match
        pred_types = [token.action_type for token in pred_tokens]
        gt_types = [ActionType(gt.get('action_type', 1)) for gt in gt_tokens]
        
        if not pred_types and not gt_types:
            return True
        
        matches = sum(1 for p, g in zip(pred_types, gt_types) if p == g)
        return matches / max(len(pred_types), len(gt_types)) >= 0.5
    
    def export_evaluation_history(self, filepath: str):
        """Export evaluation history to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.evaluation_history, f, indent=2)
        logger.info(f"Exported evaluation history to {filepath}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of recent performance."""
        if not self.evaluation_history:
            return {}
        
        latest = self.evaluation_history[-1]
        return {
            'latest_evaluation': latest['suite_name'],
            'timestamp': latest['timestamp'],
            'performance_metrics': latest['results']
        }

# ============================================================================
# 3. BUILT-IN TEST SUITES
# ============================================================================

def create_basic_test_suite() -> EvaluationSuite:
    """Create a basic test suite for fundamental functionality."""
    test_cases = [
        TestCase(
            input_text="click the submit button",
            expected_tokens=[{
                'name': 'CLICK',
                'args': {'target': 'input[type="submit"], button[type="submit"]'}
            }]
        ),
        TestCase(
            input_text='type "hello world"',
            expected_tokens=[{
                'name': 'TYPE',
                'args': {'target': 'input:focus', 'text': 'hello world'}
            }]
        ),
        TestCase(
            input_text="wait for 5 seconds",
            expected_tokens=[{
                'name': 'WAIT',
                'args': {'duration': 5.0}
            }]
        ),
        TestCase(
            input_text="navigate to https://example.com",
            expected_tokens=[{
                'name': 'NAVIGATE',
                'args': {'url': 'https://example.com'}
            }]
        ),
        TestCase(
            input_text="set username to john",
            expected_tokens=[{
                'name': 'SET',
                'args': {'key': 'username', 'value': 'john'}
            }]
        )
    ]
    
    return EvaluationSuite(
        name="basic_functionality",
        test_cases=test_cases,
        description="Basic tokenization functionality tests"
    )

def create_complex_test_suite() -> EvaluationSuite:
    """Create a more complex test suite with edge cases."""
    test_cases = [
        TestCase(
            input_text="click button then wait 2 seconds",
            expected_tokens=[
                {'name': 'CLICK', 'args': {'target': '*:contains("button")'}},
                {'name': 'WAIT', 'args': {'duration': 2.0}}
            ]
        ),
        TestCase(
            input_text="",  # Empty input
            expected_tokens=[]
        ),
        TestCase(
            input_text="invalid action xyz",
            expected_tokens=[]
        ),
        TestCase(
            input_text="click #submit-btn twice",
            expected_tokens=[{
                'name': 'CLICK',
                'args': {'target': '#submit-btn', 'count': 2}
            }]
        )
    ]
    
    return EvaluationSuite(
        name="complex_scenarios",
        test_cases=test_cases,
        description="Complex scenarios and edge cases"
    )

def create_performance_test_suite() -> EvaluationSuite:
    """Create a test suite focused on performance evaluation."""
    # Generate many simple test cases for throughput testing
    test_cases = []
    for i in range(100):
        test_cases.append(TestCase(
            input_text=f"click button {i}",
            expected_tokens=[{
                'name': 'CLICK',
                'args': {'target': f'*:contains("button {i}")'}
            }]
        ))
    
    return EvaluationSuite(
        name="performance_benchmark",
        test_cases=test_cases,
        description="Performance and throughput benchmark",
        metrics_to_evaluate=[
            EvaluationMetric.LATENCY,
            EvaluationMetric.THROUGHPUT,
            EvaluationMetric.ERROR_RATE
        ]
    )

# ============================================================================
# 4. CONVENIENCE FUNCTIONS
# ============================================================================

def evaluate_tokenizer(tokenizer: ActionTokenizer, 
                      suite_name: str = "basic") -> Dict[str, EvaluationResult]:
    """Quick evaluation function."""
    evaluator = TokenizerEvaluator(tokenizer)
    
    if suite_name == "basic":
        suite = create_basic_test_suite()
    elif suite_name == "complex":
        suite = create_complex_test_suite()
    elif suite_name == "performance":
        suite = create_performance_test_suite()
    else:
        raise ValueError(f"Unknown test suite: {suite_name}")
    
    return evaluator.evaluate_suite(suite)

def benchmark_tokenizer(tokenizer: ActionTokenizer) -> Dict[str, Any]:
    """Run comprehensive benchmark on tokenizer."""
    evaluator = TokenizerEvaluator(tokenizer)
    
    results = {}
    suites = [
        create_basic_test_suite(),
        create_complex_test_suite(),
        create_performance_test_suite()
    ]
    
    for suite in suites:
        suite_results = evaluator.evaluate_suite(suite)
        results[suite.name] = suite_results
    
    return results
