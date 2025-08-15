import asyncio
import pandas as pd
import numpy as np
import json
import csv
import xml.etree.ElementTree as ET
import logging
import re
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import statistics
from collections import Counter
import base64
import io

from .base_tool import BaseTool, ToolStatus, ToolMetadata, ToolType, ToolCapability, create_tool_metadata


class DataProcessingTool(BaseTool):
    """Comprehensive data processing tool for transformation, analysis, and manipulation."""
    
    def __init__(self, max_rows: int = 1000000, max_memory_mb: int = 512, config: Dict[str, Any] = None):
        """
        Initialize the data processing tool.
        
        Args:
            max_rows: Maximum number of rows to process
            max_memory_mb: Maximum memory usage in MB
            config: Additional configuration
        """
        # Initialize metadata
        metadata = create_tool_metadata(
            tool_id="data_processing",
            name="Data Processing",
            description="Comprehensive data processing tool for transformation, analysis, filtering, aggregation, and manipulation",
            tool_type=ToolType.DATA_PROCESSING,
            version="1.0.0",
            author="System",
            capabilities=[
                ToolCapability.ASYNC_EXECUTION,
                ToolCapability.BATCH_PROCESSING,
                ToolCapability.CACHEABLE
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["analyze", "filter", "transform", "aggregate", "sort", "join", "clean"]},
                    "data": {"type": "object", "description": "Input data to process"},
                    "analysis_type": {"type": "string", "enum": ["summary", "detailed", "statistical"]},
                    "filters": {"type": "array", "description": "Filter conditions"},
                    "transformations": {"type": "array", "description": "Data transformations"},
                    "group_by": {"type": "array", "description": "Columns to group by"},
                    "sort_by": {"type": "array", "description": "Columns to sort by"}
                },
                "required": ["operation", "data"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "result": {"type": "object"},
                    "error": {"type": "string"},
                    "metadata": {"type": "object"}
                }
            },
            timeout=120.0,
            supported_formats=["json", "csv", "dict", "list"],
            tags=["data", "analysis", "processing", "statistics", "transformation"]
        )
        
        super().__init__(metadata, config)
        self.max_rows = max_rows
        self.max_memory_mb = max_memory_mb
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Execute data processing operation.
        
        Args:
            operation: Type of operation to perform
            **kwargs: Operation-specific parameters
        """
        self.status = ToolStatus.RUNNING
        
        try:
            operation_map = {
                'load': self._load_data,
                'save': self._save_data,
                'transform': self._transform_data,
                'filter': self._filter_data,
                'aggregate': self._aggregate_data,
                'join': self._join_data,
                'analyze': self._analyze_data,
                'clean': self._clean_data,
                'convert': self._convert_format,
                'validate': self._validate_data,
                'sample': self._sample_data,
                'pivot': self._pivot_data
            }
            
            if operation not in operation_map:
                raise ValueError(f"Unknown operation: {operation}")
            
            result = await operation_map[operation](**kwargs)
            self.status = ToolStatus.COMPLETED
            return {
                'success': True,
                'operation': operation,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.status = ToolStatus.FAILED
            self.logger.error(f"Data processing operation failed: {e}")
            return {
                'success': False,
                'operation': operation,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _load_data(self, source: Union[str, Dict], 
                        format: str = 'auto',
                        **options) -> Dict[str, Any]:
        """Load data from various sources."""
        if isinstance(source, str):
            # File path
            if format == 'auto':
                if source.endswith('.csv'):
                    format = 'csv'
                elif source.endswith('.json'):
                    format = 'json'
                elif source.endswith('.xlsx') or source.endswith('.xls'):
                    format = 'excel'
                elif source.endswith('.xml'):
                    format = 'xml'
                else:
                    format = 'csv'  # Default
            
            if format == 'csv':
                df = pd.read_csv(source, **options)
            elif format == 'json':
                df = pd.read_json(source, **options)
            elif format == 'excel':
                df = pd.read_excel(source, **options)
            elif format == 'xml':
                # Basic XML parsing
                tree = ET.parse(source)
                root = tree.getroot()
                data = []
                for child in root:
                    row = {}
                    for subchild in child:
                        row[subchild.tag] = subchild.text
                    data.append(row)
                df = pd.DataFrame(data)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        elif isinstance(source, dict):
            # Dictionary data
            df = pd.DataFrame(source)
        
        else:
            raise ValueError("Source must be file path or dictionary")
        
        # Memory check
        memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
        if memory_usage > self.max_memory_mb:
            raise MemoryError(f"Data too large: {memory_usage:.2f}MB exceeds limit of {self.max_memory_mb}MB")
        
        # Row count check
        if len(df) > self.max_rows:
            raise ValueError(f"Too many rows: {len(df)} exceeds limit of {self.max_rows}")
        
        return {
            'data_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'memory_usage_mb': round(memory_usage, 2),
                'dtypes': df.dtypes.to_dict()
            },
            'sample_data': df.head(5).to_dict('records'),
            'data_id': id(df)  # For referencing in subsequent operations
        }
    
    async def _save_data(self, data: pd.DataFrame, 
                        destination: str,
                        format: str = 'auto',
                        **options) -> Dict[str, Any]:
        """Save data to various formats."""
        if format == 'auto':
            if destination.endswith('.csv'):
                format = 'csv'
            elif destination.endswith('.json'):
                format = 'json'
            elif destination.endswith('.xlsx'):
                format = 'excel'
            elif destination.endswith('.xml'):
                format = 'xml'
            else:
                format = 'csv'  # Default
        
        if format == 'csv':
            data.to_csv(destination, index=False, **options)
        elif format == 'json':
            data.to_json(destination, orient='records', **options)
        elif format == 'excel':
            data.to_excel(destination, index=False, **options)
        elif format == 'xml':
            # Basic XML generation
            root = ET.Element("data")
            for _, row in data.iterrows():
                record = ET.SubElement(root, "record")
                for col in data.columns:
                    elem = ET.SubElement(record, col)
                    elem.text = str(row[col])
            
            tree = ET.ElementTree(root)
            tree.write(destination)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return {
            'destination': destination,
            'format': format,
            'rows_saved': len(data),
            'columns_saved': len(data.columns)
        }
    
    async def _transform_data(self, data: Union[pd.DataFrame, Dict], 
                             transformations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply transformations to data."""
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        applied_transformations = []
        
        for transform in transformations:
            operation = transform.get('operation')
            params = transform.get('params', {})
            
            if operation == 'rename_columns':
                df.rename(columns=params.get('mapping', {}), inplace=True)
                applied_transformations.append(f"Renamed columns: {params.get('mapping', {})}")
            
            elif operation == 'drop_columns':
                columns_to_drop = params.get('columns', [])
                df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
                applied_transformations.append(f"Dropped columns: {columns_to_drop}")
            
            elif operation == 'add_column':
                column_name = params.get('name')
                column_value = params.get('value')
                if callable(column_value):
                    df[column_name] = df.apply(column_value, axis=1)
                else:
                    df[column_name] = column_value
                applied_transformations.append(f"Added column: {column_name}")
            
            elif operation == 'convert_type':
                column = params.get('column')
                new_type = params.get('type')
                if column in df.columns:
                    df[column] = df[column].astype(new_type)
                    applied_transformations.append(f"Converted {column} to {new_type}")
            
            elif operation == 'normalize':
                columns = params.get('columns', df.select_dtypes(include=[np.number]).columns)
                method = params.get('method', 'minmax')  # minmax, zscore
                
                for col in columns:
                    if col in df.columns:
                        if method == 'minmax':
                            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                        elif method == 'zscore':
                            df[col] = (df[col] - df[col].mean()) / df[col].std()
                
                applied_transformations.append(f"Normalized columns: {list(columns)} using {method}")
            
            elif operation == 'encode_categorical':
                columns = params.get('columns', df.select_dtypes(include=['object']).columns)
                method = params.get('method', 'onehot')  # onehot, label
                
                for col in columns:
                    if col in df.columns:
                        if method == 'onehot':
                            dummies = pd.get_dummies(df[col], prefix=col)
                            df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
                        elif method == 'label':
                            df[col] = pd.Categorical(df[col]).codes
                
                applied_transformations.append(f"Encoded categorical columns: {list(columns)} using {method}")
            
            elif operation == 'date_features':
                column = params.get('column')
                if column in df.columns:
                    df[column] = pd.to_datetime(df[column])
                    df[f'{column}_year'] = df[column].dt.year
                    df[f'{column}_month'] = df[column].dt.month
                    df[f'{column}_day'] = df[column].dt.day
                    df[f'{column}_weekday'] = df[column].dt.weekday
                    applied_transformations.append(f"Extracted date features from {column}")
        
        return {
            'transformed_data': df.to_dict('records'),
            'data_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist()
            },
            'applied_transformations': applied_transformations
        }
    
    async def _filter_data(self, data: Union[pd.DataFrame, Dict],
                          filters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Filter data based on conditions."""
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        original_rows = len(df)
        applied_filters = []
        
        for filter_config in filters:
            column = filter_config.get('column')
            operator = filter_config.get('operator')
            value = filter_config.get('value')
            
            if column not in df.columns:
                continue
            
            if operator == 'equals':
                mask = df[column] == value
            elif operator == 'not_equals':
                mask = df[column] != value
            elif operator == 'greater_than':
                mask = df[column] > value
            elif operator == 'less_than':
                mask = df[column] < value
            elif operator == 'greater_equal':
                mask = df[column] >= value
            elif operator == 'less_equal':
                mask = df[column] <= value
            elif operator == 'contains':
                mask = df[column].astype(str).str.contains(str(value), na=False)
            elif operator == 'in':
                mask = df[column].isin(value if isinstance(value, list) else [value])
            elif operator == 'not_in':
                mask = ~df[column].isin(value if isinstance(value, list) else [value])
            elif operator == 'null':
                mask = df[column].isnull()
            elif operator == 'not_null':
                mask = df[column].notnull()
            elif operator == 'regex':
                mask = df[column].astype(str).str.match(str(value), na=False)
            else:
                continue
            
            df = df[mask]
            applied_filters.append(f"{column} {operator} {value}")
        
        return {
            'filtered_data': df.to_dict('records'),
            'data_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'original_rows': original_rows,
                'filtered_rows': len(df),
                'rows_removed': original_rows - len(df)
            },
            'applied_filters': applied_filters
        }
    
    async def _aggregate_data(self, data: Union[pd.DataFrame, Dict],
                             group_by: List[str],
                             aggregations: Dict[str, Union[str, List[str]]]) -> Dict[str, Any]:
        """Aggregate data by groups."""
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Validate group_by columns
        valid_group_cols = [col for col in group_by if col in df.columns]
        if not valid_group_cols:
            raise ValueError("No valid group_by columns found")
        
        # Prepare aggregation dictionary
        agg_dict = {}
        for column, operations in aggregations.items():
            if column in df.columns:
                if isinstance(operations, str):
                    operations = [operations]
                agg_dict[column] = operations
        
        # Perform aggregation
        grouped = df.groupby(valid_group_cols)
        aggregated = grouped.agg(agg_dict)
        
        # Flatten column names if multi-level
        if isinstance(aggregated.columns, pd.MultiIndex):
            aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]
        
        # Reset index to make group columns regular columns
        aggregated.reset_index(inplace=True)
        
        return {
            'aggregated_data': aggregated.to_dict('records'),
            'data_info': {
                'rows': len(aggregated),
                'columns': len(aggregated.columns),
                'group_columns': valid_group_cols,
                'aggregation_columns': list(agg_dict.keys())
            },
            'group_counts': grouped.size().to_dict()
        }
    
    async def _join_data(self, left_data: Union[pd.DataFrame, Dict],
                        right_data: Union[pd.DataFrame, Dict],
                        join_type: str = 'inner',
                        left_on: Optional[Union[str, List[str]]] = None,
                        right_on: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
        """Join two datasets."""
        if isinstance(left_data, dict):
            left_df = pd.DataFrame(left_data)
        else:
            left_df = left_data.copy()
        
        if isinstance(right_data, dict):
            right_df = pd.DataFrame(right_data)
        else:
            right_df = right_data.copy()
        
        # Determine join keys
        if left_on is None and right_on is None:
            # Find common columns
            common_cols = list(set(left_df.columns) & set(right_df.columns))
            if not common_cols:
                raise ValueError("No common columns found for join")
            join_keys = common_cols[0] if len(common_cols) == 1 else common_cols
        else:
            join_keys = left_on
        
        # Perform join
        if join_type == 'inner':
            result = left_df.merge(right_df, left_on=left_on, right_on=right_on, how='inner')
        elif join_type == 'left':
            result = left_df.merge(right_df, left_on=left_on, right_on=right_on, how='left')
        elif join_type == 'right':
            result = left_df.merge(right_df, left_on=left_on, right_on=right_on, how='right')
        elif join_type == 'outer':
            result = left_df.merge(right_df, left_on=left_on, right_on=right_on, how='outer')
        else:
            raise ValueError(f"Unsupported join type: {join_type}")
        
        return {
            'joined_data': result.to_dict('records'),
            'data_info': {
                'rows': len(result),
                'columns': len(result.columns),
                'left_rows': len(left_df),
                'right_rows': len(right_df),
                'join_type': join_type,
                'join_keys': join_keys
            }
        }
    
    async def _analyze_data(self, data: Union[pd.DataFrame, Dict],
                           analysis_type: str = 'summary') -> Dict[str, Any]:
        """Perform data analysis."""
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        if analysis_type == 'summary':
            # Descriptive statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            analysis = {
                'overview': {
                    'total_rows': len(df),
                    'total_columns': len(df.columns),
                    'numeric_columns': len(numeric_cols),
                    'categorical_columns': len(categorical_cols),
                    'missing_values': df.isnull().sum().to_dict(),
                    'duplicate_rows': df.duplicated().sum()
                },
                'numeric_summary': df[numeric_cols].describe().to_dict() if len(numeric_cols) > 0 else {},
                'categorical_summary': {}
            }
            
            # Categorical summary
            for col in categorical_cols:
                analysis['categorical_summary'][col] = {
                    'unique_values': df[col].nunique(),
                    'top_values': df[col].value_counts().head(5).to_dict(),
                    'missing_count': df[col].isnull().sum()
                }
        
        elif analysis_type == 'correlation':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                raise ValueError("Need at least 2 numeric columns for correlation analysis")
            
            correlation_matrix = df[numeric_cols].corr()
            analysis = {
                'correlation_matrix': correlation_matrix.to_dict(),
                'strong_correlations': []
            }
            
            # Find strong correlations (> 0.7 or < -0.7)
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    col1 = correlation_matrix.columns[i]
                    col2 = correlation_matrix.columns[j]
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        analysis['strong_correlations'].append({
                            'column1': col1,
                            'column2': col2,
                            'correlation': round(corr_value, 3)
                        })
        
        elif analysis_type == 'outliers':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            analysis = {'outliers': {}}
            
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                analysis['outliers'][col] = {
                    'count': len(outliers),
                    'percentage': round(len(outliers) / len(df) * 100, 2),
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'outlier_values': outliers[col].tolist()[:10]  # Limit to 10
                }
        
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
        
        return {
            'analysis_type': analysis_type,
            'analysis_results': analysis
        }
    
    async def _clean_data(self, data: Union[pd.DataFrame, Dict],
                         cleaning_operations: List[str]) -> Dict[str, Any]:
        """Clean data using various operations."""
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        original_rows = len(df)
        applied_operations = []
        
        for operation in cleaning_operations:
            if operation == 'remove_duplicates':
                before = len(df)
                df.drop_duplicates(inplace=True)
                after = len(df)
                applied_operations.append(f"Removed {before - after} duplicate rows")
            
            elif operation == 'remove_null_rows':
                before = len(df)
                df.dropna(inplace=True)
                after = len(df)
                applied_operations.append(f"Removed {before - after} rows with null values")
            
            elif operation == 'fill_null_values':
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                categorical_cols = df.select_dtypes(include=['object']).columns
                
                # Fill numeric with median
                for col in numeric_cols:
                    if df[col].isnull().any():
                        df[col].fillna(df[col].median(), inplace=True)
                
                # Fill categorical with mode
                for col in categorical_cols:
                    if df[col].isnull().any():
                        mode_value = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown'
                        df[col].fillna(mode_value, inplace=True)
                
                applied_operations.append("Filled null values with median/mode")
            
            elif operation == 'remove_outliers':
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                before = len(df)
                
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                
                after = len(df)
                applied_operations.append(f"Removed {before - after} outlier rows")
            
            elif operation == 'trim_strings':
                text_cols = df.select_dtypes(include=['object']).columns
                for col in text_cols:
                    df[col] = df[col].astype(str).str.strip()
                applied_operations.append(f"Trimmed whitespace from text columns")
            
            elif operation == 'standardize_case':
                text_cols = df.select_dtypes(include=['object']).columns
                for col in text_cols:
                    df[col] = df[col].astype(str).str.lower()
                applied_operations.append("Standardized text to lowercase")
        
        return {
            'cleaned_data': df.to_dict('records'),
            'data_info': {
                'original_rows': original_rows,
                'cleaned_rows': len(df),
                'rows_removed': original_rows - len(df),
                'columns': len(df.columns)
            },
            'applied_operations': applied_operations
        }
    
    async def _convert_format(self, data: Any, 
                             from_format: str,
                             to_format: str) -> Dict[str, Any]:
        """Convert data between different formats."""
        conversion_result = None
        
        # Load data based on from_format
        if from_format == 'csv':
            if isinstance(data, str):
                df = pd.read_csv(io.StringIO(data))
            else:
                df = pd.DataFrame(data)
        elif from_format == 'json':
            if isinstance(data, str):
                json_data = json.loads(data)
                df = pd.DataFrame(json_data)
            else:
                df = pd.DataFrame(data)
        elif from_format == 'xml':
            # Simple XML to DataFrame conversion
            if isinstance(data, str):
                root = ET.fromstring(data)
                records = []
                for child in root:
                    record = {}
                    for subchild in child:
                        record[subchild.tag] = subchild.text
                    records.append(record)
                df = pd.DataFrame(records)
            else:
                df = pd.DataFrame(data)
        else:
            df = pd.DataFrame(data)
        
        # Convert to target format
        if to_format == 'csv':
            conversion_result = df.to_csv(index=False)
        elif to_format == 'json':
            conversion_result = df.to_json(orient='records', indent=2)
        elif to_format == 'xml':
            # Simple DataFrame to XML conversion
            root = ET.Element("data")
            for _, row in df.iterrows():
                record = ET.SubElement(root, "record")
                for col in df.columns:
                    elem = ET.SubElement(record, col)
                    elem.text = str(row[col])
            conversion_result = ET.tostring(root, encoding='unicode')
        elif to_format == 'html':
            conversion_result = df.to_html(index=False)
        elif to_format == 'dict':
            conversion_result = df.to_dict('records')
        else:
            raise ValueError(f"Unsupported target format: {to_format}")
        
        return {
            'from_format': from_format,
            'to_format': to_format,
            'converted_data': conversion_result,
            'data_info': {
                'rows': len(df),
                'columns': len(df.columns)
            }
        }
    
    async def _validate_data(self, data: Union[pd.DataFrame, Dict],
                            validation_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against rules."""
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'column_validations': {}
        }
        
        for column, rules in validation_rules.items():
            if column not in df.columns:
                validation_results['errors'].append(f"Column '{column}' not found")
                validation_results['valid'] = False
                continue
            
            column_results = {'valid': True, 'errors': [], 'warnings': []}
            
            if 'required' in rules and rules['required']:
                null_count = df[column].isnull().sum()
                if null_count > 0:
                    column_results['errors'].append(f"{null_count} null values found")
                    column_results['valid'] = False
            
            if 'type' in rules:
                expected_type = rules['type']
                if expected_type == 'numeric':
                    non_numeric = df[~pd.to_numeric(df[column], errors='coerce').notna()]
                    if len(non_numeric) > 0:
                        column_results['errors'].append(f"{len(non_numeric)} non-numeric values")
                        column_results['valid'] = False
                elif expected_type == 'date':
                    try:
                        pd.to_datetime(df[column])
                    except:
                        column_results['errors'].append("Contains non-date values")
                        column_results['valid'] = False
            
            if 'min_value' in rules:
                min_value = rules['min_value']
                violations = df[df[column] < min_value]
                if len(violations) > 0:
                    column_results['errors'].append(f"{len(violations)} values below minimum {min_value}")
                    column_results['valid'] = False
            
            if 'max_value' in rules:
                max_value = rules['max_value']
                violations = df[df[column] > max_value]
                if len(violations) > 0:
                    column_results['errors'].append(f"{len(violations)} values above maximum {max_value}")
                    column_results['valid'] = False
            
            if 'pattern' in rules:
                pattern = rules['pattern']
                non_matching = df[~df[column].astype(str).str.match(pattern)]
                if len(non_matching) > 0:
                    column_results['errors'].append(f"{len(non_matching)} values don't match pattern")
                    column_results['valid'] = False
            
            if 'unique' in rules and rules['unique']:
                duplicates = df[column].duplicated().sum()
                if duplicates > 0:
                    column_results['errors'].append(f"{duplicates} duplicate values found")
                    column_results['valid'] = False
            
            validation_results['column_validations'][column] = column_results
            if not column_results['valid']:
                validation_results['valid'] = False
                validation_results['errors'].extend([f"{column}: {error}" for error in column_results['errors']])
        
        return validation_results
    
    async def _sample_data(self, data: Union[pd.DataFrame, Dict],
                          method: str = 'random',
                          size: Optional[int] = None,
                          fraction: Optional[float] = None) -> Dict[str, Any]:
        """Sample data using various methods."""
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        if size is None and fraction is None:
            fraction = 0.1  # Default 10%
        
        if method == 'random':
            if fraction:
                sampled = df.sample(frac=fraction)
            else:
                sampled = df.sample(n=min(size, len(df)))
        
        elif method == 'systematic':
            if fraction:
                step = int(1 / fraction)
            else:
                step = len(df) // size
            sampled = df.iloc[::step]
        
        elif method == 'stratified':
            # Simple stratified sampling on first categorical column
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) == 0:
                raise ValueError("No categorical columns found for stratified sampling")
            
            strat_col = categorical_cols[0]
            if fraction:
                sampled = df.groupby(strat_col).apply(lambda x: x.sample(frac=fraction)).reset_index(drop=True)
            else:
                group_sizes = df[strat_col].value_counts()
                proportional_sizes = (group_sizes / len(df) * size).round().astype(int)
                sampled_groups = []
                
                for category, group_size in proportional_sizes.items():
                    if group_size > 0:
                        group_data = df[df[strat_col] == category]
                        sampled_groups.append(group_data.sample(n=min(group_size, len(group_data))))
                
                sampled = pd.concat(sampled_groups).reset_index(drop=True)
        
        else:
            raise ValueError(f"Unsupported sampling method: {method}")
        
        return {
            'sampled_data': sampled.to_dict('records'),
            'sample_info': {
                'original_rows': len(df),
                'sampled_rows': len(sampled),
                'sampling_method': method,
                'sampling_ratio': round(len(sampled) / len(df), 3)
            }
        }
    
    async def _pivot_data(self, data: Union[pd.DataFrame, Dict],
                         index: Union[str, List[str]],
                         columns: Union[str, List[str]],
                         values: Union[str, List[str]],
                         aggfunc: str = 'mean') -> Dict[str, Any]:
        """Create pivot table from data."""
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Validate columns exist
        all_cols = [index] if isinstance(index, str) else index
        all_cols += [columns] if isinstance(columns, str) else columns
        all_cols += [values] if isinstance(values, str) else values
        
        missing_cols = [col for col in all_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")
        
        # Create pivot table
        pivot_table = df.pivot_table(
            index=index,
            columns=columns,
            values=values,
            aggfunc=aggfunc,
            fill_value=0
        )
        
        # Flatten column names if multi-level
        if isinstance(pivot_table.columns, pd.MultiIndex):
            pivot_table.columns = ['_'.join(map(str, col)).strip() for col in pivot_table.columns.values]
        
        # Reset index
        pivot_table.reset_index(inplace=True)
        
        return {
            'pivot_data': pivot_table.to_dict('records'),
            'pivot_info': {
                'rows': len(pivot_table),
                'columns': len(pivot_table.columns),
                'index_columns': index,
                'pivot_columns': columns,
                'value_columns': values,
                'aggregation_function': aggfunc
            }
        }
