"""
Dataset validation utilities.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd


def validate_dataset(dataset_path: str) -> Dict[str, Any]:
    """
    Validate that the dataset is readable and suitable for analysis.
    
    Args:
        dataset_path: Path to the dataset file
        
    Returns:
        Dictionary with validation results
    """
    
    validation_result = {
        "valid": False,
        "readable": False,
        "format": None,
        "size": 0,
        "shape": None,
        "columns": [],
        "errors": [],
        "warnings": []
    }
    
    try:
        # Check if file exists
        dataset_file = Path(dataset_path)
        if not dataset_file.exists():
            validation_result["errors"].append(f"Dataset file does not exist: {dataset_path}")
            return validation_result
        
        # Check file size
        file_size = dataset_file.stat().st_size
        validation_result["size"] = file_size
        
        if file_size == 0:
            validation_result["errors"].append("Dataset file is empty")
            return validation_result
        
        if file_size > 100 * 1024 * 1024:  # 100MB
            validation_result["warnings"].append(f"Large dataset ({file_size / 1024 / 1024:.1f}MB) - processing may be slow")
        
        # Determine file format
        file_extension = dataset_file.suffix.lower()
        validation_result["format"] = file_extension
        
        # Try to read the dataset
        if file_extension == '.csv':
            try:
                # Try different encodings
                for encoding in ['utf-8', 'latin1', 'cp1252']:
                    try:
                        df = pd.read_csv(dataset_path, encoding=encoding, nrows=5)  # Just read first few rows for validation
                        validation_result["readable"] = True
                        break
                    except UnicodeDecodeError:
                        continue
                
                if not validation_result["readable"]:
                    validation_result["errors"].append("Could not read CSV file - encoding issues")
                    return validation_result
                    
            except Exception as e:
                validation_result["errors"].append(f"Error reading CSV file: {str(e)}")
                return validation_result
        
        else:
            validation_result["errors"].append(f"Unsupported file format: {file_extension}. Only .csv files are supported currently.")
            return validation_result
        
        # Get dataset info (read full dataset for shape info)
        try:
            full_df = pd.read_csv(dataset_path, encoding='utf-8')
            validation_result["shape"] = full_df.shape
            validation_result["columns"] = full_df.columns.tolist()
            
            # Additional validation checks
            if full_df.empty:
                validation_result["errors"].append("Dataset is empty (no rows)")
                return validation_result
            
            if len(full_df.columns) == 0:
                validation_result["errors"].append("Dataset has no columns")
                return validation_result
            
            # Check for reasonable size limits
            if full_df.shape[0] > 1000000:  # 1M rows
                validation_result["warnings"].append(f"Very large dataset ({full_df.shape[0]:,} rows) - consider sampling")
            
            if full_df.shape[1] > 1000:  # 1000 columns
                validation_result["warnings"].append(f"Many columns ({full_df.shape[1]}) - analysis may be complex")
            
            # Check data quality
            missing_pct = (full_df.isnull().sum().sum() / (full_df.shape[0] * full_df.shape[1])) * 100
            if missing_pct > 50:
                validation_result["warnings"].append(f"High percentage of missing data ({missing_pct:.1f}%)")
            
            validation_result["valid"] = True
            
        except Exception as e:
            validation_result["errors"].append(f"Error analyzing dataset structure: {str(e)}")
            
    except Exception as e:
        validation_result["errors"].append(f"Unexpected error during validation: {str(e)}")
    
    return validation_result


def get_dataset_summary(dataset_path: str) -> Optional[Dict[str, Any]]:
    """
    Get a quick summary of the dataset for display purposes.
    
    Args:
        dataset_path: Path to the dataset file
        
    Returns:
        Dictionary with dataset summary or None if failed
    """
    
    try:
        df = pd.read_csv(dataset_path, nrows=1000)  # Sample first 1000 rows
        
        summary = {
            "shape": df.shape,
            "columns": len(df.columns),
            "numeric_columns": len(df.select_dtypes(include=['number']).columns),
            "categorical_columns": len(df.select_dtypes(include=['object']).columns),
            "missing_values": df.isnull().sum().sum(),
            "duplicate_rows": df.duplicated().sum(),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f}MB"
        }
        
        # Get sample of column names (first 10)
        summary["sample_columns"] = df.columns[:10].tolist()
        if len(df.columns) > 10:
            summary["sample_columns"].append("...")
        
        return summary
        
    except Exception:
        return None