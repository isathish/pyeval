"""
PyEval Validators - Input Validation Utilities
===============================================

Provides validation utilities for:
- Type checking
- Range validation
- Schema validation
- Custom validators
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, TypeVar
from dataclasses import dataclass, field

T = TypeVar('T')


# =============================================================================
# Validation Results
# =============================================================================

@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def __bool__(self) -> bool:
        return self.is_valid
    
    def add_error(self, message: str):
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)
    
    def merge(self, other: 'ValidationResult'):
        """Merge with another result."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.is_valid:
            self.is_valid = False


# =============================================================================
# Base Validator
# =============================================================================

class Validator(ABC):
    """Abstract base class for validators."""
    
    @abstractmethod
    def validate(self, value: Any) -> ValidationResult:
        """Validate a value."""
        pass
    
    def __call__(self, value: Any) -> ValidationResult:
        return self.validate(value)


# =============================================================================
# Type Validators
# =============================================================================

class TypeValidator(Validator):
    """
    Validates that value is of specified type(s).
    
    Example:
        validator = TypeValidator(list, tuple)
        result = validator.validate([1, 2, 3])
    """
    
    def __init__(self, *types):
        self.types = types
    
    def validate(self, value: Any) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        if not isinstance(value, self.types):
            result.add_error(
                f"Expected type {self.types}, got {type(value).__name__}"
            )
        return result


class ListValidator(Validator):
    """
    Validates list with optional element type checking.
    
    Example:
        validator = ListValidator(element_type=int, min_length=1)
        result = validator.validate([1, 2, 3])
    """
    
    def __init__(self, element_type: type = None, 
                 min_length: int = None, max_length: int = None):
        self.element_type = element_type
        self.min_length = min_length
        self.max_length = max_length
    
    def validate(self, value: Any) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if not isinstance(value, (list, tuple)):
            result.add_error(f"Expected list or tuple, got {type(value).__name__}")
            return result
        
        if self.min_length is not None and len(value) < self.min_length:
            result.add_error(f"Length {len(value)} is less than minimum {self.min_length}")
        
        if self.max_length is not None and len(value) > self.max_length:
            result.add_error(f"Length {len(value)} exceeds maximum {self.max_length}")
        
        if self.element_type:
            for i, elem in enumerate(value):
                if not isinstance(elem, self.element_type):
                    result.add_error(
                        f"Element {i} has type {type(elem).__name__}, "
                        f"expected {self.element_type.__name__}"
                    )
        
        return result


class DictValidator(Validator):
    """
    Validates dictionary structure.
    
    Example:
        validator = DictValidator(required_keys=['name', 'value'])
        result = validator.validate({'name': 'test', 'value': 0.5})
    """
    
    def __init__(self, required_keys: List[str] = None,
                 optional_keys: List[str] = None,
                 value_types: Dict[str, type] = None):
        self.required_keys = required_keys or []
        self.optional_keys = optional_keys or []
        self.value_types = value_types or {}
    
    def validate(self, value: Any) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if not isinstance(value, dict):
            result.add_error(f"Expected dict, got {type(value).__name__}")
            return result
        
        # Check required keys
        for key in self.required_keys:
            if key not in value:
                result.add_error(f"Missing required key: {key}")
        
        # Check for unknown keys
        all_keys = set(self.required_keys) | set(self.optional_keys) | set(self.value_types.keys())
        for key in value.keys():
            if all_keys and key not in all_keys:
                result.add_warning(f"Unknown key: {key}")
        
        # Check value types
        for key, expected_type in self.value_types.items():
            if key in value and not isinstance(value[key], expected_type):
                result.add_error(
                    f"Key '{key}' has type {type(value[key]).__name__}, "
                    f"expected {expected_type.__name__}"
                )
        
        return result


# =============================================================================
# Numeric Validators
# =============================================================================

class NumericValidator(Validator):
    """
    Validates numeric values.
    
    Example:
        validator = NumericValidator(min_value=0, max_value=1)
        result = validator.validate(0.5)
    """
    
    def __init__(self, min_value: float = None, max_value: float = None,
                 allow_nan: bool = False, allow_inf: bool = False):
        self.min_value = min_value
        self.max_value = max_value
        self.allow_nan = allow_nan
        self.allow_inf = allow_inf
    
    def validate(self, value: Any) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if not isinstance(value, (int, float)):
            result.add_error(f"Expected numeric type, got {type(value).__name__}")
            return result
        
        import math
        
        if math.isnan(value) and not self.allow_nan:
            result.add_error("NaN values are not allowed")
        
        if math.isinf(value) and not self.allow_inf:
            result.add_error("Infinite values are not allowed")
        
        if self.min_value is not None and value < self.min_value:
            result.add_error(f"Value {value} is less than minimum {self.min_value}")
        
        if self.max_value is not None and value > self.max_value:
            result.add_error(f"Value {value} exceeds maximum {self.max_value}")
        
        return result


class ProbabilityValidator(NumericValidator):
    """Validates probability values (0 to 1)."""
    
    def __init__(self):
        super().__init__(min_value=0.0, max_value=1.0)


class PositiveValidator(NumericValidator):
    """Validates positive numeric values."""
    
    def __init__(self):
        super().__init__(min_value=0.0)


class IntegerValidator(Validator):
    """
    Validates integer values.
    
    Example:
        validator = IntegerValidator(min_value=0)
        result = validator.validate(5)
    """
    
    def __init__(self, min_value: int = None, max_value: int = None):
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, value: Any) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if not isinstance(value, int) or isinstance(value, bool):
            result.add_error(f"Expected integer, got {type(value).__name__}")
            return result
        
        if self.min_value is not None and value < self.min_value:
            result.add_error(f"Value {value} is less than minimum {self.min_value}")
        
        if self.max_value is not None and value > self.max_value:
            result.add_error(f"Value {value} exceeds maximum {self.max_value}")
        
        return result


# =============================================================================
# String Validators
# =============================================================================

class StringValidator(Validator):
    """
    Validates string values.
    
    Example:
        validator = StringValidator(min_length=1, max_length=100)
        result = validator.validate("hello")
    """
    
    def __init__(self, min_length: int = None, max_length: int = None,
                 pattern: str = None, allowed_chars: str = None):
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = pattern
        self.allowed_chars = allowed_chars
    
    def validate(self, value: Any) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if not isinstance(value, str):
            result.add_error(f"Expected string, got {type(value).__name__}")
            return result
        
        if self.min_length is not None and len(value) < self.min_length:
            result.add_error(f"Length {len(value)} is less than minimum {self.min_length}")
        
        if self.max_length is not None and len(value) > self.max_length:
            result.add_error(f"Length {len(value)} exceeds maximum {self.max_length}")
        
        if self.pattern:
            import re
            if not re.match(self.pattern, value):
                result.add_error(f"Value does not match pattern: {self.pattern}")
        
        if self.allowed_chars:
            for char in value:
                if char not in self.allowed_chars:
                    result.add_error(f"Invalid character: {char}")
                    break
        
        return result


class NonEmptyStringValidator(StringValidator):
    """Validates non-empty strings."""
    
    def __init__(self):
        super().__init__(min_length=1)


# =============================================================================
# Composite Validators
# =============================================================================

class AllOf(Validator):
    """
    Validates that all validators pass.
    
    Example:
        validator = AllOf(
            TypeValidator(list),
            ListValidator(min_length=1)
        )
        result = validator.validate([1, 2, 3])
    """
    
    def __init__(self, *validators: Validator):
        self.validators = validators
    
    def validate(self, value: Any) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        for validator in self.validators:
            sub_result = validator.validate(value)
            result.merge(sub_result)
        return result


class AnyOf(Validator):
    """
    Validates that at least one validator passes.
    
    Example:
        validator = AnyOf(
            TypeValidator(list),
            TypeValidator(tuple)
        )
        result = validator.validate([1, 2, 3])
    """
    
    def __init__(self, *validators: Validator):
        self.validators = validators
    
    def validate(self, value: Any) -> ValidationResult:
        results = []
        for validator in self.validators:
            sub_result = validator.validate(value)
            if sub_result.is_valid:
                return sub_result
            results.append(sub_result)
        
        # All failed - combine errors
        result = ValidationResult(is_valid=False)
        result.add_error("None of the validators passed")
        for sub in results:
            result.errors.extend(sub.errors)
        return result


class OptionalValidator(Validator):
    """
    Validates value if not None, allows None.
    
    Example:
        validator = OptionalValidator(NumericValidator(min_value=0))
        result = validator.validate(None)  # Valid
    """
    
    def __init__(self, validator: Validator):
        self.validator = validator
    
    def validate(self, value: Any) -> ValidationResult:
        if value is None:
            return ValidationResult(is_valid=True)
        return self.validator.validate(value)


# Alias for backward compatibility
Optional = OptionalValidator


# =============================================================================
# Metric-Specific Validators
# =============================================================================

class PredictionValidator(Validator):
    """
    Validates prediction arrays for metric computation.
    
    Example:
        validator = PredictionValidator()
        result = validator.validate({'y_true': [1, 0, 1], 'y_pred': [1, 1, 1]})
    """
    
    def __init__(self, require_same_length: bool = True,
                 require_non_empty: bool = True):
        self.require_same_length = require_same_length
        self.require_non_empty = require_non_empty
    
    def validate(self, value: Dict) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if not isinstance(value, dict):
            result.add_error("Expected dict with 'y_true' and 'y_pred' keys")
            return result
        
        if 'y_true' not in value or 'y_pred' not in value:
            result.add_error("Missing 'y_true' or 'y_pred' key")
            return result
        
        y_true, y_pred = value['y_true'], value['y_pred']
        
        # Check types
        for name, arr in [('y_true', y_true), ('y_pred', y_pred)]:
            if not isinstance(arr, (list, tuple)):
                result.add_error(f"'{name}' must be a list or tuple")
        
        if not result.is_valid:
            return result
        
        # Check non-empty
        if self.require_non_empty:
            if len(y_true) == 0 or len(y_pred) == 0:
                result.add_error("Input arrays cannot be empty")
        
        # Check same length
        if self.require_same_length:
            if len(y_true) != len(y_pred):
                result.add_error(
                    f"Length mismatch: y_true has {len(y_true)}, "
                    f"y_pred has {len(y_pred)}"
                )
        
        return result


class ProbabilityArrayValidator(Validator):
    """
    Validates arrays of probability values.
    
    Example:
        validator = ProbabilityArrayValidator()
        result = validator.validate([0.1, 0.3, 0.6])
    """
    
    def __init__(self, require_sum_one: bool = False, tolerance: float = 1e-6):
        self.require_sum_one = require_sum_one
        self.tolerance = tolerance
    
    def validate(self, value: Any) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if not isinstance(value, (list, tuple)):
            result.add_error("Expected list or tuple")
            return result
        
        for i, prob in enumerate(value):
            if not isinstance(prob, (int, float)):
                result.add_error(f"Element {i} is not numeric")
                continue
            if prob < 0 or prob > 1:
                result.add_error(f"Element {i}: {prob} is not in [0, 1]")
        
        if self.require_sum_one and result.is_valid:
            total = sum(value)
            if abs(total - 1.0) > self.tolerance:
                result.add_error(f"Probabilities sum to {total}, expected 1.0")
        
        return result


class ConfusionMatrixValidator(Validator):
    """
    Validates confusion matrix structure.
    
    Example:
        validator = ConfusionMatrixValidator()
        result = validator.validate([[10, 2], [3, 15]])
    """
    
    def validate(self, value: Any) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if not isinstance(value, (list, tuple)):
            result.add_error("Confusion matrix must be a 2D array")
            return result
        
        if len(value) == 0:
            result.add_error("Confusion matrix cannot be empty")
            return result
        
        n_rows = len(value)
        for i, row in enumerate(value):
            if not isinstance(row, (list, tuple)):
                result.add_error(f"Row {i} must be a list or tuple")
                continue
            if len(row) != n_rows:
                result.add_error(
                    f"Row {i} has {len(row)} elements, expected {n_rows} (square matrix)"
                )
            for j, cell in enumerate(row):
                if not isinstance(cell, (int, float)):
                    result.add_error(f"Cell [{i}][{j}] is not numeric")
                elif cell < 0:
                    result.add_error(f"Cell [{i}][{j}] is negative")
        
        return result


# =============================================================================
# Validation Functions
# =============================================================================

def validate_predictions(y_true: List, y_pred: List, 
                        raise_on_error: bool = True) -> ValidationResult:
    """
    Validate prediction arrays.
    
    Example:
        validate_predictions([1, 0, 1], [1, 1, 1])
    """
    validator = PredictionValidator()
    result = validator.validate({'y_true': y_true, 'y_pred': y_pred})
    
    if raise_on_error and not result.is_valid:
        raise ValueError("\n".join(result.errors))
    
    return result


def validate_probabilities(probs: List, require_sum_one: bool = False,
                          raise_on_error: bool = True) -> ValidationResult:
    """
    Validate probability array.
    
    Example:
        validate_probabilities([0.1, 0.3, 0.6], require_sum_one=True)
    """
    validator = ProbabilityArrayValidator(require_sum_one=require_sum_one)
    result = validator.validate(probs)
    
    if raise_on_error and not result.is_valid:
        raise ValueError("\n".join(result.errors))
    
    return result


def validate_range(value: float, min_val: float = None, max_val: float = None,
                  name: str = "value", raise_on_error: bool = True) -> ValidationResult:
    """
    Validate numeric value is within range.
    
    Example:
        validate_range(0.5, min_val=0, max_val=1, name="threshold")
    """
    validator = NumericValidator(min_value=min_val, max_value=max_val)
    result = validator.validate(value)
    
    if raise_on_error and not result.is_valid:
        raise ValueError(f"{name}: " + "\n".join(result.errors))
    
    return result


# =============================================================================
# Schema Validation
# =============================================================================

@dataclass
class FieldSchema:
    """Schema definition for a field."""
    name: str
    validator: Validator
    required: bool = True
    default: Any = None


class SchemaValidator(Validator):
    """
    Validates data against a schema.
    
    Example:
        schema = SchemaValidator([
            FieldSchema('y_true', ListValidator(min_length=1)),
            FieldSchema('y_pred', ListValidator(min_length=1)),
            FieldSchema('threshold', NumericValidator(0, 1), required=False, default=0.5)
        ])
        
        result = schema.validate({
            'y_true': [1, 0, 1],
            'y_pred': [1, 1, 1]
        })
    """
    
    def __init__(self, fields: List[FieldSchema]):
        self.fields = {f.name: f for f in fields}
    
    def validate(self, value: Dict) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if not isinstance(value, dict):
            result.add_error("Expected dictionary")
            return result
        
        # Validate each field
        for name, field_def in self.fields.items():
            if name not in value:
                if field_def.required:
                    result.add_error(f"Missing required field: {name}")
                continue
            
            field_result = field_def.validator.validate(value[name])
            if not field_result.is_valid:
                for error in field_result.errors:
                    result.add_error(f"Field '{name}': {error}")
                result.is_valid = False
            result.warnings.extend(field_result.warnings)
        
        # Check for unknown fields
        known_fields = set(self.fields.keys())
        for key in value.keys():
            if key not in known_fields:
                result.add_warning(f"Unknown field: {key}")
        
        return result
    
    def with_defaults(self, value: Dict) -> Dict:
        """Return value with defaults filled in."""
        result = dict(value)
        for name, field_def in self.fields.items():
            if name not in result and field_def.default is not None:
                result[name] = field_def.default
        return result


# =============================================================================
# Export all
# =============================================================================

__all__ = [
    # Results
    'ValidationResult',
    
    # Base
    'Validator',
    
    # Type validators
    'TypeValidator',
    'ListValidator',
    'DictValidator',
    
    # Numeric validators
    'NumericValidator',
    'ProbabilityValidator',
    'PositiveValidator',
    'IntegerValidator',
    
    # String validators
    'StringValidator',
    'NonEmptyStringValidator',
    
    # Composite validators
    'AllOf',
    'AnyOf',
    'OptionalValidator',
    'Optional',  # Alias for OptionalValidator
    
    # Metric validators
    'PredictionValidator',
    'ProbabilityArrayValidator',
    'ConfusionMatrixValidator',
    
    # Functions
    'validate_predictions',
    'validate_probabilities',
    'validate_range',
    
    # Schema
    'FieldSchema',
    'SchemaValidator',
]
