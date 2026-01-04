# Validators

PyEval provides flexible validators for input validation.

---

## Quick Validation Functions

### validate_predictions

Validate prediction arrays.

```python
from pyeval import validate_predictions

y_true = [1, 0, 1, 1, 0]
y_pred = [1, 0, 0, 1, 0]

# Validates and returns True, or raises ValueError
validate_predictions(y_true, y_pred)

# With options
validate_predictions(
    y_true, y_pred,
    allow_empty=False,
    check_same_length=True,
    check_same_classes=True
)
```

### validate_probabilities

Validate probability arrays.

```python
from pyeval import validate_probabilities

probs = [0.1, 0.3, 0.6]

# Basic validation (values between 0 and 1)
validate_probabilities(probs)

# Require sum to 1 (for single prediction)
validate_probabilities(probs, require_sum_one=True)

# Multi-class probabilities
multi_probs = [[0.1, 0.9], [0.7, 0.3], [0.5, 0.5]]
validate_probabilities(multi_probs, require_sum_one=True)
```

---

## Composable Validators

### TypeValidator

Validate value types.

```python
from pyeval import TypeValidator

# Single type
int_validator = TypeValidator(int)
result = int_validator.validate(42)
print(result.is_valid)  # True

result = int_validator.validate("42")
print(result.is_valid)  # False
print(result.errors)    # ["Expected int, got str"]

# Multiple types
num_validator = TypeValidator((int, float))
result = num_validator.validate(3.14)
print(result.is_valid)  # True
```

### ListValidator

Validate list properties.

```python
from pyeval import ListValidator

# Basic list validation
validator = ListValidator()
result = validator.validate([1, 2, 3])
print(result.is_valid)  # True

# With constraints
validator = ListValidator(
    element_type=int,
    min_length=1,
    max_length=100,
    unique=False
)

result = validator.validate([1, 2, 3])
print(result.is_valid)  # True

result = validator.validate([])
print(result.is_valid)  # False
print(result.errors)    # ["List length 0 below minimum 1"]

result = validator.validate([1, 2, "3"])
print(result.is_valid)  # False
print(result.errors)    # ["Element at index 2 is not int"]
```

### NumericValidator

Validate numeric values.

```python
from pyeval import NumericValidator

# Range validation
prob_validator = NumericValidator(min_val=0, max_val=1)
result = prob_validator.validate(0.5)
print(result.is_valid)  # True

result = prob_validator.validate(1.5)
print(result.is_valid)  # False
print(result.errors)    # ["Value 1.5 exceeds maximum 1"]

# Integer validation
int_validator = NumericValidator(integer_only=True)
result = int_validator.validate(3.14)
print(result.is_valid)  # False

# Positive only
pos_validator = NumericValidator(min_val=0, exclude_min=True)
result = pos_validator.validate(0)
print(result.is_valid)  # False (0 excluded)
```

---

## Schema Validation

### SchemaValidator

Validate complex data structures.

```python
from pyeval import SchemaValidator, FieldSchema, ListValidator, NumericValidator

# Define schema
schema = SchemaValidator([
    FieldSchema('y_true', ListValidator(min_length=1)),
    FieldSchema('y_pred', ListValidator(min_length=1)),
    FieldSchema('threshold', NumericValidator(0, 1), required=False, default=0.5),
    FieldSchema('average', TypeValidator(str), required=False, default='binary')
])

# Validate data
data = {
    'y_true': [1, 0, 1, 1, 0],
    'y_pred': [1, 0, 0, 1, 0]
}

result = schema.validate(data)
print(result.is_valid)  # True
print(result.validated_data)
# {'y_true': [1,0,1,1,0], 'y_pred': [1,0,0,1,0], 'threshold': 0.5, 'average': 'binary'}

# Invalid data
data = {
    'y_true': [],
    'y_pred': [1, 0]
}

result = schema.validate(data)
print(result.is_valid)  # False
print(result.errors)    # ["y_true: List length 0 below minimum 1"]
```

### Nested Schemas

```python
from pyeval import SchemaValidator, FieldSchema, ListValidator

# Nested schema for evaluation config
config_schema = SchemaValidator([
    FieldSchema('name', TypeValidator(str)),
    FieldSchema('metrics', ListValidator(element_type=str, min_length=1)),
    FieldSchema('data', SchemaValidator([
        FieldSchema('y_true', ListValidator()),
        FieldSchema('y_pred', ListValidator())
    ]))
])

config = {
    'name': 'experiment_1',
    'metrics': ['accuracy', 'f1'],
    'data': {
        'y_true': [1, 0, 1],
        'y_pred': [1, 0, 0]
    }
}

result = config_schema.validate(config)
print(result.is_valid)  # True
```

---

## Custom Validators

### Creating Custom Validators

```python
from pyeval import Validator, ValidationResult

class BinaryLabelsValidator(Validator):
    """Validates that all labels are binary (0 or 1)."""
    
    def validate(self, value):
        if not isinstance(value, (list, tuple)):
            return ValidationResult(
                is_valid=False,
                errors=["Input must be a list or tuple"]
            )
        
        invalid = [v for v in value if v not in (0, 1)]
        if invalid:
            return ValidationResult(
                is_valid=False,
                errors=[f"Invalid labels found: {invalid}. Expected 0 or 1."]
            )
        
        return ValidationResult(is_valid=True)

# Use custom validator
validator = BinaryLabelsValidator()
result = validator.validate([1, 0, 1, 0])
print(result.is_valid)  # True

result = validator.validate([1, 0, 2, 0])
print(result.is_valid)  # False
```

### Combining Validators

```python
from pyeval import CompositeValidator, ListValidator, TypeValidator

# Combine multiple validators
combined = CompositeValidator([
    TypeValidator(list),
    ListValidator(min_length=1, max_length=1000),
    BinaryLabelsValidator()
])

result = combined.validate([1, 0, 1, 0])
print(result.is_valid)  # True

result = combined.validate([1, 0, 2])
print(result.is_valid)  # False
print(result.errors)    # Shows all validation errors
```

---

## Validation Results

### ValidationResult

```python
from pyeval import ValidationResult

# Success
result = ValidationResult(is_valid=True, validated_value=[1, 0, 1])

print(result.is_valid)        # True
print(result.errors)          # []
print(result.validated_value) # [1, 0, 1]

# Failure
result = ValidationResult(
    is_valid=False,
    errors=["Length mismatch", "Invalid values"],
    context={'field': 'y_pred'}
)

print(result.is_valid)  # False
print(result.errors)    # ["Length mismatch", "Invalid values"]
print(result.context)   # {'field': 'y_pred'}

# Raise on invalid
result.raise_if_invalid()  # Raises ValueError if not valid
```

---

## Validation Decorators

### Using validators as decorators

```python
from pyeval import validated

@validated({
    'y_true': ListValidator(min_length=1),
    'y_pred': ListValidator(min_length=1),
    'threshold': NumericValidator(0, 1)
})
def custom_metric(y_true, y_pred, threshold=0.5):
    """Automatically validates inputs."""
    return compute(y_true, y_pred, threshold)

# Valid call
result = custom_metric([1,0,1], [1,0,0], 0.5)

# Invalid call - raises ValueError
result = custom_metric([], [1,0,0])
# ValueError: y_true: List length 0 below minimum 1
```

---

## Complete Validator Reference

| Validator | Description |
|-----------|-------------|
| `TypeValidator` | Validates value types |
| `ListValidator` | Validates list properties |
| `NumericValidator` | Validates numeric ranges |
| `SchemaValidator` | Validates complex schemas |
| `FieldSchema` | Defines schema fields |
| `CompositeValidator` | Combines multiple validators |
| `validate_predictions` | Quick prediction validation |
| `validate_probabilities` | Quick probability validation |

---

## Best Practices

1. **Validate early** - Check inputs at function entry
2. **Provide clear errors** - Help users fix issues
3. **Use schemas for complex data** - Better organization
4. **Combine validators** - Build complex validation from simple parts
5. **Cache validation results** - Avoid re-validating same data
