def add(a: int | float, b: int | float) -> int | float:
    """Return the sum of a and b."""
    return a + b


def subtract(a: int | float, b: int | float) -> int | float:
    """Return the difference of a and b."""
    return a - b


def multiply(a: int | float, b: int | float) -> int | float:
    """Return the product of a and b."""
    return a * b


def divide(a: int | float, b: int | float) -> int | float:
    """Return the quotient of a divided by b. Raises ValueError if b is 0."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
