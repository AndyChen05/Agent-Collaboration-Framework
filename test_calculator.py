import pytest
from calculator import add, subtract, multiply, divide


def test_add_integers():
    """Test add with integer arguments."""
    result = add(3, 5)
    assert result == 8
    assert isinstance(result, int)


def test_add_floats():
    """Test add with float arguments."""
    result = add(2.5, 3.1)
    assert result == pytest.approx(5.6)
    assert isinstance(result, float)


def test_subtract():
    """Test subtract with integers and floats."""
    result_int = subtract(10, 4)
    assert result_int == 6
    assert isinstance(result_int, int)

    result_float = subtract(7.5, 2.0)
    assert result_float == pytest.approx(5.5)
    assert isinstance(result_float, float)


def test_multiply():
    """Test multiply with integers."""
    result = multiply(3, 7)
    assert result == 21
    assert isinstance(result, int)


def test_divide_normal():
    """Test divide with normal cases."""
    result = divide(10, 3)
    assert result == pytest.approx(3.3333333)

    result_int = divide(8, 4)
    assert result_int == 2.0


def test_divide_by_zero():
    """Test divide by zero raises ValueError."""
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        divide(5, 0)


def test_multiply_negative():
    """Test multiply by a negative number."""
    result = multiply(6, -3)
    assert result == -18

    result_neg_neg = multiply(-4, -5)
    assert result_neg_neg == 20


def test_chaining():
    """Test chaining: add result fed into multiply."""
    sum_result = add(2, 3)        # 5
    product = multiply(sum_result, 4)  # 5 * 4 = 20
    assert product == 20

    # Chain subtract then divide
    diff = subtract(20, 5)        # 15
    quotient = divide(diff, 3)    # 15 / 3 = 5.0
    assert quotient == 5.0
