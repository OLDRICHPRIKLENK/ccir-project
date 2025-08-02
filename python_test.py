from sympy import symbols
from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorHead, tensorsymmetry

def simple_tensor_expression():
    # Define a tensor index type (e.g., space or Lorentz)
    Space = TensorIndexType('Space')

    # Define indices for that space
    i, j = tensor_indices('i j', Space)

    # Create a symmetric 2-index tensor head A_ij
    sym2 = tensorsymmetry([1]*2)  # Symmetric in 2 indices
    A = TensorHead('A', [sym2], [Space])

    # Symbolic indexed tensor
    T = A(i, j)

    print("Symbolic tensor expression A(i, j):")
    print(T)

    return T

# Run the function
if __name__ == "__main__":
    simple_tensor_expression()