import sympy as sp

# Define the variable
t = sp.symbols('t')

# Define the functions x_D(t) and y_D(t)
x_D = sp.Function('x_D')(t)
y_D = sp.sqrt(2.6**2 - x_D**2)

# Define the equation given in the problem
equation = (x_D - 1.5*sp.cos(4*sp.pi*t) - 11.2)**2 + (y_D - 1.5*sp.sin(4*sp.pi*t))**2 - 11.3**2

# Solve for x_D(t)
x_D_expr = sp.solve(equation, x_D)

# Substitute the first solution for x_D(t) in y_D
y_D_expr = sp.sqrt(2.6**2 - x_D_expr[0]**2)

# Differentiate the equation with respect to t
d_eq = sp.diff(equation, t)

# Solve for the first derivative of x_D(t)
x_D_prime = sp.solve(d_eq, sp.diff(x_D, t))[0]

# Define first derivative of y_D(t)
y_D_prime = sp.diff(y_D_expr, t)

# Find the second derivative of x_D(t) and y_D(t)
x_D_double_prime = sp.diff(x_D_prime, t)
y_D_double_prime = sp.diff(y_D_prime, t)

# Substitute x_D(t) and x_D'(t) into x_D''(t) and y_D''(t)
x_D_double_prime_sub = x_D_double_prime.subs(x_D, x_D_expr[0]).subs(sp.diff(x_D, t), x_D_prime)
y_D_double_prime_sub = y_D_double_prime.subs(x_D, x_D_expr[0]).subs(sp.diff(x_D, t), x_D_prime)

# Simplify the results
x_D_double_prime_simplified = sp.simplify(x_D_double_prime_sub)
y_D_double_prime_simplified = sp.simplify(y_D_double_prime_sub)

print("Second derivative of x_D(t) with substitution:", x_D_double_prime_simplified)
print("Second derivative of y_D(t) with substitution:", y_D_double_prime_simplified)
