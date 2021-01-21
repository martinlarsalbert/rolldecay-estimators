import rolldecayestimators.lambdas as lambdas
from scipy.optimize import least_squares

def residual_cubic(x, y, phi_a, omega0):
    """
    Residual function for least square fit
    """
    B_1 = x[0]
    B_2 = x[1]
    B_3 = x[2]

    B_e_pred = lambdas.B_e_lambda_cubic(B_1=B_1, B_2=B_2, B_3=B_3, omega0=omega0, phi_a=phi_a)
    B_e_true = y
    error = B_e_true - B_e_pred
    return error

def fit_cubic(y, phi_a, omega0, B_1_0=0,B_2_0=0,B_3_0=0):

    ## Use least square fit of B_44 as a function of phi_a to determine B_1, B_2 and B_3:
    x0 = [B_1_0,
          B_2_0,
          B_3_0,
          ]
    kwargs = {
        'y': y,
        'omega0': omega0,
        'phi_a':phi_a,
    }

    result = least_squares(fun=residual_cubic, x0=x0, kwargs=kwargs, method='lm')
    assert result.success

    output = {
        'B_1':result.x[0],
        'B_2': result.x[1],
        'B_3': result.x[2],
    }

    return output

def residual_quadratic(x, y, phi_a, omega0):
    """
    Residual function for least square fit
    """
    B_1 = x[0]
    B_2 = x[1]

    B_e_pred = lambdas.B_e_lambda(B_1=B_1, B_2=B_2, omega0=omega0, phi_a=phi_a)
    B_e_true = y
    error = B_e_true - B_e_pred
    return error

def fit_quadratic(y, phi_a, omega0, B_1_0=0,B_2_0=0):

    ## Use least square fit of B_44 as a function of phi_a to determine B_1, B_2 and B_3:
    x0 = [B_1_0,
          B_2_0,
          ]
    kwargs = {
        'y': y,
        'omega0': omega0,
        'phi_a':phi_a,
    }

    result = least_squares(fun=residual_quadratic, x0=x0, kwargs=kwargs, method='lm')
    assert result.success

    output = {
        'B_1':result.x[0],
        'B_2': result.x[1],
    }

    return output