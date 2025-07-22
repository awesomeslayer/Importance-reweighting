import logging
import math
from typing import Optional, Tuple, Union

import numpy as np
import scipy.stats as stats
from cvxopt import matrix, solvers
from scipy.special import logsumexp
from scipy.stats import gaussian_kde
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity

from .KL_LSCV import KL_find_bw

log = logging.getLogger("__main__")


def density_estimation(conf, hyp_params_dict, test_gen_dict, bw):
    if bw == "KL":
        bw_temp = KL_find_bw(
            conf,
            np.vstack([test_gen_dict["g_train"], test_gen_dict["g_test"]]),
            np.vstack([test_gen_dict["p_train"], test_gen_dict["p_test"]]),
            hyp_params_dict["beta"],
            hyp_params_dict["KL_enable"],
            hyp_params_dict["estim_type"],
        )
    else:
        bw_temp = bw

    if hyp_params_dict["estim_type"] == "sklearn":
        kde = KernelDensity(kernel="gaussian", bandwidth=bw_temp).fit(
            np.vstack([test_gen_dict["g_train"], test_gen_dict["g_test"]])
        )
        g_estim = lambda X: kde.score_samples(X)
        bw_temp = kde.bandwidth_

    elif hyp_params_dict["estim_type"] == "scipy":
        kde = gaussian_kde(
            np.vstack([test_gen_dict["g_train"], test_gen_dict["g_test"]]).T,
            bw_method=bw_temp,
        )
        g_estim = lambda X: np.log(kde.evaluate(X.T))
        bw_temp = kde.covariance_factor()

    log.debug(f"bw = {bw}, bw_temp = {bw_temp}")
    return g_estim, bw_temp



def ISE(err, p, g, g_sample):
    return logsumexp(err(g_sample) + p(g_sample) - g(g_sample)) - np.log(
        g_sample.shape[0]
    )


# def Classifier_error(g_train, p_train, g_test, p_test, err_func):
#     """
#     Estimate error on the target distribution using density ratio from a classifier.

#     Parameters:
#     - g_train: Training samples from source distribution g
#     - p_train: Training samples from target distribution p
#     - g_test: Test samples from source distribution g
#     - err_func: Function that computes errors on g_test samples

#     Returns:
#     - Estimated error on the target distribution
#     """
#     # Create training data for classifier
#     X_train = np.vstack([g_train, p_train])
#     y_train = np.concatenate([np.zeros(len(g_train)), np.ones(len(p_train))])

#     # Train logistic regression classifier
#     clf = GradientBoostingClassifier(n_estimators=100)

#     clf.fit(X_train, y_train)

#     # Predict P(s=1|x) for g_test samples
#     probs_target = clf.predict_proba(g_test)[:, 1]  # P(s=1|x)

#     # Calculate weights according to Equation 12
#     p_s1 = len(p_train) / (len(g_train) + len(p_train))  # P(s=1)
#     p_s0 = 1 - p_s1  # P(s=0)

#     # Compute weights: P(s=1)/P(s=0) * (1/P(s=1|x) - 1)
#     weights =  p_s0/p_s1*(probs_target)/(1-probs_target)

#     # Clip weights for numerical stability
#     weights = np.clip(weights, 1e-7, 10000)

#     log_weights =  np.log(weights)
#     # Compute errors
#     errors = err_func(g_test)

#     # Estimate weighted average error

#     return logsumexp(errors + log_weights) - np.log(len(g_test))


def Classifier_error(g_train, p_train, g_test, p_test, err_func):
    """
    Estimate error on the target distribution using density ratio from a classifier.
    This version aims to correctly implement the domain classifier based weighting
    while preserving the user's specified final calculation structure.

    Parameters:
    - g_train: Training samples from source distribution g
    - p_train: Training samples from target distribution p
    - g_test: Test samples from source distribution g
    - p_test: Test samples from target distribution p (not used for error calc, but for consistency)
    - err_func: Function that computes errors on g_test samples (assumed to be linear scale losses)

    Returns:
    - A log-transformed weighted value: log( (1/N_g_test) * sum(exp(errors_i) * weights_i) )
    """

    # Epsilon values for numerical stability
    epsilon_prob = 0  # For probabilities before division or log
    epsilon_denom = 1e-10  # For denominators in ratios if they can be zero

    if len(g_train) == 0:
        # Cannot train domain classifier or define source priors properly
        # Or err_func might fail on empty g_test if g_test depends on g_train size
        print(
            "Warning: g_train is empty. Cannot compute Classifier_error reliably. Returning NaN."
        )
        return np.nan

    if len(p_train) == 0:
        # If p_train is empty, cannot estimate target distribution properties for weighting.
        # Weights would be ill-defined (e.g., division by zero in prior ratio).
        # In this scenario, one might default to unweighted error or indicate failure.
        print(
            "Warning: p_train is empty. Density ratio weights are ill-defined. Returning NaN."
        )
        # As an alternative, one could compute unweighted error:
        # errors = err_func(g_test)
        # return np.log(np.mean(np.exp(errors))) # if the exp(error) logic is kept
        # or simply np.mean(errors) if returning linear scale error
        return np.nan

    # 1. Train the domain classifier
    # Combined training data for the domain classifier
    X_train = np.vstack([g_train, p_train])
    # Labels for domain classifier: 1 for source (g_train), 0 for target (p_train)
    # This convention aligns with s=1 for source in the paper's derivation for Equation 12
    y_train = np.concatenate([np.ones(len(g_train)), np.zeros(len(p_train))])

    clf = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3
    )  # liblinear is often good for smaller datasets
    clf.fit(X_train, y_train)

    # 2. Get probability estimates for g_test samples
    # Ensure we get P(s=1|x), i.e., probability of being from source domain
    # clf.classes_ will be [0, 1] due to y_train. Class 1 is source.
    class_1_idx = -1
    try:
        class_1_idx = np.where(clf.classes_ == 1)[0][0]
    except IndexError:
        # This should not happen if y_train contains 1s and 0s
        raise ValueError(
            "Class 1 (source) or Class 0 (target) not found in domain classifier's classes."
        )

    # P(s=1|x) for g_test samples (probability of x being from source domain)
    probs_s_eq_1_for_g_test = clf.predict_proba(g_test)[:, class_1_idx]

    # 3. Calculate density ratio components based on Equation 12
    # Density ratio p(x|target)/p(x|source) approx = [P(s=1)/P(s=0)] * [P(s=0|x)/P(s=1|x)]
    # where P(s=1) is prior for source, P(s=0) is prior for target.

    # Prior ratio: P(s=1)/P(s=0) estimated as N_g_train / N_p_train
    N_g_train = float(len(g_train))
    N_p_train = float(len(p_train))

    # Denominator N_p_train is already checked not to be zero at the start
    prior_ratio_mult = N_g_train / N_p_train

    # Likelihood ratio from classifier: P(s=0|x) / P(s=1|x)
    # P(s=0|x) = 1 - P(s=1|x)
    # Add epsilon_prob to prevent division by zero and log(0) if P(s=1|x) is 0 or 1
    p_s1_clipped = probs_s_eq_1_for_g_test
    p_s0_clipped = 1.0 - p_s1_clipped  # This will also be clipped away from 0

    likelihood_ratio_from_clf = p_s0_clipped / p_s1_clipped

    # Combine to get weights
    # weights = prior_ratio_mult * likelihood_ratio_from_clf
    raw_weights = prior_ratio_mult * likelihood_ratio_from_clf

    # Clip weights for numerical stability (as in the original snippet)
    original_weights = raw_weights.copy()  # For logging clipped count
    weights = np.clip(raw_weights, 1e-7, 1000)  # User's original clip values

    # Log any clipped weights (count)
    clipped_count = np.sum(original_weights != weights)
    # if clipped_count > 0:
    #     print(f"Classifier_error: Clipped {clipped_count}/{len(weights)} weights.")

    # 4. Calculate log_weights
    # It's safer to log the already clipped positive weights
    log_weights = np.log(weights)

    # 5. Compute errors using the provided err_func
    # These errors are assumed to be on a linear scale, not log-scale.
    errors = err_func(g_test)
    if not isinstance(errors, np.ndarray):
        errors = np.array(errors)

    if len(errors) != len(g_test):
        raise ValueError(
            f"Length of errors ({len(errors)}) from err_func does not match length of g_test ({len(g_test)})."
        )

    # 6. Compute the final "weighted_error" as per the user's logsumexp structure
    # This calculates: log ( (1/N_g_test) * sum_i ( exp(errors_i) * weights_i ) )
    # where N_g_test = len(g_test)
    if (
        len(g_test) == 0
    ):  # Should be caught by len(g_train) check if g_test derived from g_train
        print(
            "Warning: g_test is empty. Cannot compute final weighted error. Returning NaN."
        )
        return np.nan

    # The terms inside logsumexp are errors_i + log_weights_i
    terms_for_lse = errors + log_weights

    weighted_error = logsumexp(terms_for_lse) - np.log(float(len(g_test)))

    return weighted_error


def ISE_deg(err, p, g, g_sample, lam):
    if lam != 0:
        return logsumexp(err(g_sample) + lam * (p(g_sample) - g(g_sample))) - np.log(
            g_sample.shape[0]
        )
    else:
        return logsumexp(err(g_sample)) - np.log(g_sample.shape[0])


def MCE(err, p_sample):
    return logsumexp(err(p_sample)) - np.log(p_sample.shape[0])


def clip(a, b_min, b_max):
    if a < b_min:
        return b_min, 1
    elif a > b_max:
        return b_max, 1
    else:
        return a, 0


def smooth_clip(x, eps):
    return (1 + eps) / (1 + (2 * eps / (1 - eps)) * np.exp(-x))


def ISE_clip(err, p, g, g_sample, eps, smooth_flag=True, thrhold=0.95, clip_step=0.001):
    log = logging.getLogger("__main__")
    if eps == "quantile":
        for eps_temp in np.arange(1 - clip_step, -clip_step, -clip_step):
            clipped_array = []
            num_clipped = 0
            for p_elem, g_elem in zip(p(g_sample), g(g_sample)):
                weight = np.exp(p_elem - g_elem)
                clipped_weight, i = clip(weight, 1 - eps_temp, 1 + eps_temp)
                clipped_array.append(np.log(clipped_weight))
                num_clipped = num_clipped + i

            if num_clipped > len(g_sample) * thrhold:
                eps = eps_temp
                log.debug(f"eps_clip_quantile = {eps}")
                break
    else:
        clipped_array = []
        for p_elem, g_elem in zip(p(g_sample), g(g_sample)):
            weight = np.exp(p_elem - g_elem)
            clipped_weight, i = clip(weight, 1 - eps, 1 + eps)
            clipped_array.append(np.log(clipped_weight))

    if smooth_flag:
        clipped_array = []
        for p_elem, g_elem in zip(p(g_sample), g(g_sample)):
            weight = np.exp(p_elem - g_elem)
            clipped_array.append(np.log(smooth_clip(weight, eps)))

    return logsumexp(clipped_array + err(g_sample)) - np.log(g_sample.shape[0])


import numpy as np
from scipy import stats
from scipy.stats import spearmanr


def rmse(x_err, y_err, confidence=0.95):
    errors = [(x - y) ** 2 for x, y in zip(x_err, y_err)]
    rmse_value = np.sqrt(np.mean(errors))

    n = len(errors)
    std_dev = np.sqrt(np.var(errors, ddof=1))
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha / 2, n - 1)
    margin_of_error = t_critical * (std_dev / np.sqrt(n))

    return rmse_value, (rmse_value - margin_of_error, rmse_value + margin_of_error)


def rmspe(x_err, y_err, confidence=0.95):
    # Проверка деления на ноль и фильтрация y_err = 0
    relative_errors = []
    for x, y in zip(x_err, y_err):
        if y == 0:
            continue  # Пропускаем нулевые значения y
        relative_error = ((x - y) / y) ** 2
        relative_errors.append(relative_error)

    if not relative_errors:  # Все y = 0
        return 1337, (1337, 1337)

    rmspe_value = np.sqrt(np.mean(relative_errors)) * 100

    n = len(relative_errors)
    std_dev = np.sqrt(np.var(relative_errors, ddof=1))
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha / 2, n - 1)
    margin_of_error = t_critical * (std_dev / np.sqrt(n)) * 100

    return rmspe_value, (rmspe_value - margin_of_error, rmspe_value + margin_of_error)


def corr(x_err, y_err, confidence=0.95, method="spearman"):
    # Удаляем пары, где есть NaN
    x_clean, y_clean = [], []
    for x, y in zip(x_err, y_err):
        if not np.isnan(x) and not np.isnan(y):
            x_clean.append(x)
            y_clean.append(y)

    if len(x_clean) < 2:
        return 0.0, (0.0, 0.0)  # Недостаточно данных

    # Вычисляем корреляцию Спирмена
    corr_value, p_value = spearmanr(x_clean, y_clean)

    # Доверительный интервал через преобразование Фишера (для любых корреляций)
    n = len(x_clean)
    if method == "pearson":
        # Для Пирсона используем z-преобразование Фишера
        z = np.arctanh(corr_value)
        se = 1 / np.sqrt(n - 3)
    else:
        # Для Спирмена приближенный метод (менее точен)
        z = np.arctanh(corr_value)
        se = 1.06 / np.sqrt(n - 3)  # Эмпирическая поправка

    alpha = 1 - confidence
    z_critical = stats.norm.ppf(1 - alpha / 2)
    lower_z = z - z_critical * se
    upper_z = z + z_critical * se

    lower = np.tanh(lower_z)
    upper = np.tanh(upper_z)

    return corr_value, (lower, upper)


# def corr(x_err, y_err, confidence=0.95):
#     """
#     Вычисляет ковариацию между двумя массивами и её доверительный интервал.

#     :param x_err: Первый массив значений.
#     :param y_err: Второй массив значений.
#     :param confidence: Уровень доверия для интервала (по умолчанию 0.95).
#     :return: Ковариация и её доверительный интервал (нижняя и верхняя границы).
#     """
#     # Удаляем пары, где есть NaN
#     x_clean, y_clean = [], []
#     for x, y in zip(x_err, y_err):
#         if not np.isnan(x) and not np.isnan(y):
#             x_clean.append(x)
#             y_clean.append(y)

#     if len(x_clean) < 2:
#         return 0.0, (0.0, 0.0)  # Недостаточно данных

#     # Вычисляем ковариацию
#     cov_value = np.cov(x_clean, y_clean)[0, 1]  # Ковариация между x и y

#     # Вычисляем доверительный интервал для ковариации
#     n = len(x_clean)
#     std_dev = np.sqrt(np.var(x_clean) * np.sqrt(np.var(y_clean)))  # Оценка стандартного отклонения
#     alpha = 1 - confidence
#     t_critical = stats.t.ppf(1 - alpha / 2, n - 1)  # Критическое значение t-распределения
#     margin_of_error = t_critical * (std_dev / np.sqrt(n))  # Погрешность

#     # Доверительный интервал
#     lower = cov_value - margin_of_error
#     upper = cov_value + margin_of_error

#     return cov_value, (lower, upper)


def mape(x_err, y_err, confidence=0.95):
    errors = [abs(x - y) / y for x, y in zip(x_err, y_err) if y != 0]

    if not errors:
        return 1337, (1337, 1337)

    mape_value = np.mean(errors) * 100

    n = len(errors)
    std_dev = np.sqrt(np.var(errors, ddof=1))
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha / 2, n - 1)
    margin_of_error = t_critical * (std_dev / np.sqrt(n)) * 100

    return mape_value, (mape_value - margin_of_error, mape_value + margin_of_error)


def compute_rbf(X, Y, sigma=1.0):
    """
    Compute RBF kernel matrix K(x,y) = exp(-||x-y||^2 / (2 * sigma^2))

    Args:
        X (np.ndarray): First set of points, shape (n_samples_X, n_features)
        Y (np.ndarray): Second set of points, shape (n_samples_Y, n_features)
        sigma (float): Kernel bandwidth parameter

    Returns:
        np.ndarray: Kernel matrix with shape (n_samples_X, n_samples_Y)
    """
    X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y**2, axis=1).reshape(1, -1)
    dist_squared = X_norm + Y_norm - 2 * np.dot(X, Y.T)
    return np.exp(-dist_squared / (2 * sigma**2))


def adjust_sigma(X, factor=0.5):
    """
    Heuristic to adjust sigma based on data

    Args:
        X (np.ndarray): Data points, shape (n_samples, n_features)
        factor (float): Scaling factor

    Returns:
        float: Adjusted sigma value
    """
    # Median heuristic for bandwidth
    n_samples = X.shape[0]
    if n_samples > 1000:
        # Subsample for efficiency
        indices = np.random.choice(n_samples, 1000, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X

    X_norm = np.sum(X_sample**2, axis=1).reshape(-1, 1)
    dist_squared = X_norm + X_norm.T - 2 * np.dot(X_sample, X_sample.T)
    dist_squared = np.maximum(
        dist_squared, 0.0
    )  # Ensure non-negative due to numerical issues

    # Compute median of non-zero distances
    sigma = np.sqrt(np.median(dist_squared[dist_squared > 0]))
    if sigma < 1e-10:
        sigma = 1.0  # Fallback if median is too small

    return sigma * factor


def generate_rff_mapping(X: np.ndarray, num_features: int, sigma: float) -> np.ndarray:
    """
    Generate Random Fourier Features mapping for RBF kernel approximation

    Args:
        X (np.ndarray): Input data with shape (n_samples, n_features)
        num_features (int): Number of random features to generate
        sigma (float): Kernel bandwidth parameter

    Returns:
        np.ndarray: Transformed features with shape (n_samples, 2*num_features)
    """
    d = X.shape[1]

    # Random weights with appropriate scaling for RBF kernel
    # Note: 1/sigma is used because in RBF exp(-||x-y||^2/(2*sigma^2))
    # corresponds to a Gaussian spectral density with variance 1/sigma^2
    W = np.random.randn(d, num_features) / sigma

    # Random bias terms
    b = np.random.uniform(0, 2 * np.pi, size=num_features)

    # Compute features
    Z = np.cos(X @ W + b)

    # Scale appropriately
    Z = Z * np.sqrt(2.0 / num_features)

    return Z


def kernel_mean_matching(
    g_train: np.ndarray,
    g_test: np.ndarray,
    kern: str = "lin",
    B: float = 1.0,
    eps: Optional[float] = None,
    rff_dim: int = 500,
) -> np.ndarray:
    """
    Kernel Mean Matching implementation with multiple kernel options

    Args:
        g_train (np.ndarray): Training data, shape (nx, n_features)
        g_test (np.ndarray): Test data, shape (nz, n_features)
        kern (str): Kernel type: 'lin' (linear), 'rbf' (RBF kernel), or 'rff' (Random Fourier Features)
        B (float): Upper bound on the importance weights
        eps (Optional[float]): Regularization parameter
        rff_dim (int): Dimension for random Fourier features (only used when kern='rff')

    Returns:
        np.ndarray: Importance weights for test samples
    """
    nx = g_train.shape[0]
    nz = g_test.shape[0]

    if eps is None:
        eps = max(
            1e-6, B / np.sqrt(nz)
        )  # Avoid very small values for uniform distributions

    # Suppress CVXOPT output
    solvers.options["show_progress"] = False

    if kern == "lin":
        # Linear kernel
        K = np.dot(g_test, g_test.T)
        kappa = np.sum(np.dot(g_test, g_train.T) * float(nz) / float(nx), axis=1)

    elif kern == "rbf":
        # RBF kernel
        sigma = adjust_sigma(g_test)
        K = compute_rbf(g_test, g_test, sigma=sigma)
        kappa = (
            np.sum(compute_rbf(g_test, g_train, sigma=sigma), axis=1)
            * float(nz)
            / float(nx)
        )

    elif kern == "rff":
        # Random Fourier Features approximation of RBF kernel
        sigma = adjust_sigma(g_test)

        # Combine both datasets for consistent feature mapping
        X_combined = np.vstack([g_train, g_test])

        # Generate random Fourier features
        Z_combined = generate_rff_mapping(X_combined, rff_dim, sigma)

        # Split back into train and test
        Z_train = Z_combined[:nx]
        Z_test = Z_combined[nx:]

        # Compute kernel matrix and mean embedding using the RFF transformation
        K = np.dot(
            Z_test, Z_test.T
        )  # approximation of RBF kernel using inner product of features
        kappa = np.sum(np.dot(Z_test, Z_train.T) * float(nz) / float(nx), axis=1)

    else:
        raise ValueError('Unknown kernel. Available options are "lin", "rbf", or "rff"')

    # Convert to CVXOPT format
    K = matrix(K)
    kappa = matrix(kappa)

    # Regularization with dynamic epsilon
    G = matrix(
        np.vstack([np.ones((1, nz)), -np.ones((1, nz)), np.eye(nz), -np.eye(nz)])
    )
    h = matrix(
        np.hstack([nz * (1 + eps), nz * (eps - 1), B * np.ones(nz), np.zeros(nz)])
    )

    # Solve quadratic program
    sol = solvers.qp(K, -kappa, G, h)
    coef = np.array(sol["x"]).flatten()

    # Clip the coefficients to avoid extreme values
    coef = np.clip(coef, 0, B)

    return coef


def KMM_error(err, p_sample, g_sample, hyperparam):
    coef = kernel_mean_matching(p_sample, g_sample, kern="rbf", B=hyperparam)
    return logsumexp(err(g_sample) + np.log(coef)) - np.log(g_sample.shape[0])
