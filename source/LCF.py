import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


def estimate_lcf(pn_est_df, r=None, dim=None, est_name="LCF"):
    """
    Estimates the Local Correlation Function (LCF) using a smooth approximation
    of the empirical estimate of the number of points.

    Parameters:
        pn_est_df (pd.DataFrame): DataFrame with columns 'r' (distance) and 'pn' (estimated number of points)
        r (np.ndarray, optional): Vector of values for the argument r at which LCF(r) should be evaluated.
        dim (int, optional): The number of basis functions for spline approximation.
        est_name (str): The name of the column that contains the LCF's empirical estimate.

    Returns:
        pd.DataFrame: DataFrame with columns 'r', 'theo' (theoretical LCF), and est_name (empirical LCF)
    """
    if r is None:
        r = pn_est_df["r"].values

    # Find the first non-zero index
    non_zero_indices = np.where(pn_est_df["pn"] != 0)[0]
    if len(non_zero_indices) == 0:
        lcf_df = pd.DataFrame({"r": r, "theo": 0, est_name: -1})
        return lcf_df

    first_non_zero_ind = non_zero_indices[0]

    # Handle cases where there are no NaN values or all NaN values
    if np.any(np.isnan(pn_est_df["pn"])):
        nan_indices = np.where(np.isnan(pn_est_df["pn"]))[0]
        if len(nan_indices) == 0:
            last_ind = len(pn_est_df) - 1
        else:
            last_ind = min(len(pn_est_df) - 1, nan_indices[0] - 1)
    else:
        last_ind = len(pn_est_df) - 1

    pn_est_defined = pn_est_df.iloc[first_non_zero_ind : last_ind + 1]

    if dim is None:
        dim = int(np.sqrt(len(pn_est_df)))

    # Ensure the spline degree is within the valid range
    dim = max(2, dim)  # Minimum 2 basis functions to ensure a valid spline degree
    spline_degree = min(dim - 1, 5)  # Ensure degree is at most 5

    spline = interpolate.UnivariateSpline(
        pn_est_defined["r"], pn_est_defined["pn"], s=0, k=spline_degree, ext="raise"
    )
    spline_derivative = spline.derivative()

    r_li = np.where(r >= pn_est_defined["r"].iloc[0])[0]
    if len(r_li) > 0:
        r_li = r_li[0]
    else:
        r_li = 0

    r_hi = np.where(r > pn_est_defined["r"].iloc[-1])[0]
    if len(r_hi) > 0:
        r_hi = r_hi[0] - 1
    else:
        r_hi = len(r) - 1

    if r_li <= r_hi:
        r_def = r[r_li : r_hi + 1]
        pn = spline(r_def)
        pn_deriv = spline_derivative(r_def)
        pn_deriv = np.maximum(pn_deriv, 0)

        lcf = np.where((pn > 0) & (pn_deriv >= 0), compute_lcf(r_def, pn, pn_deriv), -1)

        num_ll_pad = r_li
        num_na_pad = len(r) - r_hi - 1
    else:
        lcf = np.full(len(r), np.nan)
        num_ll_pad = len(r)
        num_na_pad = 0

    lcf = np.concatenate([np.full(num_ll_pad, -1), lcf, np.full(num_na_pad, np.nan)])
    lcf_df = pd.DataFrame({"r": r, "theo": 0})
    lcf_df[est_name] = lcf

    return lcf_df


def compute_lcf(r, pn, pn_deriv, lcf_lims=(-1, 1)):
    """
    Compute the Local Correlation Function (LCF).

    Parameters:
        r (np.ndarray): Vector of distances
        pn (np.ndarray): Estimated number of points within distance r
        pn_deriv (np.ndarray): Derivative of the estimated number of points
        lcf_lims (tuple): Lower and upper limits of the LCF

    Returns:
        np.ndarray: LCF estimate at distances r
    """
    if len(lcf_lims) != 2 or not all(isinstance(i, (int, float)) for i in lcf_lims):
        raise ValueError("lcf_lims should be a tuple of two numeric values.")

    if lcf_lims[0] > lcf_lims[1]:
        raise ValueError("The first limit of lcf_lims must be less than the second.")

    if not all(isinstance(arr, np.ndarray) for arr in [r, pn, pn_deriv]):
        raise ValueError("r, pn, and pn_deriv must be numpy arrays.")

    if len(r) != len(pn) or len(pn) != len(pn_deriv):
        raise ValueError("r, pn, and pn_deriv must be of the same length.")

    scale = lcf_lims[1] - lcf_lims[0]
    shift = lcf_lims[0]
    lcf = np.exp(-np.log(2) / 2 * r * pn_deriv / pn) * scale + shift
    return lcf


def estimate_point_counts(random_points, r_values):
    """
    Estimate the average number of other points near each point in random_points
    at distances lower than each value in r_values.

    Parameters:
        random_points (np.ndarray): Array of shape (N, 2) with the coordinates of the points.
        r_values (np.ndarray): Array of distances to estimate the average number of points within.

    Returns:
        pd.DataFrame: DataFrame with columns 'r' (distance) and 'pn' (estimated number of points).
    """
    # Create a KDTree for efficient spatial queries
    tree = cKDTree(random_points)
    num_points = len(random_points)

    # List to hold the average number of points within each distance r
    pn_list = []

    for r in r_values:
        # Query the KDTree for all points within distance r
        counts = [len(tree.query_ball_point(point, r)) - 1 for point in random_points]
        average_count = np.mean(counts)
        pn_list.append([r, average_count])

    # Convert list to DataFrame
    pn_est_df = pd.DataFrame(pn_list, columns=["r", "pn"])

    return pn_est_df


def generate_random_points(grid_size, num_points):
    """
    Generate random points uniformly distributed within a grid.

    Parameters:
        grid_size (float): The size of the grid (the range of coordinates).
        num_points (int): Number of random points to generate.

    Returns:
        np.ndarray: Array of random points.
    """
    x = np.random.uniform(0, grid_size, num_points)
    y = np.random.uniform(0, grid_size, num_points)
    return np.vstack([x, y]).T


def generate_triangular_lattice_pattern(grid_size, side_length):
    """
    Generate a triangular lattice pattern with fixed side length within a fixed square grid.

    Parameters:
        grid_size (float): The size of the square grid (both width and height).
        side_length (float): The side length of each triangle in the lattice.

    Returns:
        np.ndarray: Array of points forming the triangular lattice.
    """
    height = np.sqrt(3) / 2 * side_length

    num_cols = int(np.ceil(grid_size / side_length))
    num_rows = int(np.ceil(grid_size / height))

    x_coords = []
    y_coords = []

    for row in range(num_rows):
        for col in range(num_cols):
            x = col * side_length + (row % 2) * (side_length / 2)
            y = row * height

            if x <= grid_size and y <= grid_size:
                x_coords.append(x)
                y_coords.append(y)

    return np.vstack([x_coords, y_coords]).T


def generate_single_cluster(grid_size, cluster_radius, points_per_cluster):
    """
    Generate points within a single cluster with a fixed radius around a cluster center.

    Parameters:
        cluster_radius (float): Radius of the cluster.
        points_per_cluster (int): Number of points to generate in the cluster.

    Returns:
        np.ndarray: Array of points.
    """

    center = np.array([grid_size * np.random.rand(), grid_size * np.random.rand()])

    theta = np.linspace(0, 2 * np.pi, points_per_cluster)
    r = np.random.rand(points_per_cluster) * cluster_radius
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)

    return np.vstack([x, y]).T


def plot_lcf(x, ylim=(-1, 1), title="LCF Plot", AUC=None, dir=""):
    """
    Plot the LCF values.

    Parameters:
        x (pd.DataFrame): DataFrame containing the LCF values.
        ylim (tuple): Limits for the y-axis.
        title (str): Title of the plot.

    Returns:
        None
    """
    plt.figure()
    plt.plot(x["r"], x["LCF"], label=f"LCF with AUC = {np.round(AUC, 5)}", color="blue")
    plt.axhline(y=0, color="grey", linestyle="--")
    plt.title(title)
    plt.xlabel("r")
    plt.ylabel("LCF")
    plt.legend()
    plt.savefig(f"./main/results/LCF_plots{dir}/{title}.pdf")
    return True


def plot_points(points, title="Point Pattern"):
    """
    Plot the point pattern.

    Parameters:
        points (np.ndarray): Array of point coordinates.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], c="blue", marker="o")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig(f"./main/results/LCF_plots/tests/{title}.pdf")

    return True


def compute_normalized_auc(lcf_df, r_values, rmin=None, rmax=None):
    """
    Compute the normalized area under the LCF curve between rmin and rmax.

    Parameters:
        lcf_df (pd.DataFrame): DataFrame containing LCF values (with column 'LCF').
        r_values (np.ndarray): Array of r values corresponding to the LCF values.
        rmin (float): The minimum r value to consider for the integration (default is min(r_values)).
        rmax (float): The maximum r value to consider for the integration (default is max(r_values)).

    Returns:
        float: The normalized area under the LCF curve.
    """
    if rmin is None:
        rmin = np.min(r_values)
    if rmax is None:
        rmax = np.max(r_values)

    # Filter the LCF data between rmin and rmax
    mask = (r_values >= rmin) & (r_values <= rmax)
    r_filtered = r_values[mask]
    lcf_filtered = lcf_df["LCF"].values[mask]

    # Compute delta_r
    delta_r = rmax - rmin

    # Compute area under the curve using the trapezoidal rule
    area_under_curve = np.trapz(lcf_filtered, r_filtered)

    # Normalize by delta_r
    normalized_auc = area_under_curve / delta_r

    return normalized_auc


def test_LCF():
    # Parameters
    grid_size = 10
    num_points = 1000
    side_length = 0.5
    grid_size = 10
    cluster_radius = 0.5
    r_values = np.linspace(0, 1, 100)

    # random points
    random_points = generate_random_points(grid_size, num_points)
    pn_est_df = estimate_point_counts(random_points, r_values)
    random_lcf_df = estimate_lcf(pn_est_df, r=r_values)

    plot_points(random_points, title="Random Points Pattern")
    plot_lcf(
        random_lcf_df,
        title="LCF for Random Points Pattern",
        AUC=compute_normalized_auc(random_lcf_df, r_values, rmin=0.1, rmax=1),
        dir="/tests",
    )

    # lattice
    lattice_points = generate_triangular_lattice_pattern(grid_size, side_length)
    pn_est_df = estimate_point_counts(lattice_points, r_values)
    lattice_lcf_df = estimate_lcf(pn_est_df, r=r_values)

    plot_points(
        lattice_points, title="Triangular Lattice Pattern with Fixed Side Length"
    )
    plot_lcf(
        lattice_lcf_df,
        title="LCF for Triangular Lattice Pattern with Fixed Side Length",
        AUC=compute_normalized_auc(lattice_lcf_df, r_values, rmin=0.1, rmax=1),
        dir="/tests/",
    )

    # cluster
    cluster_points = generate_single_cluster(grid_size, cluster_radius, num_points)
    pn_est_df = estimate_point_counts(cluster_points, r_values)
    cluster_lcf_df = estimate_lcf(pn_est_df, r=r_values)

    plot_points(cluster_points, title="Single Centered Cluster Pattern")
    plot_lcf(
        cluster_lcf_df,
        title="LCF for Single Centered Cluster Pattern",
        AUC=compute_normalized_auc(cluster_lcf_df, r_values, rmin=0.1, rmax=1),
        dir="/tests/",
    )

    return True
