from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.sparse.linalg import LinearOperator, cg
from skimage import color, img_as_float, io
from skimage.metrics import structural_similarity
from skimage.transform import resize


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "docs" / "Original"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
DEFAULT_IMAGE_NAME = "5.1.14.tiff"
DEFAULT_IMAGE_SIZE = 256
IMAGE_EXTENSIONS = (".tif", ".tiff")
DEFAULT_QUALITY_PRESET = "visual"
QUALITY_PRESETS = {
    "fast": {
        "lambda_min": 1e-6,
        "lambda_max": 1e-1,
        "lambda_count": 12,
        "epsilon_count": 8,
        "epsilon_refine_steps": 3,
        "max_iter": 80,
    },
    "balanced": {
        "lambda_min": 1e-6,
        "lambda_max": 1e-1,
        "lambda_count": 24,
        "epsilon_count": 14,
        "epsilon_refine_steps": 5,
        "max_iter": 160,
    },
    "visual": {
        "lambda_min": 1e-8,
        "lambda_max": 1e0,
        "lambda_count": 60,
        "epsilon_count": 20,
        "epsilon_refine_steps": 6,
        "max_iter": 400,
    },
}

Matrix = np.ndarray
Vector = np.ndarray


@dataclass
class Candidate:
    index: int
    lambda_reg: float
    image: np.ndarray
    fidelity_error: float
    noise_penalty: float
    objective_value: float
    cg_info: int
    method: str = "weighted"
    epsilon_bound: float | None = None
    residual_norm: float | None = None
    roughness_norm: float | None = None
    reference_error_norm: float | None = None
    reference_mse: float | None = None
    reference_psnr: float | None = None
    reference_ssim: float | None = None
    is_pareto: bool = False
    compromise_score: float | None = None
    is_best_compromise: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?")
    parser.add_argument("--list-images", action="store_true")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--sigma-blur", type=float, default=2.0)
    parser.add_argument("--noise-level", type=float, default=0.01)
    parser.add_argument(
        "--quality-preset",
        choices=tuple(QUALITY_PRESETS),
        default=DEFAULT_QUALITY_PRESET,
    )
    parser.add_argument("--lambda-min", type=float, default=None)
    parser.add_argument("--lambda-max", type=float, default=None)
    parser.add_argument("--lambda-count", type=int, default=None)
    parser.add_argument(
        "--mols-method",
        choices=("weighted", "epsilon"),
        default="epsilon",
    )
    parser.add_argument("--epsilon-count", type=int, default=None)
    parser.add_argument("--epsilon-refine-steps", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--best-selection",
        choices=("reference", "compromise"),
        default="reference",
    )
    args = parser.parse_args()
    apply_quality_preset(args)
    return args


def apply_quality_preset(args: argparse.Namespace) -> None:
    preset = QUALITY_PRESETS[args.quality_preset]
    for name, value in preset.items():
        if getattr(args, name) is None:
            setattr(args, name, value)


def load_grayscale_image(path: Path, image_size: int) -> Matrix:
    if image_size <= 0:
        raise ValueError("image-size must be a positive integer.")

    image = img_as_float(io.imread(path))
    if image.ndim == 3:
        if image.shape[-1] == 4:
            image = image[..., :3]
        image = color.rgb2gray(image)
    if image.ndim != 2:
        raise ValueError(f"Expected grayscale/RGB image, got shape {image.shape}")

    return resize(
        image,
        (image_size, image_size),
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float64)


def vectorize(matrix: Matrix) -> Vector:
    return matrix.ravel()


def matrixize(vector: Vector, shape: tuple[int, int]) -> Matrix:
    return vector.reshape(shape)


def squared_l2_norm(vector: Vector) -> float:
    return float(np.linalg.norm(vector) ** 2)


def blur_operator(shape: tuple[int, int], sigma: float):
    def A(x: Vector) -> Vector:
        return ndimage.gaussian_filter(
            matrixize(x, shape),
            sigma=sigma,
            mode="reflect",
        ).ravel()

    return A


def laplacian_operator(x: Vector, shape: tuple[int, int]) -> Vector:
    X = matrixize(x, shape)
    LX = (
        -4.0 * X
        + np.roll(X, 1, axis=0)
        + np.roll(X, -1, axis=0)
        + np.roll(X, 1, axis=1)
        + np.roll(X, -1, axis=1)
    )
    return vectorize(LX)


def regularization_grid(lambda_min: float, lambda_max: float, count: int) -> np.ndarray:
    if lambda_min <= 0 or lambda_max <= 0:
        raise ValueError("lambda-min and lambda-max must be positive.")
    if lambda_min >= lambda_max:
        raise ValueError("lambda-min must be smaller than lambda-max.")
    if count < 2:
        raise ValueError("lambda-count must be at least 2.")
    return np.logspace(np.log10(lambda_min), np.log10(lambda_max), count)


def solve_tikhonov_candidate(
    index: int,
    lambda_reg: float,
    Y: Matrix,
    shape: tuple[int, int],
    A,
    max_iter: int,
    x0: Vector | None = None,
    method: str = "weighted",
    epsilon_bound: float | None = None,
) -> Candidate:
    y = vectorize(Y)
    n = y.size

    L = lambda x: laplacian_operator(x, shape)

    def normal_equation_matrix_vector_product(x: Vector) -> Vector:
        return A(A(x)) + lambda_reg * L(L(x))

    normal_matrix = LinearOperator(
        (n, n),
        matvec=normal_equation_matrix_vector_product,
    )
    rhs = A(y)
    x_hat, info = cg(
        normal_matrix,
        rhs,
        x0=y.copy() if x0 is None else x0.copy(),
        maxiter=max_iter,
        rtol=1e-5,
    )

    X_hat = np.clip(matrixize(x_hat, shape), 0.0, 1.0)
    x_hat_clipped = vectorize(X_hat)
    residual = A(x_hat_clipped) - y
    roughness = L(x_hat_clipped)
    fidelity_error = squared_l2_norm(residual)
    noise_penalty = squared_l2_norm(roughness)

    return Candidate(
        index=index,
        lambda_reg=float(lambda_reg),
        image=X_hat,
        fidelity_error=fidelity_error,
        noise_penalty=noise_penalty,
        objective_value=fidelity_error + float(lambda_reg) * noise_penalty,
        cg_info=info,
        method=method,
        epsilon_bound=epsilon_bound,
        residual_norm=float(np.sqrt(fidelity_error)),
        roughness_norm=float(np.sqrt(noise_penalty)),
    )


def solve_weighted_path(
    lambdas: np.ndarray,
    Y: Matrix,
    shape: tuple[int, int],
    A,
    max_iter: int,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    x0: Vector | None = None
    for index, lambda_reg in enumerate(lambdas):
        candidate = solve_tikhonov_candidate(
            index=index,
            lambda_reg=lambda_reg,
            Y=Y,
            shape=shape,
            A=A,
            max_iter=max_iter,
            x0=x0,
            method="weighted",
        )
        candidates.append(candidate)
        x0 = vectorize(candidate.image)
    return candidates


def epsilon_grid(min_epsilon: float, max_epsilon: float, count: int) -> np.ndarray:
    if count < 2:
        raise ValueError("epsilon-count must be at least 2.")
    min_epsilon = max(float(min_epsilon), np.finfo(float).tiny)
    max_epsilon = max(float(max_epsilon), min_epsilon * (1.0 + 1e-12))
    return np.geomspace(min_epsilon, max_epsilon, count)


def candidate_for_epsilon(
    epsilon: float,
    lambda_low: float,
    lambda_high: float,
    Y: Matrix,
    shape: tuple[int, int],
    A,
    max_iter: int,
    refine_steps: int,
    index: int,
    x0: Vector,
) -> Candidate:
    low = float(lambda_low)
    high = float(lambda_high)
    best_feasible: Candidate | None = None
    current_x0 = x0

    for _ in range(refine_steps):
        mid = float(np.sqrt(low * high))
        candidate = solve_tikhonov_candidate(
            index=index,
            lambda_reg=mid,
            Y=Y,
            shape=shape,
            A=A,
            max_iter=max_iter,
            x0=current_x0,
            method="epsilon",
            epsilon_bound=epsilon,
        )
        current_x0 = vectorize(candidate.image)
        if candidate.noise_penalty <= epsilon:
            best_feasible = candidate
            high = mid
        else:
            low = mid

    if best_feasible is None:
        best_feasible = solve_tikhonov_candidate(
            index=index,
            lambda_reg=high,
            Y=Y,
            shape=shape,
            A=A,
            max_iter=max_iter,
            x0=current_x0,
            method="epsilon",
            epsilon_bound=epsilon,
        )
    best_feasible.index = index
    best_feasible.method = "epsilon"
    best_feasible.epsilon_bound = epsilon
    best_feasible.objective_value = best_feasible.fidelity_error
    return best_feasible


def solve_epsilon_path(
    anchor_candidates: list[Candidate],
    Y: Matrix,
    shape: tuple[int, int],
    A,
    max_iter: int,
    epsilon_count: int,
    refine_steps: int,
) -> list[Candidate]:
    anchors = sorted(anchor_candidates, key=lambda candidate: candidate.lambda_reg)
    roughness_values = np.array([candidate.noise_penalty for candidate in anchors])
    epsilons = epsilon_grid(float(roughness_values.min()), float(roughness_values.max()), epsilon_count)
    epsilon_candidates: list[Candidate] = []

    for index, epsilon in enumerate(epsilons):
        feasible = [candidate for candidate in anchors if candidate.noise_penalty <= epsilon]
        if not feasible:
            anchor = max(anchors, key=lambda candidate: candidate.lambda_reg)
            candidate = solve_tikhonov_candidate(
                index=index,
                lambda_reg=anchor.lambda_reg,
                Y=Y,
                shape=shape,
                A=A,
                max_iter=max_iter,
                x0=vectorize(anchor.image),
                method="epsilon",
                epsilon_bound=float(epsilon),
            )
        else:
            feasible_index, first_feasible = min(
                (
                    (candidate_index, candidate)
                    for candidate_index, candidate in enumerate(anchors)
                    if candidate.noise_penalty <= epsilon
                ),
                key=lambda item: item[1].lambda_reg,
            )
            if feasible_index == 0:
                candidate = solve_tikhonov_candidate(
                    index=index,
                    lambda_reg=first_feasible.lambda_reg,
                    Y=Y,
                    shape=shape,
                    A=A,
                    max_iter=max_iter,
                    x0=vectorize(first_feasible.image),
                    method="epsilon",
                    epsilon_bound=float(epsilon),
                )
            else:
                previous = anchors[feasible_index - 1]
                candidate = candidate_for_epsilon(
                    epsilon=float(epsilon),
                    lambda_low=previous.lambda_reg,
                    lambda_high=first_feasible.lambda_reg,
                    Y=Y,
                    shape=shape,
                    A=A,
                    max_iter=max_iter,
                    refine_steps=refine_steps,
                    index=index,
                    x0=vectorize(first_feasible.image),
                )
        epsilon_candidates.append(candidate)

    return epsilon_candidates


def mark_pareto_front(candidates: list[Candidate]) -> None:
    F = np.array(
        [[candidate.fidelity_error, candidate.noise_penalty] for candidate in candidates],
        dtype=np.float64,
    )
    tolerance = 1e-12

    for i, candidate in enumerate(candidates):
        no_worse = np.all(F <= F[i] + tolerance, axis=1)
        strictly_better = np.any(F < F[i] - tolerance, axis=1)
        candidate.is_pareto = not bool(np.any(no_worse & strictly_better))


def mark_best_compromise(candidates: list[Candidate]) -> Candidate:
    pareto = [candidate for candidate in candidates if candidate.is_pareto]
    if not pareto:
        raise ValueError("No Pareto candidates were found.")

    F = np.array(
        [[candidate.fidelity_error, candidate.noise_penalty] for candidate in pareto],
        dtype=np.float64,
    )
    ideal = F.min(axis=0)
    nadir = F.max(axis=0)
    span = np.where(nadir > ideal, nadir - ideal, 1.0)
    normalized = (F - ideal) / span
    scores = np.linalg.norm(normalized, axis=1)

    best_index = int(np.argmin(scores))
    for candidate, score in zip(pareto, scores):
        candidate.compromise_score = float(score)
    pareto[best_index].is_best_compromise = True
    return pareto[best_index]


def mark_reference_quality(candidates: list[Candidate], original: Matrix) -> None:
    for candidate in candidates:
        reference_error = candidate.image - original
        mse = float(np.mean(reference_error ** 2))
        candidate.reference_error_norm = float(np.linalg.norm(reference_error))
        candidate.reference_mse = mse
        candidate.reference_psnr = float("inf") if mse == 0 else float(-10.0 * np.log10(mse))
        candidate.reference_ssim = float(
            structural_similarity(
                original,
                candidate.image,
                data_range=1.0,
            )
        )


def mark_selected_pareto_candidate(
    candidates: list[Candidate],
    original: Matrix,
    selection: str,
) -> Candidate:
    pareto = [candidate for candidate in candidates if candidate.is_pareto]
    if not pareto:
        raise ValueError("No Pareto candidates were found.")

    for candidate in candidates:
        candidate.is_best_compromise = False

    mark_reference_quality(candidates, original)
    compromise = mark_best_compromise(candidates)
    compromise.is_best_compromise = False

    if selection == "compromise":
        selected = compromise
    elif selection == "reference":
        selected = min(pareto, key=lambda candidate: candidate.reference_mse)
    else:
        raise ValueError(f"Unknown best-selection mode: {selection}")

    selected.is_best_compromise = True
    return selected


def write_metrics_csv(path: Path, candidates: list[Candidate]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "method",
                "lambda_reg",
                "epsilon_bound",
                "is_pareto",
                "is_best_compromise",
                "compromise_score",
                "residual_norm",
                "roughness_norm",
                "reference_error_norm",
                "reference_mse",
                "reference_psnr",
                "reference_ssim",
                "fidelity_error",
                "noise_penalty",
                "scalarized_or_primary_objective",
                "cg_info",
            ]
        )
        for candidate in candidates:
            writer.writerow(
                [
                    candidate.index,
                    candidate.method,
                    candidate.lambda_reg,
                    candidate.epsilon_bound,
                    candidate.is_pareto,
                    candidate.is_best_compromise,
                    candidate.compromise_score,
                    candidate.residual_norm,
                    candidate.roughness_norm,
                    candidate.reference_error_norm,
                    candidate.reference_mse,
                    candidate.reference_psnr,
                    candidate.reference_ssim,
                    candidate.fidelity_error,
                    candidate.noise_penalty,
                    candidate.objective_value,
                    candidate.cg_info,
                ]
            )


def write_experiment_summary_csv(path: Path, candidates: list[Candidate]) -> None:
    pareto = [candidate for candidate in candidates if candidate.is_pareto]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "lambda_reg",
                "residual_norm",
                "roughness_norm",
                "reference_mse",
                "reference_psnr",
                "reference_ssim",
                "interpretation",
            ]
        )
        if not pareto:
            return

        selected_indexes = sorted(
            {
                0,
                len(pareto) // 4,
                len(pareto) // 2,
                (3 * len(pareto)) // 4,
                len(pareto) - 1,
                next(
                    (
                        position
                        for position, candidate in enumerate(pareto)
                        if candidate.is_best_compromise
                    ),
                    len(pareto) // 2,
                ),
            }
        )
        for position in selected_indexes:
            candidate = pareto[position]
            if position == 0:
                interpretation = "smoothest / strongest regularization"
            elif position == len(pareto) - 1:
                interpretation = "sharpest / weakest regularization"
            elif candidate.is_best_compromise:
                interpretation = "selected Pareto reconstruction"
            else:
                interpretation = "intermediate Pareto trade-off"
            writer.writerow(
                [
                    candidate.index,
                    candidate.lambda_reg,
                    candidate.residual_norm,
                    candidate.roughness_norm,
                    candidate.reference_mse,
                    candidate.reference_psnr,
                    candidate.reference_ssim,
                    interpretation,
                ]
            )


def write_matrix(path: Path, matrix: Matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path.with_suffix(".csv"), matrix, delimiter=",", fmt="%.10f")
    np.save(path.with_suffix(".npy"), matrix)


def prepare_output_dir(output_dir: Path, input_dir: Path) -> None:
    output_path = output_dir.resolve()
    protected_paths = {
        ROOT.resolve(),
        input_dir.resolve(),
        input_dir.resolve().parent,
    }
    if output_path in protected_paths:
        raise ValueError(f"Refusing to empty protected directory: {output_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    for child in output_path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def save_pareto_plot(path: Path, candidates: list[Candidate], selection: str) -> None:
    dominated = [candidate for candidate in candidates if not candidate.is_pareto]
    pareto = [candidate for candidate in candidates if candidate.is_pareto]
    best = [candidate for candidate in candidates if candidate.is_best_compromise]

    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    if dominated:
        axis.scatter(
            [candidate.fidelity_error for candidate in dominated],
            [candidate.noise_penalty for candidate in dominated],
            c="0.7",
            label="Dominated candidate",
        )
    axis.plot(
        [candidate.fidelity_error for candidate in pareto],
        [candidate.noise_penalty for candidate in pareto],
        color="crimson",
        linewidth=1.2,
        alpha=0.65,
    )
    axis.scatter(
        [candidate.fidelity_error for candidate in pareto],
        [candidate.noise_penalty for candidate in pareto],
        c="crimson",
        marker="*",
        s=95,
        label="Pareto-optimal solution",
    )
    if best:
        best_label = (
            "Closest Pareto image to original"
            if selection == "reference"
            else "Best normalized compromise"
        )
        axis.scatter(
            [candidate.fidelity_error for candidate in best],
            [candidate.noise_penalty for candidate in best],
            c="gold",
            edgecolors="black",
            marker="D",
            s=80,
            label=best_label,
        )

    for candidate in pareto:
        axis.annotate(
            f"#{candidate.index}",
            (candidate.fidelity_error, candidate.noise_penalty),
            fontsize=7,
        )

    axis.set_xlabel("Fidelity error J1 = ||Ax - y||^2")
    axis.set_ylabel("Noise control J2 = ||Lx||^2")
    axis.set_title("Pareto front from MOLS/Tikhonov candidate solutions")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_reconstruction_grid(
    path: Path,
    original: np.ndarray,
    blurred_noisy: np.ndarray,
    candidates: list[Candidate],
    title: str,
) -> None:
    selected = candidates[: min(6, len(candidates))]
    panel_count = 2 + len(selected)
    figure, axes = plt.subplots(1, panel_count, figsize=(3 * panel_count, 4))
    panels = [("Original", original), ("Blurred + noise", blurred_noisy)]

    for candidate in selected:
        panels.append(
            (
                f"#{candidate.index} Pareto={candidate.is_pareto}\n"
                f"lambda={candidate.lambda_reg:.1e}",
                candidate.image,
            )
        )

    for axis, (panel_title, image) in zip(axes, panels):
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(panel_title)
        axis.axis("off")

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_best_compromise_grid(
    path: Path,
    original: np.ndarray,
    blurred_noisy: np.ndarray,
    best: Candidate,
    selection: str,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(10, 4))
    selected_label = "Closest Pareto to original" if selection == "reference" else "Best Pareto compromise"
    panels = [
        ("Original", original),
        ("Blurred + noise", blurred_noisy),
        (f"{selected_label} #{best.index}\nlambda={best.lambda_reg:.1e}", best.image),
    ]

    for axis, (panel_title, image) in zip(axes, panels):
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(panel_title)
        axis.axis("off")

    figure.suptitle("Selected two-objective Pareto reconstruction")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_metric_tradeoff_plot(path: Path, candidates: list[Candidate]) -> None:
    pareto = [candidate for candidate in candidates if candidate.is_pareto]
    if not pareto:
        return

    indexes = [candidate.index for candidate in pareto]
    psnr = [candidate.reference_psnr for candidate in pareto]
    ssim = [candidate.reference_ssim for candidate in pareto]
    residual = [candidate.residual_norm for candidate in pareto]

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].plot(indexes, psnr, marker="o", color="seagreen")
    axes[0].set_title("PSNR")
    axes[0].set_ylabel("dB")

    axes[1].plot(indexes, ssim, marker="o", color="royalblue")
    axes[1].set_title("SSIM")
    axes[1].set_ylim(0.0, 1.0)

    axes[2].plot(indexes, residual, marker="o", color="darkorange")
    axes[2].set_title("Residual norm")
    axes[2].set_ylabel("||Ax - y||")

    for axis in axes:
        axis.set_xlabel("Pareto candidate index")
        axis.grid(True, alpha=0.3)

    figure.suptitle("Quantitative metrics along the Pareto front")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_lambda_tradeoff_plot(path: Path, candidates: list[Candidate]) -> None:
    lambdas = np.array([candidate.lambda_reg for candidate in candidates])
    fidelity = np.array([candidate.fidelity_error for candidate in candidates])
    regularization = np.array([candidate.noise_penalty for candidate in candidates])

    order = np.argsort(lambdas)
    lambdas = lambdas[order]
    fidelity = fidelity[order]
    regularization = regularization[order]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].semilogx(lambdas, fidelity, marker="o", color="royalblue", label="J1")
    axes[0].semilogx(
        lambdas,
        regularization,
        marker="s",
        color="darkorange",
        label="J2",
    )
    axes[0].set_xlabel("lambda")
    axes[0].set_ylabel("objective value")
    axes[0].set_title("Objectives for different lambda values")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    scatter = axes[1].scatter(
        fidelity,
        regularization,
        c=np.log10(lambdas),
        cmap="viridis",
        s=55,
    )
    axes[1].plot(fidelity, regularization, color="0.5", linewidth=1.0, alpha=0.6)
    axes[1].set_xlabel("fidelity J1 = ||Ax - y||^2")
    axes[1].set_ylabel("regularization J2 = ||Lx||^2")
    axes[1].set_title("Fidelity-regularization trade-off")
    axes[1].grid(True, alpha=0.3)
    colorbar = figure.colorbar(scatter, ax=axes[1])
    colorbar.set_label("log10(lambda)")

    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_svd_explanation_figure(path: Path) -> None:
    sigma = np.logspace(0, -4, 240)
    inverse_filter = 1.0 / sigma
    lambda_reg = 1e-2
    tikhonov_filter = sigma / (sigma**2 + lambda_reg)

    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.loglog(sigma, inverse_filter, label="Unregularized inverse: 1/sigma")
    axis.loglog(
        sigma,
        tikhonov_filter,
        label="Tikhonov filter: sigma / (sigma^2 + lambda)",
    )
    axis.invert_xaxis()
    axis.set_xlabel("Singular value sigma")
    axis.set_ylabel("Noise amplification factor")
    axis.set_title("SVD view: regularization suppresses unstable small singular values")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def resolve_image_path(image_argument: str, input_dir: Path) -> Path:
    candidate = Path(image_argument)
    if candidate.exists():
        return candidate

    direct_match = input_dir / image_argument
    if direct_match.exists():
        return direct_match

    for extension in IMAGE_EXTENSIONS:
        stem_match = input_dir / f"{image_argument}{extension}"
        if stem_match.exists():
            return stem_match

    available = ", ".join(path.name for path in available_image_paths(input_dir))
    raise FileNotFoundError(
        f"Could not find image '{image_argument}'. Available images: {available}"
    )


def available_image_paths(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def run_image_experiment(
    image_path: Path,
    output_dir: Path,
    image_size: int,
    sigma_blur: float,
    noise_level: float,
    lambdas: np.ndarray,
    mols_method: str,
    epsilon_count: int,
    epsilon_refine_steps: int,
    max_iter: int,
    best_selection: str,
    rng: np.random.Generator,
) -> None:
    print(f"Processing {image_path.name}")
    print(f"  Working grid: {image_size}x{image_size}")
    image_output_dir = output_dir
    matrix_output_dir = image_output_dir / "matrices"
    matrix_output_dir.mkdir(parents=True, exist_ok=True)

    original = load_grayscale_image(image_path, image_size)
    shape = original.shape
    A = blur_operator(shape, sigma_blur)

    blurred = matrixize(A(vectorize(original)), shape)
    blurred_noisy = np.clip(
        blurred + noise_level * rng.standard_normal(shape),
        0.0,
        1.0,
    )

    weighted_candidates = solve_weighted_path(
        lambdas=lambdas,
        Y=blurred_noisy,
        shape=shape,
        A=A,
        max_iter=max_iter,
    )
    if mols_method == "epsilon":
        candidates = solve_epsilon_path(
            anchor_candidates=weighted_candidates,
            Y=blurred_noisy,
            shape=shape,
            A=A,
            max_iter=max_iter,
            epsilon_count=epsilon_count,
            refine_steps=epsilon_refine_steps,
        )
    elif mols_method == "weighted":
        candidates = weighted_candidates
    else:
        raise ValueError(f"Unknown MOLS method: {mols_method}")

    mark_pareto_front(candidates)
    best_compromise = mark_selected_pareto_candidate(
        candidates,
        original,
        best_selection,
    )
    pareto_candidates = [candidate for candidate in candidates if candidate.is_pareto]

    plt.imsave(image_output_dir / "original.png", original, cmap="gray", vmin=0, vmax=1)
    write_matrix(matrix_output_dir / "original", original)
    plt.imsave(
        image_output_dir / "blurred_noisy.png",
        blurred_noisy,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    write_matrix(matrix_output_dir / "blurred", blurred)
    write_matrix(matrix_output_dir / "blurred_noisy", blurred_noisy)

    for candidate in candidates:
        if candidate.cg_info != 0:
            print(
                f"  warning: CG did not fully converge for "
                f"lambda={candidate.lambda_reg:.3e}; info={candidate.cg_info}"
            )
        plt.imsave(
            image_output_dir / f"reconstruction_{candidate.index:02d}.png",
            candidate.image,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        write_matrix(
            matrix_output_dir / f"reconstruction_{candidate.index:02d}",
            candidate.image,
        )

    for candidate in pareto_candidates:
        plt.imsave(
            image_output_dir / f"pareto_reconstruction_{candidate.index:02d}.png",
            candidate.image,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        write_matrix(
            matrix_output_dir / f"pareto_reconstruction_{candidate.index:02d}",
            candidate.image,
        )

    plt.imsave(
        image_output_dir / "best_pareto_compromise.png",
        best_compromise.image,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    write_matrix(matrix_output_dir / "best_pareto_compromise", best_compromise.image)

    write_metrics_csv(image_output_dir / "metrics.csv", candidates)
    write_metrics_csv(image_output_dir / "pareto_front.csv", pareto_candidates)
    write_experiment_summary_csv(image_output_dir / "experiment_summary.csv", candidates)
    save_pareto_plot(image_output_dir / "pareto.png", candidates, best_selection)
    save_lambda_tradeoff_plot(image_output_dir / "lambda_tradeoff.png", weighted_candidates)
    save_metric_tradeoff_plot(image_output_dir / "metric_tradeoffs.png", candidates)
    save_svd_explanation_figure(image_output_dir / "svd_regularization_explanation.png")
    save_reconstruction_grid(
        image_output_dir / "summary.png",
        original,
        blurred_noisy,
        candidates,
        title=f"{mols_method.capitalize()} MOLS deblurring candidates",
    )
    save_reconstruction_grid(
        image_output_dir / "pareto_summary.png",
        original,
        blurred_noisy,
        pareto_candidates,
        title="Non-dominated two-objective Pareto reconstructions",
    )
    save_best_compromise_grid(
        image_output_dir / "best_pareto_compromise_summary.png",
        original,
        blurred_noisy,
        best_compromise,
        best_selection,
    )

    print(f"  Pareto-optimal candidates found: {len(pareto_candidates)}")
    print(f"  MOLS method: {mols_method}")
    print(
        f"  Selected Pareto candidate: #{best_compromise.index} "
        f"(lambda={best_compromise.lambda_reg:.3e})"
    )
    if best_compromise.reference_mse is not None:
        print(
            f"  Reference MSE/PSNR/SSIM: {best_compromise.reference_mse:.6e} / "
            f"{best_compromise.reference_psnr:.2f} dB / "
            f"{best_compromise.reference_ssim:.4f}"
        )


def main() -> None:
    args = parse_args()
    image_paths = available_image_paths(args.input_dir)

    if not image_paths:
        raise FileNotFoundError(f"No .tif or .tiff files found in {args.input_dir}")

    if args.list_images:
        print("Available images:")
        for image_path in image_paths:
            print(f"  {image_path.name}")
        return

    if args.image is None:
        args.image = DEFAULT_IMAGE_NAME
        print(f"No image selected. Using default image: {args.image}")

    rng = np.random.default_rng(args.seed)
    lambdas = regularization_grid(args.lambda_min, args.lambda_max, args.lambda_count)
    image_path = resolve_image_path(args.image, args.input_dir)
    prepare_output_dir(args.output_dir, args.input_dir)
    print(
        f"Quality preset: {args.quality_preset} "
        f"(lambda=[{args.lambda_min:.1e}, {args.lambda_max:.1e}], "
        f"count={args.lambda_count}, max_iter={args.max_iter})"
    )
    if args.mols_method == "epsilon":
        print(
            f"Epsilon-constraint settings: count={args.epsilon_count}, "
            f"refine_steps={args.epsilon_refine_steps}"
        )

    run_image_experiment(
        image_path=image_path,
        output_dir=args.output_dir,
        image_size=args.image_size,
        sigma_blur=args.sigma_blur,
        noise_level=args.noise_level,
        lambdas=lambdas,
        mols_method=args.mols_method,
        epsilon_count=args.epsilon_count,
        epsilon_refine_steps=args.epsilon_refine_steps,
        max_iter=args.max_iter,
        best_selection=args.best_selection,
        rng=rng,
    )

    print(f"Done. Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
