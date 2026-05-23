from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.sparse.linalg import LinearOperator, cg
from skimage import color, img_as_float, io
from skimage.transform import resize


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "docs" / "Original"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
DEFAULT_IMAGE_NAME = "couple.tif"


@dataclass
class Candidate:
    index: int
    lambda_reg: float
    image: np.ndarray
    fidelity_error: float
    noise_penalty: float
    objective_value: float
    cg_info: int
    is_pareto: bool = False
    compromise_score: float | None = None
    is_best_compromise: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Two-objective image deblurring as a Multi-Objective Least Squares "
            "problem: minimize fidelity ||Ax-y||^2 and noise/roughness ||Lx||^2."
        )
    )
    parser.add_argument(
        "image",
        nargs="?",
        help=(
            "Image filename or path to process, for example baboon.tif. "
            f"If omitted, the script uses {DEFAULT_IMAGE_NAME}."
        ),
    )
    parser.add_argument("--list-images", action="store_true")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--sigma-blur", type=float, default=2.0)
    parser.add_argument("--noise-level", type=float, default=0.01)
    parser.add_argument("--lambda-min", type=float, default=1e-6)
    parser.add_argument("--lambda-max", type=float, default=1e-1)
    parser.add_argument("--lambda-count", type=int, default=24)
    parser.add_argument("--max-iter", type=int, default=160)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def load_grayscale_image(path: Path, image_size: int) -> np.ndarray:
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


def blur_operator(shape: tuple[int, int], sigma: float):
    def blur(x: np.ndarray) -> np.ndarray:
        return ndimage.gaussian_filter(
            x.reshape(shape),
            sigma=sigma,
            mode="reflect",
        ).ravel()

    return blur


def laplacian_operator(x: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    image = x.reshape(shape)
    laplacian = (
        -4.0 * image
        + np.roll(image, 1, axis=0)
        + np.roll(image, -1, axis=0)
        + np.roll(image, 1, axis=1)
        + np.roll(image, -1, axis=1)
    )
    return laplacian.ravel()


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
    y: np.ndarray,
    shape: tuple[int, int],
    blur,
    max_iter: int,
) -> Candidate:
    y_flat = y.ravel()
    n = y_flat.size

    # Weighted MOLS scalarization:
    # min_x J_1(x) + lambda J_2(x)
    #   J_1(x) = ||Ax - y||^2,     data fidelity
    #   J_2(x) = ||Lx||^2,         Tikhonov/Laplacian regularization
    # Normal equations:
    #   (A^T A + lambda L^T L)x = A^T y.
    # Gaussian blur and this Laplacian stencil are self-adjoint under the
    # boundary model used here, so A^T and L^T are applied by the same routines.
    def matvec(x: np.ndarray) -> np.ndarray:
        ata_x = blur(blur(x))
        ltl_x = laplacian_operator(laplacian_operator(x, shape), shape)
        return ata_x + lambda_reg * ltl_x

    normal_matrix = LinearOperator((n, n), matvec=matvec)
    rhs = blur(y_flat)
    x_hat, info = cg(
        normal_matrix,
        rhs,
        x0=y_flat.copy(),
        maxiter=max_iter,
        rtol=1e-5,
    )

    image = np.clip(x_hat.reshape(shape), 0.0, 1.0)
    residual = blur(image.ravel()) - y_flat
    roughness = laplacian_operator(image.ravel(), shape)
    fidelity_error = float(np.linalg.norm(residual) ** 2)
    noise_penalty = float(np.linalg.norm(roughness) ** 2)

    return Candidate(
        index=index,
        lambda_reg=float(lambda_reg),
        image=image,
        fidelity_error=fidelity_error,
        noise_penalty=noise_penalty,
        objective_value=fidelity_error + float(lambda_reg) * noise_penalty,
        cg_info=info,
    )


def mark_pareto_front(candidates: list[Candidate]) -> None:
    objectives = np.array(
        [[candidate.fidelity_error, candidate.noise_penalty] for candidate in candidates],
        dtype=np.float64,
    )
    tolerance = 1e-12

    for i, candidate in enumerate(candidates):
        no_worse = np.all(objectives <= objectives[i] + tolerance, axis=1)
        strictly_better = np.any(objectives < objectives[i] - tolerance, axis=1)
        candidate.is_pareto = not bool(np.any(no_worse & strictly_better))


def mark_best_compromise(candidates: list[Candidate]) -> Candidate:
    pareto = [candidate for candidate in candidates if candidate.is_pareto]
    if not pareto:
        raise ValueError("No Pareto candidates were found.")

    objectives = np.array(
        [[candidate.fidelity_error, candidate.noise_penalty] for candidate in pareto],
        dtype=np.float64,
    )
    ideal = objectives.min(axis=0)
    nadir = objectives.max(axis=0)
    span = np.where(nadir > ideal, nadir - ideal, 1.0)
    normalized = (objectives - ideal) / span
    scores = np.linalg.norm(normalized, axis=1)

    best_index = int(np.argmin(scores))
    for candidate, score in zip(pareto, scores):
        candidate.compromise_score = float(score)
    pareto[best_index].is_best_compromise = True
    return pareto[best_index]


def write_metrics_csv(path: Path, candidates: list[Candidate]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "lambda_reg",
                "is_pareto",
                "is_best_compromise",
                "compromise_score",
                "fidelity_error",
                "noise_penalty",
                "weighted_objective",
                "cg_info",
            ]
        )
        for candidate in candidates:
            writer.writerow(
                [
                    candidate.index,
                    candidate.lambda_reg,
                    candidate.is_pareto,
                    candidate.is_best_compromise,
                    candidate.compromise_score,
                    candidate.fidelity_error,
                    candidate.noise_penalty,
                    candidate.objective_value,
                    candidate.cg_info,
                ]
            )


def save_pareto_plot(path: Path, candidates: list[Candidate]) -> None:
    dominated = [candidate for candidate in candidates if not candidate.is_pareto]
    pareto = [candidate for candidate in candidates if candidate.is_pareto]
    best = [candidate for candidate in candidates if candidate.is_best_compromise]

    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    if dominated:
        axis.scatter(
            [candidate.fidelity_error for candidate in dominated],
            [candidate.noise_penalty for candidate in dominated],
            c="0.7",
            label="Dominated weighted solution",
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
        axis.scatter(
            [candidate.fidelity_error for candidate in best],
            [candidate.noise_penalty for candidate in best],
            c="gold",
            edgecolors="black",
            marker="D",
            s=80,
            label="Best normalized compromise",
        )

    for candidate in pareto:
        axis.annotate(
            f"#{candidate.index}",
            (candidate.fidelity_error, candidate.noise_penalty),
            fontsize=7,
        )

    axis.set_xlabel("Fidelity error J1 = ||Ax - y||^2")
    axis.set_ylabel("Noise control J2 = ||Lx||^2")
    axis.set_title("Pareto front from weighted MOLS/Tikhonov solutions")
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
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(10, 4))
    panels = [
        ("Original", original),
        ("Blurred + noise", blurred_noisy),
        (f"Best Pareto compromise #{best.index}\nlambda={best.lambda_reg:.1e}", best.image),
    ]

    for axis, (panel_title, image) in zip(axes, panels):
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(panel_title)
        axis.axis("off")

    figure.suptitle("Selected two-objective Pareto reconstruction")
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

    stem_match = input_dir / f"{image_argument}.tif"
    if stem_match.exists():
        return stem_match

    available = ", ".join(path.name for path in sorted(input_dir.glob("*.tif")))
    raise FileNotFoundError(
        f"Could not find image '{image_argument}'. Available images: {available}"
    )


def run_image_experiment(
    image_path: Path,
    output_dir: Path,
    image_size: int,
    sigma_blur: float,
    noise_level: float,
    lambdas: np.ndarray,
    max_iter: int,
    rng: np.random.Generator,
) -> None:
    print(f"Processing {image_path.name}")
    image_output_dir = output_dir / image_path.stem
    image_output_dir.mkdir(parents=True, exist_ok=True)

    original = load_grayscale_image(image_path, image_size)
    shape = original.shape
    blur = blur_operator(shape, sigma_blur)

    blurred = blur(original.ravel()).reshape(shape)
    blurred_noisy = np.clip(
        blurred + noise_level * rng.standard_normal(shape),
        0.0,
        1.0,
    )

    candidates = [
        solve_tikhonov_candidate(
            index=index,
            lambda_reg=lambda_reg,
            y=blurred_noisy,
            shape=shape,
            blur=blur,
            max_iter=max_iter,
        )
        for index, lambda_reg in enumerate(lambdas)
    ]
    mark_pareto_front(candidates)
    best_compromise = mark_best_compromise(candidates)
    pareto_candidates = [candidate for candidate in candidates if candidate.is_pareto]

    plt.imsave(image_output_dir / "original.png", original, cmap="gray", vmin=0, vmax=1)
    plt.imsave(
        image_output_dir / "blurred_noisy.png",
        blurred_noisy,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

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

    for candidate in pareto_candidates:
        plt.imsave(
            image_output_dir / f"pareto_reconstruction_{candidate.index:02d}.png",
            candidate.image,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

    plt.imsave(
        image_output_dir / "best_pareto_compromise.png",
        best_compromise.image,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )

    write_metrics_csv(image_output_dir / "metrics.csv", candidates)
    write_metrics_csv(image_output_dir / "pareto_front.csv", pareto_candidates)
    save_pareto_plot(image_output_dir / "pareto.png", candidates)
    save_reconstruction_grid(
        image_output_dir / "summary.png",
        original,
        blurred_noisy,
        candidates,
        title="Weighted MOLS deblurring candidates",
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
    )

    print(f"  Pareto-optimal candidates found: {len(pareto_candidates)}")
    print(
        f"  Best compromise candidate: #{best_compromise.index} "
        f"(lambda={best_compromise.lambda_reg:.3e})"
    )


def main() -> None:
    args = parse_args()
    image_paths = sorted(args.input_dir.glob("*.tif"))

    if not image_paths:
        raise FileNotFoundError(f"No .tif files found in {args.input_dir}")

    if args.list_images:
        print("Available images:")
        for image_path in image_paths:
            print(f"  {image_path.name}")
        return

    if args.image is None:
        args.image = DEFAULT_IMAGE_NAME
        print(f"No image selected. Using default image: {args.image}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    lambdas = regularization_grid(args.lambda_min, args.lambda_max, args.lambda_count)
    image_path = resolve_image_path(args.image, args.input_dir)

    run_image_experiment(
        image_path=image_path,
        output_dir=args.output_dir,
        image_size=args.image_size,
        sigma_blur=args.sigma_blur,
        noise_level=args.noise_level,
        lambdas=lambdas,
        max_iter=args.max_iter,
        rng=rng,
    )

    print(f"Done. Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
