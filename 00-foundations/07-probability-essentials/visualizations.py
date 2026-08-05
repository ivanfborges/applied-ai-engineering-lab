"""Reusable probability calculations and Plotly visualizations.

All examples use deterministic synthetic data. Probability calculations are
kept separate from presentation helpers so they can be tested independently.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


EVENT_CATEGORY_ORDER = ("Neither", "A only", "B only", "A ∩ B")
EVENT_SYMBOLS = {
    "Neither": "circle-open",
    "A only": "square",
    "B only": "diamond",
    "A ∩ B": "star",
}
EVENT_COLORS = {
    "Neither": "#94A3B8",
    "A only": "#2563EB",
    "B only": "#F59E0B",
    "A ∩ B": "#7C3AED",
}

FRAUD_CATEGORY_ORDER = (
    "True positive",
    "False positive",
    "False negative",
    "True negative",
)
FRAUD_SYMBOLS = {
    "True positive": "star",
    "False positive": "x",
    "False negative": "diamond-open",
    "True negative": "circle-open",
}
FRAUD_COLORS = {
    "True positive": "#15803D",
    "False positive": "#DC2626",
    "False negative": "#D97706",
    "True negative": "#64748B",
}


def validate_probability(value: float, name: str) -> None:
    """Raise ``ValueError`` unless ``value`` is a finite probability."""
    numeric_value = float(value)
    if not np.isfinite(numeric_value) or not 0.0 <= numeric_value <= 1.0:
        raise ValueError(f"{name} must be a finite number between 0 and 1.")


def validate_discrete_distribution(
    values: np.ndarray | Sequence[float],
    probabilities: np.ndarray | Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return one-dimensional values and probabilities."""
    values_array = np.asarray(values, dtype=float)
    probabilities_array = np.asarray(probabilities, dtype=float)

    if values_array.ndim != 1 or probabilities_array.ndim != 1:
        raise ValueError("values and probabilities must be one-dimensional.")
    if values_array.size == 0:
        raise ValueError("values and probabilities cannot be empty.")
    if values_array.shape != probabilities_array.shape:
        raise ValueError("values and probabilities must have the same shape.")
    if not np.all(np.isfinite(values_array)):
        raise ValueError("values must contain only finite numbers.")
    if not np.all(np.isfinite(probabilities_array)):
        raise ValueError("probabilities must contain only finite numbers.")
    if np.any(probabilities_array < 0.0):
        raise ValueError("probabilities cannot be negative.")
    if not np.isclose(probabilities_array.sum(), 1.0, atol=1e-10):
        raise ValueError("probabilities must sum to 1.")

    return values_array, probabilities_array


def bayes_posterior(
    prior: float,
    true_positive_rate: float,
    false_positive_rate: float,
) -> float:
    """Return ``P(Fraud | Alert)`` for a binary detector."""
    validate_probability(prior, "prior")
    validate_probability(true_positive_rate, "true_positive_rate")
    validate_probability(false_positive_rate, "false_positive_rate")

    numerator = true_positive_rate * prior
    denominator = numerator + false_positive_rate * (1.0 - prior)
    if denominator <= np.finfo(float).eps:
        raise ValueError("P(Alert) must be greater than zero.")
    return numerator / denominator


def bayes_surface_values(
    priors: np.ndarray,
    false_positive_rates: np.ndarray,
    true_positive_rate: float,
) -> np.ndarray:
    """Vectorized, numerically safe Bayes posterior for a surface grid."""
    validate_probability(true_positive_rate, "true_positive_rate")
    prior_array = np.asarray(priors, dtype=float)
    fpr_array = np.asarray(false_positive_rates, dtype=float)
    if np.any((prior_array < 0.0) | (prior_array > 1.0)):
        raise ValueError("priors must be between 0 and 1.")
    if np.any((fpr_array < 0.0) | (fpr_array > 1.0)):
        raise ValueError("false_positive_rates must be between 0 and 1.")

    numerator = true_positive_rate * prior_array
    denominator = numerator + fpr_array * (1.0 - prior_array)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(denominator, np.nan, dtype=float),
        where=denominator > np.finfo(float).eps,
    )


def frechet_bounds(
    probability_a: float,
    probability_b: float,
) -> tuple[float, float]:
    """Return valid lower and upper bounds for ``P(A ∩ B)``."""
    validate_probability(probability_a, "probability_a")
    validate_probability(probability_b, "probability_b")
    lower = max(0.0, probability_a + probability_b - 1.0)
    upper = min(probability_a, probability_b)
    return lower, upper


def expected_value(
    values: np.ndarray | Sequence[float],
    probabilities: np.ndarray | Sequence[float],
) -> float:
    """Return the expected value of a finite discrete distribution."""
    values_array, probabilities_array = validate_discrete_distribution(
        values,
        probabilities,
    )
    return float(np.sum(values_array * probabilities_array))


def discrete_variance(
    values: np.ndarray | Sequence[float],
    probabilities: np.ndarray | Sequence[float],
) -> float:
    """Return population variance for a finite discrete distribution."""
    values_array, probabilities_array = validate_discrete_distribution(
        values,
        probabilities,
    )
    mean = float(np.sum(values_array * probabilities_array))
    return float(np.sum((values_array - mean) ** 2 * probabilities_array))


def bernoulli_variance(
    probability: np.ndarray | float,
) -> np.ndarray | float:
    """Return ``p(1-p)`` after validating scalar or array probabilities."""
    probability_array = np.asarray(probability, dtype=float)
    if not np.all(np.isfinite(probability_array)):
        raise ValueError("probability must contain only finite values.")
    if np.any((probability_array < 0.0) | (probability_array > 1.0)):
        raise ValueError("probability must be between 0 and 1.")
    result = probability_array * (1.0 - probability_array)
    if probability_array.ndim == 0:
        return float(result)
    return result


def expected_decision_threshold(
    review_cost: float,
    missed_event_cost: float,
) -> float:
    """Return the posterior threshold where review and non-review costs meet."""
    if not np.isfinite(review_cost) or review_cost < 0.0:
        raise ValueError("review_cost must be finite and non-negative.")
    if not np.isfinite(missed_event_cost) or missed_event_cost <= 0.0:
        raise ValueError("missed_event_cost must be finite and positive.")
    return float(review_cost / missed_event_cost)


def joint_probability_table(
    probability_a: float,
    probability_b: float,
    intersection_probability: float,
) -> pd.DataFrame:
    """Return a 2×2 table for a valid binary joint distribution."""
    lower, upper = frechet_bounds(probability_a, probability_b)
    if not lower - 1e-12 <= intersection_probability <= upper + 1e-12:
        raise ValueError(
            "intersection_probability must lie within the Fréchet bounds "
            f"[{lower:.6f}, {upper:.6f}]."
        )

    p11 = float(intersection_probability)
    p10 = probability_a - p11
    p01 = probability_b - p11
    p00 = 1.0 - p11 - p10 - p01
    values = np.clip(np.array([[p00, p01], [p10, p11]]), 0.0, 1.0)
    return pd.DataFrame(
        values,
        index=["A = False", "A = True"],
        columns=["B = False", "B = True"],
    )


def simulate_binary_joint_distribution(
    probability_a: float,
    probability_b: float,
    intersection_probability: float,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    """Simulate Boolean events from a specified 2×2 joint distribution."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    table = joint_probability_table(
        probability_a,
        probability_b,
        intersection_probability,
    )
    probabilities = table.to_numpy().ravel()
    rng = np.random.default_rng(seed)
    cells = rng.choice(4, size=sample_size, p=probabilities)
    return pd.DataFrame(
        {
            "event_a": cells >= 2,
            "event_b": (cells % 2) == 1,
        }
    )


def allocate_integer_counts(
    probabilities: Sequence[float],
    population_size: int,
) -> np.ndarray:
    """Allocate integer counts that sum exactly to ``population_size``."""
    if population_size <= 0:
        raise ValueError("population_size must be positive.")
    probability_array = np.asarray(probabilities, dtype=float)
    if probability_array.ndim != 1 or probability_array.size == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional array.")
    if np.any(probability_array < 0.0) or not np.isclose(
        probability_array.sum(),
        1.0,
        atol=1e-10,
    ):
        raise ValueError("probabilities must be non-negative and sum to 1.")

    exact_counts = probability_array * population_size
    counts = np.floor(exact_counts).astype(int)
    remainder = population_size - int(counts.sum())
    if remainder:
        order = np.argsort(-(exact_counts - counts))
        counts[order[:remainder]] += 1
    return counts


def fraud_outcome_probabilities(
    prior: float,
    true_positive_rate: float,
    false_positive_rate: float,
) -> dict[str, float]:
    """Return probabilities for the four fraud-detector outcomes."""
    validate_probability(prior, "prior")
    validate_probability(true_positive_rate, "true_positive_rate")
    validate_probability(false_positive_rate, "false_positive_rate")
    return {
        "True positive": prior * true_positive_rate,
        "False positive": (1.0 - prior) * false_positive_rate,
        "False negative": prior * (1.0 - true_positive_rate),
        "True negative": (1.0 - prior) * (1.0 - false_positive_rate),
    }


def fraud_outcome_counts(
    prior: float,
    true_positive_rate: float,
    false_positive_rate: float,
    population_size: int,
) -> dict[str, int]:
    """Return deterministic integer counts for a synthetic population."""
    probabilities = fraud_outcome_probabilities(
        prior,
        true_positive_rate,
        false_positive_rate,
    )
    counts = allocate_integer_counts(list(probabilities.values()), population_size)
    return dict(zip(probabilities, counts, strict=True))


def simulate_fraud_population(
    population_size: int,
    prior: float,
    true_positive_rate: float,
    false_positive_rate: float,
    seed: int,
) -> pd.DataFrame:
    """Simulate fraud truth, alerts, and labeled detector outcomes."""
    if population_size <= 0:
        raise ValueError("population_size must be positive.")
    validate_probability(prior, "prior")
    validate_probability(true_positive_rate, "true_positive_rate")
    validate_probability(false_positive_rate, "false_positive_rate")

    rng = np.random.default_rng(seed)
    is_fraud = rng.random(population_size) < prior
    alert_probability = np.where(
        is_fraud,
        true_positive_rate,
        false_positive_rate,
    )
    has_alert = rng.random(population_size) < alert_probability
    outcome = np.select(
        [
            is_fraud & has_alert,
            ~is_fraud & has_alert,
            is_fraud & ~has_alert,
        ],
        ["True positive", "False positive", "False negative"],
        default="True negative",
    )
    return pd.DataFrame(
        {
            "is_fraud": is_fraud,
            "has_alert": has_alert,
            "outcome": outcome,
        }
    )


def event_sample_space(population_size: int = 100) -> pd.DataFrame:
    """Return integer outcomes with A=divisible by 2 and B=divisible by 5."""
    if population_size <= 0:
        raise ValueError("population_size must be positive.")
    outcome = np.arange(1, population_size + 1)
    event_a = outcome % 2 == 0
    event_b = outcome % 5 == 0
    category = np.select(
        [
            event_a & event_b,
            event_a,
            event_b,
        ],
        ["A ∩ B", "A only", "B only"],
        default="Neither",
    )
    columns = int(np.ceil(np.sqrt(population_size)))
    return pd.DataFrame(
        {
            "outcome": outcome,
            "event_a": event_a,
            "event_b": event_b,
            "category": category,
            "x": (outcome - 1) % columns + 1,
            "y": columns - (outcome - 1) // columns,
        }
    )


def event_operation_mask(data: pd.DataFrame, operation: str) -> pd.Series:
    """Return the selected event mask for a supported operation."""
    operations = {
        "A": data["event_a"],
        "B": data["event_b"],
        "A ∪ B": data["event_a"] | data["event_b"],
        "A ∩ B": data["event_a"] & data["event_b"],
        "Aᶜ": ~data["event_a"],
        "Bᶜ": ~data["event_b"],
    }
    if operation not in operations:
        raise ValueError(f"Unsupported event operation: {operation}.")
    return operations[operation]


def conditional_population(
    total_population: int,
    count_a: int,
    count_b: int,
    count_intersection: int,
) -> pd.DataFrame:
    """Construct a deterministic labeled population from valid event counts."""
    if total_population <= 0:
        raise ValueError("total_population must be positive.")
    if not all(
        isinstance(value, (int, np.integer))
        for value in (count_a, count_b, count_intersection)
    ):
        raise ValueError("event counts must be integers.")
    if not 0 <= count_a <= total_population:
        raise ValueError("count_a must be between 0 and total_population.")
    if not 0 <= count_b <= total_population:
        raise ValueError("count_b must be between 0 and total_population.")

    lower = max(0, count_a + count_b - total_population)
    upper = min(count_a, count_b)
    if not lower <= count_intersection <= upper:
        raise ValueError(
            "count_intersection must satisfy the count Fréchet bounds "
            f"[{lower}, {upper}]."
        )

    counts = {
        "A ∩ B": count_intersection,
        "A only": count_a - count_intersection,
        "B only": count_b - count_intersection,
        "Neither": (
            total_population - count_a - count_b + count_intersection
        ),
    }
    category = np.concatenate(
        [np.repeat(name, count) for name, count in counts.items()]
    )
    outcome = np.arange(1, total_population + 1)
    columns = int(np.ceil(np.sqrt(total_population)))
    return pd.DataFrame(
        {
            "outcome": outcome,
            "category": category,
            "event_a": np.isin(category, ["A only", "A ∩ B"]),
            "event_b": np.isin(category, ["B only", "A ∩ B"]),
            "x": (outcome - 1) % columns + 1,
            "y": columns - (outcome - 1) // columns,
        }
    )


def die_monte_carlo(max_sample_size: int, seed: int) -> pd.DataFrame:
    """Return cumulative fair-die mean, variance, and estimation errors."""
    if max_sample_size <= 0:
        raise ValueError("max_sample_size must be positive.")
    rng = np.random.default_rng(seed)
    rolls = rng.integers(1, 7, size=max_sample_size)
    sample_size = np.arange(1, max_sample_size + 1)
    cumulative_mean = np.cumsum(rolls) / sample_size
    cumulative_second_moment = np.cumsum(rolls**2) / sample_size
    cumulative_variance = cumulative_second_moment - cumulative_mean**2
    return pd.DataFrame(
        {
            "sample_size": sample_size,
            "estimate": cumulative_mean,
            "variance_estimate": cumulative_variance,
            "absolute_error": np.abs(cumulative_mean - 3.5),
        }
    )


def fraud_posterior_monte_carlo(
    max_sample_size: int,
    seed: int,
    prior: float = 0.01,
    true_positive_rate: float = 0.90,
    false_positive_rate: float = 0.05,
) -> pd.DataFrame:
    """Return cumulative empirical ``P(Fraud | Alert)`` and its error."""
    data = simulate_fraud_population(
        max_sample_size,
        prior,
        true_positive_rate,
        false_positive_rate,
        seed,
    )
    sample_size = np.arange(1, max_sample_size + 1)
    alert_count = np.cumsum(data["has_alert"].to_numpy(dtype=int))
    true_positive_count = np.cumsum(
        (data["is_fraud"] & data["has_alert"]).to_numpy(dtype=int)
    )
    estimate = np.divide(
        true_positive_count,
        alert_count,
        out=np.full(max_sample_size, np.nan, dtype=float),
        where=alert_count > 0,
    )
    theoretical = bayes_posterior(
        prior,
        true_positive_rate,
        false_positive_rate,
    )
    return pd.DataFrame(
        {
            "sample_size": sample_size,
            "estimate": estimate,
            "absolute_error": np.abs(estimate - theoretical),
            "alert_count": alert_count,
        }
    )


def make_event_grid_figure(
    operation: str = "A ∪ B",
    population_size: int = 100,
) -> go.Figure:
    """Create an interactive sample-space grid for an event operation."""
    data = event_sample_space(population_size)
    selected = event_operation_mask(data, operation)
    figure = go.Figure()

    for category in EVENT_CATEGORY_ORDER:
        subset = data.loc[data["category"] == category].copy()
        selected_subset = selected.loc[subset.index].to_numpy()
        figure.add_trace(
            go.Scatter(
                x=subset["x"],
                y=subset["y"],
                mode="markers+text",
                name=category,
                text=subset["outcome"],
                textposition="middle center",
                textfont={"size": 8, "color": "#111827"},
                hovertemplate=(
                    "Outcome %{text}<br>"
                    f"Category: {category}<br>"
                    f"Selected by {operation}: "
                    "%{customdata}<extra></extra>"
                ),
                customdata=np.where(selected_subset, "yes", "no"),
                marker={
                    "symbol": EVENT_SYMBOLS[category],
                    "color": EVENT_COLORS[category],
                    "size": np.where(selected_subset, 19, 12),
                    "opacity": np.where(selected_subset, 1.0, 0.20),
                    "line": {"color": "#111827", "width": 1},
                },
            )
        )

    probability = float(selected.mean())
    figure.update_layout(
        title=(
            f"Sample space: selected event {operation} "
            f"contains {int(selected.sum())}/{population_size} outcomes "
            f"(P={probability:.2f})"
        ),
        xaxis={"title": "Grid column", "showticklabels": False},
        yaxis={
            "title": "Grid row",
            "showticklabels": False,
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        legend={"orientation": "h", "y": -0.12},
        margin={"l": 30, "r": 20, "t": 70, "b": 70},
        height=610,
    )
    return figure


def make_conditional_grid_figure(
    total_population: int,
    count_a: int,
    count_b: int,
    count_intersection: int,
) -> go.Figure:
    """Show event B as the denominator and A∩B as the numerator."""
    data = conditional_population(
        total_population,
        count_a,
        count_b,
        count_intersection,
    )
    figure = go.Figure()
    for category in EVENT_CATEGORY_ORDER:
        subset = data.loc[data["category"] == category]
        in_denominator = subset["event_b"].to_numpy()
        figure.add_trace(
            go.Scatter(
                x=subset["x"],
                y=subset["y"],
                mode="markers",
                name=category,
                marker={
                    "symbol": EVENT_SYMBOLS[category],
                    "color": EVENT_COLORS[category],
                    "size": np.where(in_denominator, 15, 10),
                    "opacity": np.where(in_denominator, 1.0, 0.12),
                    "line": {
                        "color": np.where(
                            subset["category"].eq("A ∩ B"),
                            "#111827",
                            "#64748B",
                        ),
                        "width": np.where(
                            subset["category"].eq("A ∩ B"),
                            2.5,
                            0.7,
                        ),
                    },
                },
                customdata=np.column_stack(
                    [subset["event_a"], subset["event_b"]]
                ),
                hovertemplate=(
                    f"{category}<br>A=%{{customdata[0]}}"
                    "<br>B=%{customdata[1]}<extra></extra>"
                ),
            )
        )

    conditional = (
        count_intersection / count_b if count_b > 0 else float("nan")
    )
    figure.update_layout(
        title=(
            "Condition on B: faded outcomes leave the denominator; "
            f"{count_intersection}/{count_b} remain in A∩B "
            f"(P(A|B)={conditional:.3f})"
        ),
        xaxis={"showticklabels": False, "title": "Grid column"},
        yaxis={
            "showticklabels": False,
            "title": "Grid row",
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        legend={"orientation": "h", "y": -0.12},
        margin={"l": 30, "r": 20, "t": 75, "b": 70},
        height=580,
    )
    return figure


def make_joint_heatmap(
    probability_a: float,
    probability_b: float,
    intersection_probability: float,
) -> go.Figure:
    """Create a labeled heatmap of a 2×2 joint distribution."""
    table = joint_probability_table(
        probability_a,
        probability_b,
        intersection_probability,
    )
    values = table.to_numpy()
    text = np.vectorize(lambda value: f"{value:.3f}<br>{value:.1%}")(values)
    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=table.columns,
            y=table.index,
            colorscale="Blues",
            zmin=0.0,
            zmax=max(0.5, float(values.max())),
            text=text,
            texttemplate="%{text}",
            hovertemplate="%{y}, %{x}<br>Probability=%{z:.4f}<extra></extra>",
            colorbar={"title": "Joint<br>probability"},
        )
    )
    difference = intersection_probability - probability_a * probability_b
    figure.update_layout(
        title=(
            "Binary joint distribution: "
            f"P(A∩B) − P(A)P(B) = {difference:+.4f}"
        ),
        xaxis_title="Event B",
        yaxis_title="Event A",
        height=430,
        margin={"l": 70, "r": 30, "t": 70, "b": 55},
    )
    return figure


def make_fraud_population_figure(
    prior: float,
    true_positive_rate: float,
    false_positive_rate: float,
    population_size: int,
    max_visual_cells: int = 400,
) -> go.Figure:
    """Create a representative grid of four labeled detector outcomes."""
    display_size = min(population_size, max_visual_cells)
    probabilities = fraud_outcome_probabilities(
        prior,
        true_positive_rate,
        false_positive_rate,
    )
    display_counts = allocate_integer_counts(
        list(probabilities.values()),
        display_size,
    )
    category = np.concatenate(
        [
            np.repeat(name, count)
            for name, count in zip(probabilities, display_counts, strict=True)
        ]
    )
    columns = int(np.ceil(np.sqrt(display_size)))
    index = np.arange(display_size)
    figure = go.Figure()

    for name in FRAUD_CATEGORY_ORDER:
        mask = category == name
        figure.add_trace(
            go.Scatter(
                x=index[mask] % columns + 1,
                y=columns - index[mask] // columns,
                mode="markers",
                name=name,
                marker={
                    "symbol": FRAUD_SYMBOLS[name],
                    "color": FRAUD_COLORS[name],
                    "size": 11,
                    "line": {"color": "#111827", "width": 0.7},
                },
                hovertemplate=(
                    f"{name}<br>Population probability: "
                    f"{probabilities[name]:.3%}<extra></extra>"
                ),
            )
        )

    counts = fraud_outcome_counts(
        prior,
        true_positive_rate,
        false_positive_rate,
        population_size,
    )
    figure.update_layout(
        title=(
            f"Representative outcome grid ({display_size} cells), scaled counts: "
            f"TP={counts['True positive']:,}, FP={counts['False positive']:,}, "
            f"FN={counts['False negative']:,}, TN={counts['True negative']:,}"
        ),
        xaxis={"showticklabels": False, "title": "Representative population"},
        yaxis={
            "showticklabels": False,
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        legend={"orientation": "h", "y": -0.13},
        margin={"l": 25, "r": 20, "t": 80, "b": 75},
        height=580,
    )
    return figure


def make_confusion_matrix_figure(
    prior: float,
    true_positive_rate: float,
    false_positive_rate: float,
    population_size: int,
) -> go.Figure:
    """Create a confusion matrix with deterministic counts and percentages."""
    counts = fraud_outcome_counts(
        prior,
        true_positive_rate,
        false_positive_rate,
        population_size,
    )
    values = np.array(
        [
            [counts["True negative"], counts["False positive"]],
            [counts["False negative"], counts["True positive"]],
        ]
    )
    percentages = values / population_size
    labels = np.array(
        [
            ["True negative", "False positive"],
            ["False negative", "True positive"],
        ]
    )
    text = np.empty_like(labels, dtype=object)
    for row in range(2):
        for column in range(2):
            text[row, column] = (
                f"<b>{labels[row, column]}</b><br>"
                f"{values[row, column]:,}<br>"
                f"{percentages[row, column]:.2%} of population"
            )

    figure = go.Figure(
        go.Heatmap(
            z=percentages,
            x=["No alert", "Alert"],
            y=["Legitimate", "Fraud"],
            colorscale="Blues",
            text=text,
            texttemplate="%{text}",
            hovertemplate="%{text}<extra></extra>",
            colorbar={"title": "Population<br>share"},
        )
    )
    figure.update_layout(
        title=f"Synthetic confusion matrix (population={population_size:,})",
        xaxis_title="Detector decision",
        yaxis_title="Actual class",
        height=430,
        margin={"l": 70, "r": 25, "t": 65, "b": 55},
    )
    return figure


def make_probability_flow_figure(
    prior: float,
    true_positive_rate: float,
    false_positive_rate: float,
    population_size: int,
) -> go.Figure:
    """Create a Sankey flow that exposes the alert denominator."""
    fraud_count = population_size * prior
    legitimate_count = population_size * (1.0 - prior)
    true_positive = fraud_count * true_positive_rate
    false_negative = fraud_count * (1.0 - true_positive_rate)
    false_positive = legitimate_count * false_positive_rate
    true_negative = legitimate_count * (1.0 - false_positive_rate)
    labels = [
        "Population",
        "Fraud",
        "Not fraud",
        "Alert | fraud (TP)",
        "No alert | fraud (FN)",
        "Alert | not fraud (FP)",
        "No alert | not fraud (TN)",
        "All alerts = TP + FP",
    ]
    figure = go.Figure(
        go.Sankey(
            arrangement="snap",
            node={
                "label": labels,
                "pad": 18,
                "thickness": 20,
                "line": {"color": "#111827", "width": 0.7},
                "color": [
                    "#CBD5E1",
                    "#F59E0B",
                    "#64748B",
                    "#15803D",
                    "#D97706",
                    "#DC2626",
                    "#64748B",
                    "#7C3AED",
                ],
            },
            link={
                "source": [0, 0, 1, 1, 2, 2, 3, 5],
                "target": [1, 2, 3, 4, 5, 6, 7, 7],
                "value": [
                    fraud_count,
                    legitimate_count,
                    true_positive,
                    false_negative,
                    false_positive,
                    true_negative,
                    true_positive,
                    false_positive,
                ],
                "label": [
                    f"{fraud_count:,.1f}",
                    f"{legitimate_count:,.1f}",
                    f"{true_positive:,.1f}",
                    f"{false_negative:,.1f}",
                    f"{false_positive:,.1f}",
                    f"{true_negative:,.1f}",
                    "True-positive contribution to alert denominator",
                    "False-positive contribution to alert denominator",
                ],
            },
        )
    )
    posterior = bayes_posterior(
        prior,
        true_positive_rate,
        false_positive_rate,
    )
    figure.update_layout(
        title=(
            "Probability flow: P(Fraud | Alert) = TP / (TP + FP) "
            f"= {posterior:.2%}"
        ),
        height=520,
        margin={"l": 15, "r": 15, "t": 70, "b": 20},
    )
    return figure


def make_bayes_surface_figure(
    true_positive_rate: float,
    selected_prior: float,
    selected_false_positive_rate: float,
) -> go.Figure:
    """Create an interactive 3D Bayes surface with a scenario marker."""
    priors = np.linspace(0.001, 0.25, 65)
    false_positive_rates = np.linspace(0.001, 0.30, 65)
    prior_grid, fpr_grid = np.meshgrid(priors, false_positive_rates)
    posterior_grid = bayes_surface_values(
        prior_grid,
        fpr_grid,
        true_positive_rate,
    )
    selected_posterior = bayes_posterior(
        selected_prior,
        true_positive_rate,
        selected_false_positive_rate,
    )

    figure = go.Figure()
    figure.add_trace(
        go.Surface(
            x=prior_grid,
            y=fpr_grid,
            z=posterior_grid,
            colorscale="Viridis",
            opacity=0.88,
            colorbar={"title": "Posterior"},
            hovertemplate=(
                "Prior=%{x:.3%}<br>False-positive rate=%{y:.3%}"
                "<br>Posterior=%{z:.3%}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[selected_prior],
            y=[selected_false_positive_rate],
            z=[selected_posterior],
            mode="markers+text",
            name="Selected scenario",
            text=[f"Current: {selected_posterior:.1%}"],
            textposition="top center",
            marker={
                "size": 8,
                "color": "#DC2626",
                "symbol": "diamond",
                "line": {"color": "#111827", "width": 2},
            },
            hovertemplate=(
                "Selected scenario<br>Prior=%{x:.3%}"
                "<br>False-positive rate=%{y:.3%}"
                "<br>Posterior=%{z:.3%}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=(
            "Bayes surface: posterior sensitivity to prevalence and "
            f"false positives (TPR={true_positive_rate:.1%})"
        ),
        scene={
            "xaxis_title": "Prior P(Fraud)",
            "yaxis_title": "False-positive rate P(Alert | not fraud)",
            "zaxis_title": "Posterior P(Fraud | Alert)",
            "zaxis": {"range": [0.0, 1.0]},
            "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 0.95}},
        },
        height=650,
        margin={"l": 0, "r": 0, "t": 75, "b": 0},
    )
    return figure


def make_expected_value_figure(
    values: np.ndarray | Sequence[float],
    probabilities: np.ndarray | Sequence[float],
    title: str,
) -> go.Figure:
    """Create a probability-mass chart with weighted contributions."""
    values_array, probabilities_array = validate_discrete_distribution(
        values,
        probabilities,
    )
    mean = expected_value(values_array, probabilities_array)
    contributions = values_array * probabilities_array
    figure = go.Figure(
        go.Bar(
            x=values_array,
            y=probabilities_array,
            name="Probability mass",
            marker={
                "color": "#2563EB",
                "line": {"color": "#111827", "width": 0.8},
            },
            text=[f"x·p={value:.3f}" for value in contributions],
            textposition="outside",
            customdata=contributions,
            hovertemplate=(
                "Value=%{x}<br>Probability=%{y:.4f}"
                "<br>Weighted contribution=%{customdata:.4f}<extra></extra>"
            ),
        )
    )
    figure.add_vline(
        x=mean,
        line_width=3,
        line_dash="dash",
        line_color="#DC2626",
        annotation_text=f"E[X] = {mean:.3f}",
        annotation_position="top",
    )
    figure.update_layout(
        title=title,
        xaxis_title="Outcome value x",
        yaxis_title="Probability P(X=x)",
        yaxis={"range": [0.0, max(0.35, float(probabilities_array.max()) * 1.3)]},
        height=480,
        margin={"l": 55, "r": 20, "t": 75, "b": 55},
    )
    return figure


def make_variance_distribution_figure(
    selected_standard_deviation: float,
) -> go.Figure:
    """Compare same-mean Gaussian densities with different spreads."""
    if selected_standard_deviation <= 0.0:
        raise ValueError("selected_standard_deviation must be positive.")
    x = np.linspace(-8.0, 8.0, 500)
    standard_deviations = [0.75, 1.75, selected_standard_deviation]
    names = ["Narrow (σ=0.75)", "Medium (σ=1.75)", f"Selected (σ={selected_standard_deviation:.2f})"]
    styles = ["solid", "dash", "dot"]
    symbols = ["circle", "square", "diamond"]
    figure = go.Figure()
    for sigma, name, dash, symbol in zip(
        standard_deviations,
        names,
        styles,
        symbols,
        strict=True,
    ):
        density = np.exp(-0.5 * (x / sigma) ** 2) / (
            sigma * np.sqrt(2.0 * np.pi)
        )
        figure.add_trace(
            go.Scatter(
                x=x,
                y=density,
                mode="lines",
                name=name,
                line={"width": 3, "dash": dash},
                marker={"symbol": symbol},
                hovertemplate=(
                    f"{name}<br>x=%{{x:.2f}}<br>density=%{{y:.4f}}"
                    "<extra></extra>"
                ),
            )
        )
    figure.add_vline(
        x=0.0,
        line_dash="dash",
        line_color="#111827",
        annotation_text="Shared mean μ=0",
    )
    figure.update_layout(
        title="Same mean, different variance: spread changes while center stays fixed",
        xaxis_title="Outcome",
        yaxis_title="Probability density",
        legend={"orientation": "h", "y": -0.18},
        height=480,
        margin={"l": 55, "r": 20, "t": 70, "b": 90},
    )
    return figure


def make_squared_deviation_figure(values: Sequence[float]) -> go.Figure:
    """Show distances from the mean and their squared contributions."""
    values_array = np.asarray(values, dtype=float)
    if values_array.ndim != 1 or values_array.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence.")
    if not np.all(np.isfinite(values_array)):
        raise ValueError("values must contain only finite numbers.")
    mean = float(values_array.mean())
    squared_deviations = (values_array - mean) ** 2
    index = np.arange(1, values_array.size + 1)
    figure = go.Figure()
    for point_index, value, squared in zip(
        index,
        values_array,
        squared_deviations,
        strict=True,
    ):
        figure.add_trace(
            go.Scatter(
                x=[point_index, point_index],
                y=[mean, value],
                mode="lines",
                line={"color": "#94A3B8", "dash": "dot"},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[point_index],
                y=[value],
                mode="markers+text",
                marker={
                    "size": 13,
                    "symbol": "diamond",
                    "color": "#2563EB",
                    "line": {"color": "#111827", "width": 1},
                },
                text=[f"{squared:.2f}"],
                textposition="top center",
                name="Observation" if point_index == 1 else None,
                showlegend=bool(point_index == 1),
                customdata=[[value - mean, squared]],
                hovertemplate=(
                    "Value=%{y:.2f}<br>Deviation=%{customdata[0]:.2f}"
                    "<br>Squared deviation=%{customdata[1]:.2f}"
                    "<extra></extra>"
                ),
            )
        )
    figure.add_hline(
        y=mean,
        line_width=3,
        line_dash="dash",
        line_color="#DC2626",
        annotation_text=f"Mean = {mean:.2f}",
    )
    figure.update_layout(
        title="Squared deviations: labels show each contribution before averaging",
        xaxis_title="Observation index",
        yaxis_title="Observed value",
        height=460,
        margin={"l": 55, "r": 20, "t": 70, "b": 55},
    )
    return figure


def make_monte_carlo_figure(
    data: pd.DataFrame,
    theoretical_value: float,
    estimate_label: str,
) -> go.Figure:
    """Create estimate and absolute-error panels for Monte Carlo results."""
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(estimate_label, "Absolute estimation error"),
    )
    figure.add_trace(
        go.Scatter(
            x=data["sample_size"],
            y=data["estimate"],
            mode="lines",
            name="Empirical estimate",
            line={"color": "#2563EB", "width": 2},
            hovertemplate="n=%{x:,}<br>estimate=%{y:.5f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_hline(
        y=theoretical_value,
        line_dash="dash",
        line_color="#DC2626",
        annotation_text=f"Theory = {theoretical_value:.5f}",
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=data["sample_size"],
            y=data["absolute_error"],
            mode="lines",
            name="Absolute error",
            line={"color": "#D97706", "width": 2},
            hovertemplate="n=%{x:,}<br>|error|=%{y:.5f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    figure.update_xaxes(title_text="Sample size", row=2, col=1)
    figure.update_yaxes(title_text="Estimate", row=1, col=1)
    figure.update_yaxes(title_text="Absolute error", row=2, col=1)
    figure.update_layout(
        title="Monte Carlo convergence: empirical frequency approaches theory",
        height=620,
        legend={"orientation": "h", "y": -0.12},
        margin={"l": 60, "r": 20, "t": 80, "b": 80},
    )
    return figure


def make_expected_cost_figure(
    review_cost: float,
    missed_event_cost: float,
    selected_posterior: float,
) -> go.Figure:
    """Plot review and non-review expected costs across posterior probability."""
    validate_probability(selected_posterior, "selected_posterior")
    threshold = expected_decision_threshold(review_cost, missed_event_cost)
    probabilities = np.linspace(0.0, 1.0, 300)
    no_review_cost = probabilities * missed_event_cost
    selected_no_review_cost = selected_posterior * missed_event_cost
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=probabilities,
            y=np.full_like(probabilities, review_cost),
            mode="lines",
            name="Review: fixed cost Cᵣ",
            line={"color": "#2563EB", "width": 3, "dash": "dash"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=probabilities,
            y=no_review_cost,
            mode="lines",
            name="Do not review: p × Cₘ",
            line={"color": "#DC2626", "width": 3},
        )
    )
    if threshold <= 1.0:
        figure.add_trace(
            go.Scatter(
                x=[threshold],
                y=[review_cost],
                mode="markers+text",
                name="Decision threshold",
                text=[f"p*={threshold:.3f}"],
                textposition="top center",
                marker={
                    "size": 13,
                    "symbol": "diamond",
                    "color": "#7C3AED",
                    "line": {"color": "#111827", "width": 1},
                },
            )
        )
    figure.add_trace(
        go.Scatter(
            x=[selected_posterior, selected_posterior],
            y=[review_cost, selected_no_review_cost],
            mode="markers",
            name="Selected scenario costs",
            marker={
                "size": 11,
                "symbol": "x",
                "color": "#111827",
            },
        )
    )
    figure.update_layout(
        title="Expected-cost decision rule: choose the lower cost at each posterior",
        xaxis_title="Posterior event probability p",
        yaxis_title="Expected cost",
        xaxis={"range": [0.0, 1.0]},
        legend={"orientation": "h", "y": -0.18},
        height=500,
        margin={"l": 60, "r": 20, "t": 70, "b": 90},
    )
    return figure


def make_reliability_comparison_figure(
    service_success_probability: float = 0.99,
    shared_region_failure_probability: float = 0.02,
) -> go.Figure:
    """Contrast an independence calculation with a shared-failure model."""
    validate_probability(
        service_success_probability,
        "service_success_probability",
    )
    validate_probability(
        shared_region_failure_probability,
        "shared_region_failure_probability",
    )
    independent_success = service_success_probability**2
    conditional_success = (
        (1.0 - shared_region_failure_probability)
        * service_success_probability**2
    )
    figure = go.Figure(
        go.Bar(
            x=["Assumed independent", "Shared region failure"],
            y=[independent_success, conditional_success],
            marker={
                "color": ["#2563EB", "#D97706"],
                "pattern": {"shape": ["", "/"]},
                "line": {"color": "#111827", "width": 1},
            },
            text=[f"{independent_success:.3%}", f"{conditional_success:.3%}"],
            textposition="outside",
            hovertemplate="%{x}<br>Joint success=%{y:.4%}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Synthetic two-service reliability: shared causes reduce joint success",
        xaxis_title="Dependency model",
        yaxis_title="Probability both services succeed",
        yaxis={"range": [max(0.0, conditional_success - 0.04), 1.0]},
        height=400,
        margin={"l": 60, "r": 20, "t": 70, "b": 55},
    )
    return figure


def make_llm_quality_comparison_figure(seed: int = 42) -> go.Figure:
    """Compare synthetic quality systems with similar means and different tails."""
    rng = np.random.default_rng(seed)
    stable = np.clip(rng.normal(0.78, 0.05, 600), 0.0, 1.0)
    variable = np.clip(rng.normal(0.80, 0.16, 600), 0.0, 1.0)
    # Shift the second system to align sample means while keeping its spread.
    variable = np.clip(variable + stable.mean() - variable.mean(), 0.0, 1.0)
    figure = go.Figure()
    for values, name, color, symbol in (
        (stable, "Stable quality", "#2563EB", "circle"),
        (variable, "Variable quality", "#D97706", "diamond"),
    ):
        figure.add_trace(
            go.Box(
                y=values,
                name=name,
                boxpoints="outliers",
                marker={"color": color, "symbol": symbol},
                line={"color": color},
                hovertemplate=f"{name}<br>quality=%{{y:.3f}}<extra></extra>",
            )
        )
    figure.update_layout(
        title="Synthetic LLM evaluation: similar mean, different variance and tails",
        xaxis_title="Evaluation system",
        yaxis_title="Synthetic quality score",
        yaxis={"range": [0.0, 1.0]},
        height=430,
        margin={"l": 60, "r": 20, "t": 70, "b": 55},
    )
    return figure
