import numpy as np
import matplotlib.pyplot as plt


def hessian_phi(x, D):
    gap = D**2 - np.dot(x, x)
    if gap <= 0:
        raise ValueError("The internal point is outside the barrier domain.")

    return 2.0 / gap * np.eye(x.size) + 4.0 / gap**2 * np.outer(x, x)


def update_x(grad_sum, eta, D, delta):
    """Solve the log-barrier FTRL update on the ball of radius (1-delta)D."""
    grad_norm = np.linalg.norm(grad_sum)
    scale = eta * D**2 / (1.0 + np.hypot(1.0, eta * D * grad_norm))
    x_next = -scale * grad_sum

    radius = (1.0 - delta) * D
    norm = np.linalg.norm(x_next)
    shrinkage_active = delta > 0 and norm > radius
    if shrinkage_active:
        x_next *= radius / norm

    return x_next, shrinkage_active


def sigma(y, epsilon, action_sum, D, boundary_sign):
    """Evaluate the history-dependent perturbation at the played point."""
    norm = np.linalg.norm(y)

    # if np.isclose(norm, D, rtol=0.0, atol=1e-10):
    #     return epsilon * boundary_sign

    # if np.dot(y, action_sum) >= 0:
    #     return epsilon * np.sin(1.0 / (D - norm))

    # return  boundary_sign * epsilon * np.sin(1000*norm)

    diff = D - norm  # not 0
    sinx = np.sin(1/(diff))
    
    return epsilon*sinx if np.dot(y, action_sum) >= 0 else epsilon

def sigma2(epsilon, x, A_inv_u):
    """Maximize the perturbation's gradient projection toward w_star."""
    w_star = np.zeros(2)
    w_star[0] = 0.625
    score = A_inv_u @ (w_star - x)
    return float(epsilon * np.sign(score))


def run_algorithm(theta_list, directions, boundary_signs, *, eta, b, delta,
                  D, epsilon, beta):
    T, d = theta_list.shape
    x = np.zeros(d)
    grad_sum = np.zeros(d)
    action_sum = np.zeros(d)

    internal_points = np.empty((T, d))
    played_points = np.empty((T, d))
    observed_losses = np.empty(T)
    reference_losses = np.empty(T)
    update_sizes = np.empty(T)
    exploration_distances = np.empty(T)
    expected_exploration_costs = np.empty(T)
    shrinkage_count = 0

    for t in range(T):
        H = hessian_phi(x, D)
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        sqrt_eigenvalues = np.sqrt(eigenvalues)

        # A = b H^{-1/2}. Work in the Hessian eigenbasis.
        u_in_basis = eigenvectors.T @ directions[t]
        exploration_step = eigenvectors @ (b * u_in_basis / sqrt_eigenvalues)
        y = x + exploration_step

        if np.linalg.norm(y) > D + 1e-10:
            raise ValueError(f"Played point is outside K at round {t + 1}.")

        reference_loss = theta_list[t] @ y + beta * (y @ y)
        # observed_loss = reference_loss + sigma(
        #     y, epsilon, action_sum, D, boundary_signs[t]
        # )
       
        A_inv_u = eigenvectors @ (sqrt_eigenvalues * u_in_basis / b)
       
        observed_loss = reference_loss + sigma2(
                            epsilon, x,  A_inv_u
                        )

        grad_sum += d * observed_loss * A_inv_u

        internal_points[t] = x
        played_points[t] = y
        observed_losses[t] = observed_loss
        reference_losses[t] = reference_loss
        exploration_distances[t] = np.linalg.norm(exploration_step)

        # For the quadratic reference loss,
        # E_t[h_t(y_t) - h_t(x_t)] = tr(A A^T) / (2d).
        expected_exploration_costs[t] = (
            b**2 * np.sum(1.0 / eigenvalues) / (2.0 * d)
        )

        # Measure only transitions between the plotted internal points.
        if t + 1 < T:
            x_next, shrinkage_active = update_x(grad_sum, eta, D, delta)
            step = x_next - x
            update_sizes[t] = np.sqrt(step @ H @ step)
            shrinkage_count += int(shrinkage_active)
            x = x_next

        action_sum += y

    return {
        "x": internal_points,
        "y": played_points,
        "observed_losses": observed_losses,
        "reference_losses": reference_losses,
        "update_sizes": update_sizes[:-1],
        "exploration_distances": exploration_distances,
        "expected_exploration_costs": expected_exploration_costs,
        "shrinkage_count": shrinkage_count,
    }


def plot_cumulative_loss(ax, rounds, runs, loss_key, label, color):
    cumulative = np.cumsum([run[loss_key] for run in runs], axis=1)
    mean = cumulative.mean(axis=0)
    ax.plot(rounds, mean, label=label, color=color)

    if len(runs) > 1:
        half_width = 1.96 * cumulative.std(axis=0, ddof=1) / np.sqrt(len(runs))
        ax.fill_between(
            rounds, mean - half_width, mean + half_width,
            color=color, alpha=0.15,
        )


def main():
    d, T = 2, 2000
    D, G = 1.2, 0.4
    C, n_runs = 800, 1
    beta = 0.16
    rng = np.random.default_rng(42)

    if T < 2 or not 0 <= C < T:
        raise ValueError("This parameterization requires T >= 2 and 0 <= C < T.")

    epsilon = C / T
    delta = np.sqrt(C / T)
    L = G * D + beta * D**2 + epsilon
    denominator = 4.0 * d * np.sqrt(L * T)

    eta2 = np.sqrt(np.log(T)) / denominator

    gamma = max(delta, 1 / T)

    eta = np.sqrt(np.log(1 / gamma)) / (
        T**(2/3) * d * D * L
    )

    b = T**(-1.0 / 6.0) / D

    # Values are (learning rate, exploration scale, domain shrinkage).
    variants = {
        "SS-SCRiBLe": (eta, b, delta),
        "Convex": (eta, b, 0),
        "SCRiBLe": (eta2, 1, 0),
    }
    colors = {
        "SS-SCRiBLe": "tab:orange",
        "Convex": "tab:green",
        "SCRiBLe": "tab:blue",
    }
    results = {name: [] for name in variants}

    for _ in range(n_runs):
        # Pair the three algorithms on the same reference losses,
        # exploration directions, and boundary signs within each run.
        # theta_list = rng.normal(size=(T, d))
        # theta_list *= G / np.linalg.norm(theta_list, axis=1, keepdims=True)

        # noise = rng.normal(size=(T, d))
        # noise /= np.linalg.norm(noise, axis=1, keepdims=True)

        # preferred_direction = np.zeros(d)
        # preferred_direction[0] = -0.5

        # bias = 0.85
        # theta_list = G * (
        #     bias * preferred_direction + (1.0 - bias) * noise
        # )
        
        theta_list = np.zeros((T, d))
        theta_list[:, 0] = -G

        directions = rng.normal(size=(T, d))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        boundary_signs = rng.choice([-1, 1], size=T)

        for name, (eta_variant, b_variant, delta_variant) in variants.items():
            results[name].append(
                run_algorithm(
                    theta_list,
                    directions,
                    boundary_signs,
                    eta=eta_variant,
                    b=b_variant,
                    delta=delta_variant,
                    D=D,
                    epsilon=epsilon,
                    beta=beta
                )
            )

    # rounds = np.arange(1, T + 1)
    # fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)

    # for name, runs in results.items():
    #     plot_cumulative_loss(
    #         axes[0], rounds, runs, "observed_losses", name, colors[name]
    #     )
    #     plot_cumulative_loss(
    #         axes[1], rounds, runs, "reference_losses", name, colors[name]
    #     )

    # axes[0].set_ylabel("Cumulative observed loss")
    # axes[1].set_ylabel("Cumulative reference loss")
    # for ax in axes:
    #     ax.set_xlabel("Round")
    #     ax.grid(alpha=0.25)
    # axes[0].legend()

    rounds = np.arange(1, T + 1)
    fig, ax = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)

    for name, runs in results.items():
        plot_cumulative_loss(
            ax, rounds, runs, "observed_losses", name, colors[name]
        )

    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative observed loss")
    ax.grid(alpha=0.25)
    ax.legend()

    # fig.savefig("cumulative_losses.pdf", bbox_inches="tight")
    fig.savefig("cumulative_losses.eps", format="eps",
            bbox_inches="tight", dpi=300)

    # Use the first paired run and the same axis limits in all three panels.
    w_star = np.array([D, 0.0]) 

    fig, axes = plt.subplots(
        1, 3, figsize=(10, 3.6), sharex=True, sharey=True,
        constrained_layout=True,
    )
    for ax, (name, runs) in zip(axes, results.items()):
        x = runs[0]["x"]
        y = runs[0]["y"]
        ax.add_patch(plt.Circle((0, 0), D, fill=False, color="0.6"))

        if variants[name][2] > 0:
            ax.add_patch(plt.Circle(
                (0, 0), (1.0 - variants[name][2]) * D,
                fill=False, color="0.6", linestyle="--",
                label=r"Boundary of $\mathcal{K}_{\delta}$",
            ))

        ax.scatter(y[:, 0], y[:, 1], s=6, alpha=0.3, color=colors[name],
                   label=r"Played $y_t$")
        ax.plot(x[:, 0], x[:, 1], color="black", linewidth=0.8,
                label=r"Internal $x_t$")

        ax.scatter(
            w_star[0], w_star[1],
            marker="*", s=115, color="crimson",
            edgecolors="white", linewidths=0.7, zorder=10,
            label=r"optimum $w$",
        )
        
        ax.set(
            xlim=(-1.05 * D, 1.05 * D),
            ylim=(-1.05 * D, 1.05 * D),
            xlabel="First coordinate",
            title=name,
        )
        ax.set_aspect("equal")

    axes[0].set_ylabel("Second coordinate")
    axes[0].legend(fontsize=7, loc="upper right")
    # fig.savefig("exploration_points_2d.pdf", bbox_inches="tight")
    fig.savefig("exploration_points_2d.eps", format="eps",
            bbox_inches="tight", dpi=300)

    

    plt.show()


if __name__ == "__main__":
    main()