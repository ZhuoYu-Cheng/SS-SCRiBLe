import numpy as np
import matplotlib.pyplot as plt
import asyncio


def barrier_phi(x, D):
    # Log-barrier function for Euclidean ball ||x|| < D
    norm_sq = np.dot(x, x)
    if norm_sq >= D ** 2:
        raise ValueError("x is outside the domain (||x||^2 >= D^2)")
    return -np.log(D**2 - norm_sq)


def hessian_phi(x, D):
    # Hessian of the log-barrier function
    norm_sq = np.dot(x, x)
    if norm_sq >= D ** 2:
        raise ValueError("x is outside the domain (||x||^2 >= D^2)")
    
    d = len(x)
    denom = D**2 - norm_sq
    outer = np.outer(x, x)

    # Hessian = scaled identity + rank-1 correction
    H = (2 / denom) * np.eye(d) + (4 / denom**2) * outer
    return H


def update_x(grad_sum, eta, D, delta, SSSCRiBLe=0):
    # Compute the FTRL-style update using closed-form solution
    g = grad_sum
    G_sq = np.dot(g, g)
    
    # Avoid division by zero
    if G_sq == 0:
        return np.zeros(len(g))
    
    # Solve quadratic equation for alpha
    sqrt_term = np.sqrt(4 + 4 * (eta**2) * G_sq * (D**2))
    alpha = (-2 + sqrt_term) / (2 * eta * G_sq)
    
    x_t = -alpha * g

    # Optional shrinking to stay strictly inside feasible region
    radius = (1 - delta) * D
    norm = np.linalg.norm(x_t)
    
    
    if norm > radius and SSSCRiBLe:
        x_t = x_t / norm * radius
    return x_t


def sigma(x, D, epsilon):
    """
    Adversarial perturbation term
    """
    norm = np.linalg.norm(x)
    diff = max(D - norm, 1e-6)   # not 0
    sinx = np.sin(1/(diff))

    return epsilon * abs(sinx)


async def ScriblePlay(C_loss_list2, epsilon, theta_list, u_list, d, T, D, G):
    # Standard SCRiBLe
    x_t = np.zeros(d)
    grad_sum = np.zeros(d)

    delta = 1 / T
    eta = np.sqrt(np.log(T)) / (4 * d * np.sqrt(T*(G*D+0.5*D**2+epsilon)))
    All_reward1 = 0.0
    
    for t in range(T):
        # Compute local geometry (Hessian of barrier)
        hessian_barrier = hessian_phi(x_t, D)

        # Eigen-decomposition for inverse square root
        eigvals, eigvecs = np.linalg.eigh(hessian_barrier)
        eigvals = np.clip(eigvals, 1e-8, None)
        A = eigvecs @ np.diag(eigvals**(-0.5)) @ eigvecs.T

        # Sample direction from Dikin ellipsoid
        u = u_list[t]
        y_t = x_t + np.dot(A, u)

        if np.linalg.norm(y_t) >= D:
            print("error: outside feasible set")

        theta = theta_list[t]
          
        epsilon3 = sigma(y_t, D, epsilon)

        # Observed reward
        reward = np.dot(theta, y_t) + 0.5 * np.dot(y_t, y_t) + epsilon3

        # Gradient estimator
        A_inv = np.linalg.inv(A)
        g = d * reward * np.dot(A_inv, u)
        
        grad_sum += g

        # Update decision
        x_t = update_x(grad_sum, eta, D, delta)
        All_reward1 += reward
    
    TrueC_loss = All_reward1 
    C_loss_list2.append(TrueC_loss)


async def SS_SCRiBLe(C_loss_list1, epsilon, theta_list, u_list, d, T, D, G):
    # Shrunk SCRiBLe (keeps iterates away from boundary)
    x_t = np.zeros(d)
    grad_sum = np.zeros(d)

    delta = T**(-1)
    if epsilon != 0:
        delta = np.sqrt(epsilon)

    eta = np.sqrt(np.log(1 / delta)) / (4 * d * np.sqrt(T*(G*D+0.5*D**2+epsilon)))
    b = T**(-1/6)* D**(-1)
    All_reward1 = 0.0
    
    for t in range(T):
        hessian_barrier = hessian_phi(x_t, D)

        eigvals, eigvecs = np.linalg.eigh(hessian_barrier)
        eigvals = np.clip(eigvals, 1e-8, None)
        A = b*eigvecs @ np.diag(eigvals**(-0.5)) @ eigvecs.T

        u = u_list[t]
        y_t = x_t + np.dot(A, u)
        if np.linalg.norm(y_t) >= D:
            print("error: outside feasible set")

        theta = theta_list[t]
        epsilon3 = sigma(y_t, D, epsilon)

        reward = np.dot(theta, y_t) + 0.5 * np.dot(y_t, y_t) + epsilon3

        A_inv = np.linalg.inv(A)
        g = d * reward * np.dot(A_inv, u)
        
        grad_sum += g
    
        x_t = update_x(grad_sum, eta, D, delta, SSSCRiBLe=1)
        All_reward1 += reward
    
    TrueC_loss = All_reward1 
    C_loss_list1.append(TrueC_loss)

async def ConvexScrible(C_loss_list3, epsilon, theta_list, u_list, d, T, D, G):
    x_t = np.zeros(d)
    grad_sum = np.zeros(d)

    delta = T**(-1)
    if epsilon != 0:
        delta = np.sqrt(epsilon)

    eta = np.sqrt(np.log(1 / delta)) / (4 * d * np.sqrt(T*(G*D+0.5*D**2+epsilon)))
    b = T**(-1/6)* D**(-1)
    All_reward1 = 0.0
    
    for t in range(T):
        hessian_barrier = hessian_phi(x_t, D)

        eigvals, eigvecs = np.linalg.eigh(hessian_barrier)
        eigvals = np.clip(eigvals, 1e-8, None)
        A = b*eigvecs @ np.diag(eigvals**(-0.5)) @ eigvecs.T

        u = u_list[t]
        y_t = x_t + np.dot(A, u)
        if np.linalg.norm(y_t) >= D:
            print("error: outside feasible set")

        theta = theta_list[t]
        epsilon3 = sigma(y_t, D, epsilon)

        reward = np.dot(theta, y_t) + 0.5 * np.dot(y_t, y_t) + epsilon3

        A_inv = np.linalg.inv(A)
        g = d * reward * np.dot(A_inv, u)
        
        grad_sum += g
    
        x_t = update_x(grad_sum, eta, D, delta)
        All_reward1 += reward
    
    TrueC_loss = All_reward1 
    C_loss_list3.append(TrueC_loss)


async def main():
    C_loss_list1 = []
    C_loss_list2 = []
    C_loss_list3 = []

    d = 10
    T = 2000
    D = 5
    G = 10    
    
    rng = np.random.default_rng(seed=42)

    theta_list = rng.normal(size=(T, d))
    # Normalize theta to fixed norm G
    Theta_norms = np.linalg.norm(theta_list, axis=1, keepdims=True)
    theta_list = theta_list / Theta_norms * G
        
    for k in range(50):
        # Different corruption levels
        C_list = np.array([0, 200, 400, 600, 800])
        epsilon_list = C_list / T
        
        # Random directions (normalized)
        u_list = rng.normal(size=(T, d))
        # uniform distribution on the sphere
        u_list = u_list / np.linalg.norm(u_list, axis=1, keepdims=True)

        for t in range(len(epsilon_list)):
            await SS_SCRiBLe(C_loss_list1, epsilon_list[t], theta_list, u_list, d, T, D, G)
            await ScriblePlay(C_loss_list2, epsilon_list[t], theta_list, u_list, d, T, D, G)
            await ConvexScrible(C_loss_list3, epsilon_list[t], theta_list, u_list, d, T, D, G)
            print("Process", t)

    n = len(epsilon_list)

    average_C_loss1 = []
    average_C_loss2 = []
    average_C_loss3 = []
    std_C_loss1 = []
    std_C_loss2 = []
    std_C_loss3 = []

    for j in range(n):
        group1 = C_loss_list1[j::n]
        avg1 = np.mean(group1)
        average_C_loss1.append(avg1)

        std1 = np.std(group1, ddof=1)
        std_C_loss1.append(std1)

        group2 = C_loss_list2[j::n]
        average_C_loss2.append(np.mean(group2))

        std2 = np.std(group2, ddof=1)
        std_C_loss2.append(std2)

        group3 = C_loss_list3[j::n]
        average_C_loss3.append(np.mean(group3))
        std3 = np.std(group3, ddof=1)
        std_C_loss3.append(std3)

    # 95% confidence interval
    n_runs = len(group1)
    ci_C_loss1 = [1.96 * s / np.sqrt(n_runs) for s in std_C_loss1]
    ci_C_loss2 = [1.96 * s / np.sqrt(n_runs) for s in std_C_loss2]
    ci_C_loss3 = [1.96 * s / np.sqrt(n_runs) for s in std_C_loss3]

    # Plot results
    plt.plot(C_list, average_C_loss1, color="orange", label="Shrunk-SCRiBLe", linestyle='-', marker='o')
    # plt.fill_between(
    #     C_list,
    #     np.array(average_C_loss1) - np.array(ci_C_loss1),
    #     np.array(average_C_loss1) + np.array(ci_C_loss1),
    #     alpha=0.5
    # )
    plt.errorbar(
    C_list,
    average_C_loss1,
    yerr=ci_C_loss1,
    fmt='-o',
    capsize=3,
    color="orange",
    linewidth=1
    )

    plt.plot(C_list, average_C_loss2, color="blue", label="SCRiBLe", linestyle='-', marker='o')
    plt.errorbar(
    C_list,
    average_C_loss2,
    yerr=ci_C_loss2,
    fmt='-o',
    capsize=3,
    color="blue",
    linewidth=1
    )

    plt.plot(C_list, average_C_loss3, color="green", label="Convex", linestyle='-', marker='o')
    plt.errorbar(
    C_list,
    average_C_loss3,
    yerr=ci_C_loss3,
    fmt='-o',
    capsize=3,
    color="green",
    linewidth=1
    )

    plt.xlabel("C")
    plt.ylabel("Cumulative loss")
    plt.title("SS-SCRiBLe vs SCRiBLe vs Bandit OCO with T=2000")

    # plt.ylim(
    #     # bottom=min(min(average_C_loss1), max(average_C_loss2), max(average_C_loss3)) * 1.3,
    #     top=max(max(average_C_loss1), max(average_C_loss2), max(average_C_loss3)) * 1.5
    # )

    plt.grid(True)
    plt.legend()
    plt.savefig("figure.eps", format='eps', dpi=300)
    plt.show()


asyncio.run(main())