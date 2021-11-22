import numpy as np

def calculate_straight_line_path(x_input, x_baseline, steps):
    """Returns straight-line path between two points."""
    alphas = np.linspace(0.0, 1.0, num=steps + 1, endpoint=True)
    return [x_baseline + alpha * (x_input - x_baseline) for alpha in alphas]


def l1_distance(x1, x2):
    """Returns L1 distance between two inputs."""
    return np.abs(x1 - x2).sum()


def unbounded_guided_ig(x_input, x_baseline, steps, grad_func, fraction):
    """Returns unbounded Guided IG attribution."""
    l1_total = l1_distance(x_input, x_baseline)
    x = x_baseline
    attr = np.zeros_like(x_input, dtype=np.float64)
    for step in range(steps):
        grad = grad_func(x)
        gamma = np.inf
        while gamma > 1.0:
            grad[x_input == x_baseline] = np.inf
            l1_target = l1_total * (1 - (step + 1) / steps)
            l1_current = l1_distance(x, x_input)
            if l1_target == l1_current:
                break
            threshold = np.quantile(np.abs(grad), fraction, interpolation='lower')
            s = np.abs(grad) <= threshold
            l1_s = (np.abs(x - x_input) * s).sum()
            x_temp = x.copy()
            if l1_s > 0:
                gamma = (l1_current - l1_target) / l1_s
            else:
                gamma = np.inf
            if gamma > 1.0:
                x[s] = x_input[s]
            else:
                x[s] = (1 - gamma) * x[s] + gamma * x_input[s]
            grad[grad == np.inf] = 0
            attr[s] = attr[s] + (x - x_temp)[s] * grad[s]
    return attr


def anchored_guided_ig(x_input, x_baseline, grad_func, steps=200, fraction=0.1, anchors=20):
    """Returns Guided IG attribution."""
    attr = np.zeros_like(x_input, dtype=np.float64)
    anchor_points = calculate_straight_line_path(
        x_input=x_input, 
        x_baseline=x_baseline, 
        steps=anchors + 1
    )
    for anchor in range(anchors + 1):
        seg_attr = unbounded_guided_ig(
            x_input=anchor_points[anchor + 1],
            x_baseline=anchor_points[anchor],
            steps=int(steps / (anchors + 1)),
            grad_func=grad_func,
            fraction=fraction,
        )
        attr += seg_attr
    return attr