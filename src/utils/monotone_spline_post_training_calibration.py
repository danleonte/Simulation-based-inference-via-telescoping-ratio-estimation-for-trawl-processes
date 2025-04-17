import numpy as np
import jax
import jax.numpy as jnp
import optax
import distrax

# Dummy classifier output and labels
# probs = jnp.array([0.1, 0.4, 0.7, 0.9])
# labels = jnp.array([0, 0, 1, 1])

# probs = pred_prob_Y.squeeze()
# labels = Y


def fit_spline(probs, labels, num_bins=10, random_seed=0, params=None):
    """calibrates the provided probs via splines and saves the params of the spline"""

    # Number of bins for spline
    # num_bins = 8
    param_dim = 3 * num_bins + 1

    # Initialize spline params (unconstrained)
    # key = jax.random.PRNGKey(0)
    if params is None:
        key = jax.random.PRNGKey(random_seed)
        params = jax.random.normal(key, shape=(param_dim,))

    # BCE loss function

    def bce_loss(params, probs, labels):
        spline = distrax.RationalQuadraticSpline(
            params=params,
            range_min=0.0,
            range_max=1.0,
            boundary_slopes='identity'  # can be changed
        )
        calibrated = spline.forward(probs)
        # Clamp to prevent log(0)
        eps = 1e-8
        calibrated = jnp.clip(calibrated, eps, 1 - eps)
        loss = -jnp.mean(labels * jnp.log(calibrated) +
                         (1 - labels) * jnp.log(1 - calibrated))
        return loss

    @jax.jit
    def update(params, opt_state, probs, labels):
        loss, grads = jax.value_and_grad(bce_loss)(params, probs, labels)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    lr = 0.005
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    # Training loop
    for step in range(251):
        params, opt_state, loss = update(params, opt_state, probs, labels)
        if step % 50 == 0:
            print(f"Step {step}, Loss: {loss:.5f}")

    lr = lr / 5

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)
    # Training loop
    for step in range(251):
        params, opt_state, loss = update(params, opt_state, probs, labels)
        if step % 50 == 0:
            print(f"Step {step}, Loss: {loss:.5f}")

    lr = lr / 10

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)
    # Training loop
    for step in range(251):
        params, opt_state, loss = update(params, opt_state, probs, labels)
        if step % 50 == 0:
            print(f"Step {step}, Loss: {loss:.5f}")

    lr = lr / 5

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)
    # Training loop
    for step in range(251):
        params, opt_state, loss = update(params, opt_state, probs, labels)
        if step % 50 == 0:
            print(f"Step {step}, Loss: {loss:.6f}")

    return params

    # Use fitted spline for calibration
    # spline = distrax.RationalQuadraticSpline(
    #    params=params, range_min=0.0, range_max=1.0)
    # calibrated_probs = spline.forward(probs)
    # print("Calibrated:", calibrated_probs)

    # linspace = np.linspace(0, 1, 100)
    # perform isotonic regression, Beta and Plat scaling
    # lr = LogisticRegression(C=99999999999)
    # iso = IsotonicRegression(y_min=0.001, y_max=0.999)
    # bc = BetaCalibration(parameters="abm")

    # lr.fit(pred_prob_Y, np.array(Y))
    # iso.fit(pred_prob_Y, np.array(Y))
    # bc.fit(pred_prob_Y,  np.array(Y))

    # pr = [lr.predict_proba(linspace.reshape(-1, 1))[:, 1], iso.predict(linspace),
    #      bc.predict(linspace), spline.forward(linspace)]
    # methods_text = ['logistic', 'isotonic', 'beta', 'spline']
