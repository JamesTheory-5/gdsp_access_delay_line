# gdsp_access_delay_line
```python
"""
gdsp_access_delay_line.py

Fully differentiable time-varying fractional delay line implemented in pure
functional JAX. This module focuses on array access patterns inspired by
Gamma's access.h utilities: index mapping, wrapping, and interpolation over
a circular buffer, but expressed in a JAX-friendly, differentiable style.

MODULE NAME:
access_delay_line

DESCRIPTION:
Time-varying fractional delay line with a circular buffer and linear
interpolation between neighboring samples. Delay time is smoothed with a
1-pole low-pass to avoid zipper noise, and reading from the delay line
uses wrapped indices analogous to Gamma's Wrap accessor and IndexMap
functionality.

INPUTS:
- x : input audio sample (tick) or sequence of samples (process), shape (T,)
- delay_samples : desired delay in samples; per-tick scalar or per-sample
                  array of shape (T,) for process()
- feedback : feedback gain (scalar), typically in [-0.99, 0.99]
- smooth_coeff : smoothing coefficient for delay (scalar in [0, 1));
                 higher values => slower change in delay

OUTPUTS:
- y : delayed output sample (tick) or sequence (process), shape (T,)

STATE VARIABLES:
(state) = (buffer, write_idx, delay_z)
- buffer   : jnp.ndarray, shape (max_delay_samples,), circular buffer
- write_idx: jnp.int32 scalar, current write position in buffer
- delay_z  : jnp.float32 scalar, smoothed delay value (in samples)

EQUATIONS / MATH:

Let:
- N           = length of circular buffer
- x[n]        = input at time n
- d[n]        = target delay in samples at time n
- a           = smooth_coeff clamped to [0, 0.9999]
- d̃[n]       = smoothed delay (state delay_z)

Parameter smoothing:
    a        = clip(smooth_coeff, 0, 0.9999)
    d̃[n]    = clip(a * d̃[n-1] + (1 - a) * d[n], 0, N - 2)

Read index (in samples, fractional):
    w[n]     = write_idx[n]       # integer write index
    r[n]     = w[n] - d̃[n]       # floating-point read index
    r̃[n]    = mod(r[n], N)       # wrap into [0, N)

Neighbor indices:
    i0       = floor(r̃[n])
    frac     = r̃[n] - i0
    i1       = mod(i0 + 1, N)

Linear interpolation:
    y[n]     = (1 - frac) * buffer[i0] + frac * buffer[i1]

Feedback and write:
    f        = clip(feedback, -0.9999, 0.9999)
    write    = x[n] + f * y[n]

Circular buffer update:
    buffer'[w[n]] = write
    buffer'       = buffer with one-element update at index w[n]
    write_idx'    = mod(w[n] + 1, N)
    delay_z'      = d̃[n]

STATE UPDATE:
    state[n+1] = (buffer', write_idx', delay_z')

THROUGH-ZERO RULES:
The read position r[n] is allowed to go negative or exceed N; the modulo
wrapping r̃[n] = mod(r[n], N) ensures we always access within [0, N).

PHASE WRAPPING RULES:
The circular buffer index space is treated as a phase-like variable in
[0, N). All indexing is wrapped via jnp.mod, analogous to Gamma's Wrap
access policy.

NONLINEARITIES:
- Clamping of smooth_coeff to [0, 0.9999]
- Clamping of feedback to [-0.9999, 0.9999]
- Clamping of smoothed delay to [0, N - 2] to keep interpolation safe

INTERPOLATION RULES:
- Linear interpolation between the two nearest wrapped buffer positions.

TIME-VARYING COEFFICIENT RULES:
- Delay time d[n] may vary per-sample. It is smoothed over time by the
  1-pole recursion on delay_z, ensuring continuity and differentiability.

NOTES:
- max_delay_samples sets the maximum allowed delay; delay_samples must not
  exceed max_delay_samples - 2 to avoid accessing outside the linear
  interpolation range. The module enforces this via clipping.
- All operations are differentiable with respect to x, delay_samples,
  feedback, and smooth_coeff.
- No classes, no dicts, state is a tuple only.
- All control flow inside jit is expressed via JAX primitives only.
"""

from typing import Tuple

import jax
from jax import jit, lax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Public API
#   - access_delay_line_init(...)
#   - access_delay_line_update_state(...)
#   - access_delay_line_tick(x, state, params)
#   - access_delay_line_process(x, state, params)
# ---------------------------------------------------------------------------


def access_delay_line_init(
    max_delay_samples: int,
    initial_delay_samples: float,
    feedback: float,
    smooth_coeff: float,
    dtype=jnp.float32,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Initialize delay line state and default parameters.

    Args:
        max_delay_samples: Maximum delay in samples (buffer length). Must be >= 2.
        initial_delay_samples: Initial delay in samples (will be clamped).
        feedback: Initial feedback gain (will be clamped).
        smooth_coeff: Initial smoothing coefficient (will be clamped).
        dtype: JAX dtype for internal buffers (default: float32).

    Returns:
        state: (buffer, write_idx, delay_z)
        params: (delay_samples, feedback, smooth_coeff) as scalars (JAX arrays)
    """
    max_delay_samples = int(max_delay_samples)
    if max_delay_samples < 2:
        raise ValueError("max_delay_samples must be >= 2")

    buffer = jnp.zeros((max_delay_samples,), dtype=dtype)
    write_idx = jnp.array(0, dtype=jnp.int32)

    # Clamp and set delay in samples
    max_delay_for_init = float(max_delay_samples - 2)
    init_delay = jnp.clip(
        jnp.array(initial_delay_samples, dtype=dtype), 0.0, max_delay_for_init
    )

    # Clamp feedback
    fb = jnp.clip(jnp.array(feedback, dtype=dtype), -0.9999, 0.9999)

    # Clamp smoothing coefficient
    sc = jnp.clip(jnp.array(smooth_coeff, dtype=dtype), 0.0, 0.9999)

    delay_z = init_delay

    state = (buffer, write_idx, delay_z)
    params = (init_delay, fb, sc)
    return state, params


@jit
def access_delay_line_update_state(
    state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    params: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray]:
    """
    Update time-varying coefficients based on the current state and parameters.

    This implements parameter smoothing and constraint enforcement, inspired by
    Gamma's access mapping utilities.

    Args:
        state: (buffer, write_idx, delay_z)
        params: (delay_samples, feedback, smooth_coeff) for this tick

    Returns:
        new_state: (buffer, write_idx, delay_z_next)
        delay_current: smoothed and clipped delay in samples (scalar)
        feedback_clamped: clamped feedback (scalar)
    """
    buffer, write_idx, delay_z = state
    delay_samples, feedback, smooth_coeff = params

    # Clamp smoothing coefficient to [0, 0.9999]
    smooth_coeff_clamped = jnp.clip(smooth_coeff, 0.0, 0.9999)

    # Maximum safe delay: N - 2 to allow i1 = i0 + 1
    buffer_len = buffer.shape[0]
    max_delay = jnp.array(buffer_len - 2, dtype=buffer.dtype)

    # 1-pole smoothing of delay parameter
    delay_raw = delay_samples
    delay_next = smooth_coeff_clamped * delay_z + (1.0 - smooth_coeff_clamped) * delay_raw

    # Clamp delay to [0, max_delay]
    delay_current = jnp.clip(delay_next, 0.0, max_delay)

    # Clamp feedback to stable range
    feedback_clamped = jnp.clip(feedback, -0.9999, 0.9999)

    # Update state (buffer and write_idx unchanged here)
    new_state = (buffer, write_idx, delay_current)
    return new_state, delay_current, feedback_clamped


@jit
def access_delay_line_tick(
    x: jnp.ndarray,
    state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    params: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Process a single audio sample through the fractional delay line.

    Args:
        x: input sample (scalar JAX array)
        state: (buffer, write_idx, delay_z)
        params: (delay_samples, feedback, smooth_coeff) for this tick,
                each scalar or 0-D JAX array.

    Returns:
        y: output sample (scalar JAX array)
        new_state: updated (buffer, write_idx, delay_z)
    """
    # First update smoothed parameters and state
    state_s, delay_current, feedback_clamped = access_delay_line_update_state(state, params)
    buffer, write_idx, delay_z = state_s  # delay_z is now the smoothed delay

    buffer_len = buffer.shape[0]
    buffer_len_f = jnp.array(buffer_len, dtype=buffer.dtype)

    # Compute floating-point read position (may be out-of-range)
    write_idx_f = write_idx.astype(buffer.dtype)
    read_pos = write_idx_f - delay_current

    # Wrap read position into [0, buffer_len) (phase-like wrapping)
    read_pos_wrapped = jnp.mod(read_pos, buffer_len_f)

    # Neighbor indices for linear interpolation
    i0 = jnp.floor(read_pos_wrapped).astype(jnp.int32)
    frac = read_pos_wrapped - i0.astype(buffer.dtype)
    i1 = jnp.mod(i0 + 1, buffer_len)

    # Sample the buffer with wrapped indices
    y0 = buffer[i0]
    y1 = buffer[i1]

    # Linear interpolation
    one = jnp.array(1.0, dtype=buffer.dtype)
    y = (one - frac) * y0 + frac * y1

    # Feedback and write value
    write_val = x + feedback_clamped * y

    # Circular buffer update via lax.dynamic_update_slice (functional update)
    write_val_vec = jnp.reshape(write_val, (1,))
    buffer_new = lax.dynamic_update_slice(buffer, write_val_vec, (write_idx,))

    # Increment and wrap write index
    write_idx_new = jnp.mod(write_idx + 1, buffer_len).astype(jnp.int32)

    new_state = (buffer_new, write_idx_new, delay_z)
    return y, new_state


def access_delay_line_process(
    x: jnp.ndarray,
    state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    params: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Process a sequence of samples through the delay line using lax.scan.

    Args:
        x: input signal, shape (T,)
        state: initial state (buffer, write_idx, delay_z)
        params:
            (delay_samples, feedback, smooth_coeff), where:
            - delay_samples: array of shape (T,) (time-varying delay)
            - feedback: scalar or 0-D JAX array
            - smooth_coeff: scalar or 0-D JAX array

    Returns:
        y: output signal, shape (T,)
        final_state: state after processing all samples
    """
    x = jnp.asarray(x)
    delay_seq, feedback, smooth_coeff = params
    delay_seq = jnp.asarray(delay_seq)

    if delay_seq.shape != x.shape:
        raise ValueError("delay_samples must have the same shape as x in process().")

    def step_fn(carry, inputs):
        x_n, d_n = inputs
        # Per-sample params: delay varies with time; feedback and smooth_coeff are static
        params_n = (d_n, feedback, smooth_coeff)
        y_n, carry_next = access_delay_line_tick(x_n, carry, params_n)
        return carry_next, y_n

    inputs = (x, delay_seq)
    final_state, y = lax.scan(step_fn, state, inputs)
    return y, final_state


# ---------------------------------------------------------------------------
# Smoke test, plotting, and listening examples
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    try:
        import sounddevice as sd  # type: ignore
        HAVE_SD = True
    except Exception:
        HAVE_SD = False

    # Basic smoke test: impulse response of the delay line
    sample_rate = 48000
    duration_sec = 0.1
    num_samples = int(sample_rate * duration_sec)

    # Impulse input
    x_np = np.zeros(num_samples, dtype=np.float32)
    x_np[0] = 1.0
    x = jnp.asarray(x_np)

    # Create a static delay sequence (e.g., 480 samples = 10 ms at 48 kHz)
    static_delay_samples = 480.0
    delay_seq = jnp.ones_like(x) * static_delay_samples

    # Initialize delay line
    max_delay_samples = 2048
    state, base_params = access_delay_line_init(
        max_delay_samples=max_delay_samples,
        initial_delay_samples=static_delay_samples,
        feedback=0.3,
        smooth_coeff=0.99,
    )

    # Use per-sample delay sequence, but keep feedback and smooth_coeff from base_params
    _, fb, sc = base_params
    params_seq = (delay_seq, fb, sc)

    # Process signal
    y, final_state = access_delay_line_process(x, state, params_seq)
    y_np = np.array(y)

    # Plot impulse response
    t = np.arange(num_samples) / sample_rate
    plt.figure(figsize=(10, 4))
    plt.plot(t, y_np, label="Delayed impulse response")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("access_delay_line: Impulse Response")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Listen example (if sounddevice is available)
    if HAVE_SD:
        print("Playing delayed impulse response...")
        sd.play(y_np, sample_rate)
        sd.wait()
    else:
        print("sounddevice not available; skipping audio playback.")

    # Suggested follow-up prompts (as comments):
    #
    # 1. "Modify access_delay_line so that delay_samples is specified in seconds
    #     instead of samples, and add a sample_rate parameter."
    #
    # 2. "Add an option for cubic interpolation instead of linear, while keeping
    #     everything differentiable and jittable."
    #
    # 3. "Create a time-varying vibrato effect by modulating delay_samples with
    #     a sinusoidal LFO inside a new JAX module."
    #
    # 4. "Extend this module to a stereo delay where left and right channels
    #     share a buffer but have different time-varying delays."

```
