//! CTC (Connectionist Temporal Classification) loss with gradient.
//!
//! Pure-Rust implementation of the CTC forward-backward algorithm in log
//! space, returning the per-batch loss and the gradient of the loss with
//! respect to the input logits. The caller is responsible for using the
//! gradient to update model parameters via backprop.
//!
//! References:
//! - Graves et al., "Connectionist Temporal Classification" (2006)
//! - Hannun et al., "Sequence Modeling with CTC" (2017), Awni et al.

/// CTC loss + per-timestep gradient for one sample.
///
/// `logits[t, c]` is the network output at time `t` for class `c` (raw logits,
/// NOT log-softmax). `targets` is the target sequence of class indices.
///
/// Returns `(loss, grad_logits)` where:
/// - `loss` is the negative log-likelihood of the target sequence
/// - `grad_logits[t, c]` is d(loss) / d(logits[t, c])
pub fn ctc_loss_and_grad(
    logits: &[f32],
    targets: &[usize],
    num_classes: usize,
    blank: usize,
) -> (f32, Vec<f32>) {
    let t = logits.len() / num_classes;
    debug_assert_eq!(logits.len(), t * num_classes);
    let s = targets.len();

    // Compute log-softmax of logits for stability and reuse.
    let log_probs = log_softmax_2d(logits, t, num_classes);

    // Build extended label: [blank, l1, blank, l2, blank, ..., ls, blank]
    // Length 2*s + 1, with blanks at even indices.
    let mut ext = Vec::with_capacity(2 * s + 1);
    for i in 0..=s {
        ext.push(blank);
        if i < s {
            ext.push(targets[i]);
        }
    }
    let ext_len = ext.len();

    // Forward variables (log space): alpha[t][s]
    let mut alpha = vec![f32::NEG_INFINITY; t * ext_len];
    // Initialization at t = 0 (row 0, i.e. indices without the t*ext_len offset).
    alpha[0] = log_probs[blank];
    if s > 0 {
        alpha[1] = log_probs[targets[0]];
    }

    // Forward recursion.
    for ti in 1..t {
        for si in 0..ext_len {
            let lp = log_probs[ti * num_classes + ext[si]];
            let mut vals: Vec<f32> = Vec::with_capacity(3);
            // Stay at si
            if alpha[(ti - 1) * ext_len + si] > f32::NEG_INFINITY {
                vals.push(alpha[(ti - 1) * ext_len + si]);
            }
            // Move from si - 1
            if si > 0 && alpha[(ti - 1) * ext_len + (si - 1)] > f32::NEG_INFINITY {
                vals.push(alpha[(ti - 1) * ext_len + (si - 1)]);
            }
            // Move from si - 2 (only valid if L[si] == blank OR L[si] != L[si-2])
            if si >= 2 && alpha[(ti - 1) * ext_len + (si - 2)] > f32::NEG_INFINITY {
                let allow_skip = ext[si] == blank || ext[si] != ext[si - 2];
                if allow_skip {
                    vals.push(alpha[(ti - 1) * ext_len + (si - 2)]);
                }
            }
            let sum = if vals.is_empty() {
                f32::NEG_INFINITY
            } else {
                log_sum_exp(&vals)
            };
            alpha[ti * ext_len + si] = lp + sum;
        }
    }

    // Loss = -log P(target | input) = -log(alpha[t-1][2s] + alpha[t-1][2s+1])
    let last_row = &alpha[(t - 1) * ext_len..];
    let last_two = if ext_len >= 2 {
        vec![last_row[ext_len - 1], last_row[ext_len - 2]]
    } else {
        vec![last_row[ext_len - 1]]
    };
    let log_z = log_sum_exp(&last_two);
    let loss = -log_z;

    // Backward variables (log space): beta[t][s]
    let mut beta = vec![f32::NEG_INFINITY; t * ext_len];
    // Initialization at t = T-1: end states have beta = 0 (log 1); non-end = -inf.
    beta[(t - 1) * ext_len + (ext_len - 1)] = 0.0;
    if ext_len >= 2 {
        beta[(t - 1) * ext_len + (ext_len - 2)] = 0.0;
    }

    // Backward recursion.
    for ti in (0..t - 1).rev() {
        for si in 0..ext_len {
            let mut vals: Vec<f32> = Vec::with_capacity(3);
            // Stay at si at t+1 (emit L[si] at t+1)
            let lp_stay = log_probs[(ti + 1) * num_classes + ext[si]];
            if beta[(ti + 1) * ext_len + si] > f32::NEG_INFINITY {
                vals.push(beta[(ti + 1) * ext_len + si] + lp_stay);
            }
            // Move from si+1 to si at t+1 (emit L[si+1] at t+1)
            if si + 1 < ext_len && beta[(ti + 1) * ext_len + (si + 1)] > f32::NEG_INFINITY {
                let lp_move = log_probs[(ti + 1) * num_classes + ext[si + 1]];
                vals.push(beta[(ti + 1) * ext_len + (si + 1)] + lp_move);
            }
            // Move from si+2 to si at t+1 (emit L[si+2] at t+1; valid if L[si+2] == blank OR L[si+2] != L[si])
            if si + 2 < ext_len && beta[(ti + 1) * ext_len + (si + 2)] > f32::NEG_INFINITY {
                let allow_skip = ext[si + 2] == blank || ext[si + 2] != ext[si];
                if allow_skip {
                    let lp_skip = log_probs[(ti + 1) * num_classes + ext[si + 2]];
                    vals.push(beta[(ti + 1) * ext_len + (si + 2)] + lp_skip);
                }
            }
            beta[ti * ext_len + si] = if vals.is_empty() {
                f32::NEG_INFINITY
            } else {
                log_sum_exp(&vals)
            };
        }
    }

    // Posterior gamma[t][c] = sum over s where L[s] == c: alpha[t][s] * beta[t][s] / Z
    // In log space: log gamma[t][c] = logaddexp over s: alpha[t][s] + beta[t][s] - log_z
    let mut log_gamma = vec![f32::NEG_INFINITY; t * num_classes];
    for ti in 0..t {
        for si in 0..ext_len {
            let a = alpha[ti * ext_len + si];
            let b = beta[ti * ext_len + si];
            if a > f32::NEG_INFINITY && b > f32::NEG_INFINITY {
                let c = ext[si];
                let new_val = a + b - log_z;
                let cur = log_gamma[ti * num_classes + c];
                log_gamma[ti * num_classes + c] = if cur > f32::NEG_INFINITY {
                    logaddexp(cur, new_val)
                } else {
                    new_val
                };
            }
        }
    }

    // Gradient: d loss / d logits[t][c] = softmax(logits)[t][c] - gamma[t][c]
    // softmax(logits)[t][c] = exp(log_probs[t][c])
    // gamma[t][c] = exp(log_gamma[t][c])
    let mut grad = vec![0.0f32; t * num_classes];
    for ti in 0..t {
        for c in 0..num_classes {
            let softmax_val = log_probs[ti * num_classes + c].exp();
            let gamma_val = if log_gamma[ti * num_classes +c] > f32::NEG_INFINITY {
                log_gamma[ti * num_classes + c].exp()
            } else {
                0.0
            };
            grad[ti * num_classes + c] = softmax_val - gamma_val;
        }
    }

    (loss, grad)
}

/// Batch CTC loss + gradient.
///
/// `logits` is [B, T, C] flattened. `targets` is a list of target sequences.
/// `target_lengths[b]` is the length of each target. All targets use the same
/// blank index. Returns (mean_loss, grad_logits).
pub fn ctc_loss_batch(
    logits: &[f32],
    target_lengths: &[usize],
    targets: &[usize],
    num_classes: usize,
    blank: usize,
) -> (f32, Vec<f32>) {
    let b = target_lengths.len();
    let t = logits.len() / (b * num_classes);
    let mut total_loss = 0.0f32;
    let mut total_grad = vec![0.0f32; logits.len()];

    let mut offset = 0usize;
    for bi in 0..b {
        let l = target_lengths[bi];
        let target = &targets[offset..offset + l];
        offset += l;
        let sample_logits = &logits[bi * t * num_classes..(bi + 1) * t * num_classes];
        let (loss, grad) = ctc_loss_and_grad(sample_logits, target, num_classes, blank);
        total_loss += loss;
        for i in 0..grad.len() {
            total_grad[bi * t * num_classes + i] += grad[i];
        }
    }
    let denom = b as f32;
    total_loss /= denom;
    for g in total_grad.iter_mut() {
        *g /= denom;
    }
    (total_loss, total_grad)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn log_softmax_2d(logits: &[f32], t: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; t * c];
    for ti in 0..t {
        let row = &logits[ti * c..(ti + 1) * c];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for ci in 0..c {
            sum += (row[ci] - max).exp();
        }
        let log_sum = max + sum.ln();
        for ci in 0..c {
            out[ti * c + ci] = row[ci] - log_sum;
        }
    }
    out
}

fn log_sum_exp(values: &[f32]) -> f32 {
    if values.is_empty() {
        return f32::NEG_INFINITY;
    }
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    let mut sum = 0.0f32;
    for v in values {
        sum += (v - max).exp();
    }
    max + sum.ln()
}

fn logaddexp(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let (lo, hi) = if a > b { (b, a) } else { (a, b) };
    hi + (-(hi - lo)).exp().ln_1p()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_softmax_2d() {
        let logits = vec![1.0, 2.0, 3.0];
        let out = log_softmax_2d(&logits, 1, 3);
        let s: f32 = out.iter().map(|v| v.exp()).sum();
        assert!((s - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_logaddexp() {
        assert!((logaddexp(0.0, 0.0) - 0.693147).abs() < 1e-4);
        assert!((logaddexp(1.0, 2.0) - 2.313262).abs() < 1e-4);
    }

    #[test]
    fn test_ctc_simple() {
        // Single timestep, single target
        let logits = vec![0.0, 1.0, 0.0]; // 1 timestep, 3 classes
        let targets = vec![1usize];
        let num_classes = 3;
        let blank = 0;
        let (loss, grad) = ctc_loss_and_grad(&logits, &targets, num_classes, blank);
        // For T=1, target=[1]:
        //   alpha[0][0] = y_0(blank), alpha[0][1] = y_0(1) (end state s=1), alpha[0][2] = -inf
        // Z = exp(alpha[0][1]) = y_0(1) (end at s=1)
        // Loss = -log(y_0(1))
        // softmax([0,1,0])[1] = e/(2+e) ≈ 0.422 → loss ≈ 0.86
        assert!(loss > 0.0 && loss < 2.0, "loss should be moderate, got {loss}");
        // grad sum per t = sum(softmax) - sum(gamma) = 1 - 1 = 0
        let sum_grad: f32 = grad.iter().sum();
        assert!(sum_grad.abs() < 1e-3, "grad sum should be ~0, got {sum_grad}");
    }

    #[test]
    fn test_ctc_converges_on_identity() {
        // Construct logits that perfectly predict the target.
        let t = 4usize;
        let _s = 1usize;
        let num_classes = 3usize;
        let blank = 0;
        let targets = vec![1usize];
        // Make logits heavily favor [blank, l1, blank, blank] for the 4 timesteps
        let mut logits = vec![0.0f32; t * num_classes];
        // t=0: blank strongly
        logits[0 * num_classes + 0] = 5.0;
        logits[0 * num_classes + 1] = 0.0;
        logits[0 * num_classes + 2] = 0.0;
        // t=1: l1 strongly
        logits[1 * num_classes + 0] = 0.0;
        logits[1 * num_classes + 1] = 5.0;
        logits[1 * num_classes + 2] = 0.0;
        // t=2: blank strongly
        logits[2 * num_classes + 0] = 5.0;
        logits[2 * num_classes + 1] = 0.0;
        logits[2 * num_classes + 2] = 0.0;
        // t=3: blank strongly
        logits[3 * num_classes + 0] = 5.0;
        logits[3 * num_classes + 1] = 0.0;
        logits[3 * num_classes + 2] = 0.0;

        let (loss, grad) = ctc_loss_and_grad(&logits, &targets, num_classes, blank);
        assert!(loss < 0.1, "expected near-zero loss for identity, got {loss}");
        let sum_grad: f32 = grad.iter().sum();
        assert!(sum_grad.abs() < 1e-3, "grad sum should be ~0, got {sum_grad}");
    }

    #[test]
    fn test_ctc_grad_finite_on_random() {
        let t = 8usize;
        let num_classes = 5usize;
        let blank = 0;
        let targets = vec![1, 2, 3];
        let mut logits = vec![0.0f32; t * num_classes];
        for v in logits.iter_mut() {
            *v = (rand::random::<f32>() - 0.5) * 2.0;
        }
        let (loss, grad) = ctc_loss_and_grad(&logits, &targets, num_classes, blank);
        assert!(loss.is_finite(), "loss should be finite, got {loss}");
        assert!(grad.iter().all(|v| v.is_finite()), "grad should be finite");
        // gamma sums to 1 per t (CTC property), so total grad sum = 0
        let sum_grad: f32 = grad.iter().sum();
        assert!(sum_grad.abs() < 1e-2, "grad sum should be ~0, got {sum_grad}");
    }

    #[test]
    fn test_alpha_beta_consistency() {
        let _t = 4usize;
        let num_classes = 3usize;
        let blank = 0;
        let targets = vec![1, 2];
        let logits = vec![1.0, 2.0, 0.5, 0.8, 1.5, 0.3, 1.2, 2.0, 0.5, 0.6, 1.0, 1.8];
        let (loss, _) = ctc_loss_and_grad(&logits, &targets, num_classes, blank);
        assert!(loss > 0.0, "loss should be > 0 for non-perfect prediction, got {loss}");
    }
}