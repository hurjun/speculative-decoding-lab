# Derivation: acceptance sampling, the acceptance probability, and tokens per round

This note collects the three results that the implementation in
[`specdec/`](../specdec) relies on and that the unit tests verify numerically:

1. the modified rejection step emits a token distributed **exactly** as the
   target $p$ (losslessness),
2. the probability that a single draft token is accepted equals the overlap
   $\sum_x \min(p(x), q(x)) = 1 - \mathrm{TV}(p, q)$, and
3. the expected number of confirmed tokens per round is
   $(1 - \alpha^{K+1}) / (1 - \alpha)$ under the standard i.i.d. acceptance model.

Notation follows Leviathan et al. (2023) and Chen et al. (2023): $p$ is the
target next-token distribution and $q$ is the draft distribution over a shared
vocabulary $V$. The corresponding code is in
[`specdec/sampling.py`](../specdec/sampling.py).

## Setup

For one position, the draft proposes $x \sim q$. The token is **accepted** with
probability $\min\!\big(1,\, p(x)/q(x)\big)$. On rejection a replacement is drawn
from the normalized residual

$$r(\cdot) = \frac{\max(0,\, p(\cdot) - q(\cdot))}{\sum_{x} \max(0,\, p(x) - q(x))}.$$

This is `residual_distribution` in the code; the emitted token is the accepted
draft token or, on rejection, the residual draw.

A small identity is used repeatedly. For every token $x$,

$$\min(p(x), q(x)) + \max(0,\, p(x) - q(x)) = p(x), \tag{$\ast$}$$

because if $p(x) \ge q(x)$ the terms are $q(x) + (p(x) - q(x)) = p(x)$, and if
$p(x) < q(x)$ they are $p(x) + 0 = p(x)$. Summing $(\ast)$ over $V$ and using
$\sum_x p(x) = 1$ gives $\sum_x \max(0, p(x) - q(x)) = 1 - \sum_x \min(p(x), q(x))$,
so the residual's normalizer is $1 - \beta$ with $\beta := \sum_x \min(p(x), q(x))$.

## 1. Distribution preservation (losslessness)

**Claim.** The emitted token $z$ satisfies $\Pr[z = v] = p(v)$ for all $v \in V$,
independent of the draft $q$.

**Proof.** The emitted token equals $v$ either by accepting a proposed $x = v$,
or by rejecting and resampling $v$ from the residual:

$$\Pr[z = v] = \underbrace{q(v)\,\min\!\Big(1, \tfrac{p(v)}{q(v)}\Big)}_{\text{accept } x = v}
\; + \; \underbrace{(1 - \beta)\, r(v)}_{\text{reject, then draw } v}.$$

The first term is $\min(q(v), p(v))$. For the second, $\Pr[\text{reject}] = 1 - \beta$
(shown in Section 2) cancels the residual normalizer, leaving
$(1 - \beta)\, r(v) = \max(0,\, p(v) - q(v))$. Adding the two terms and applying
$(\ast)$,

$$\Pr[z = v] = \min(p(v), q(v)) + \max(0,\, p(v) - q(v)) = p(v). \qquad\blacksquare$$

Applying this position by position, the speculative sequence has the same joint
distribution as target-only sampling. The greedy ($T = 0$) specialization is
sharper: a draft token is accepted iff it is the target's argmax, and the
emitted token is the target's argmax otherwise, so the output is *bit-for-bit*
identical to greedy autoregressive decoding.

*Verified by:* `tests/test_sampling.py::test_emitted_token_matches_target_distribution`
(Monte Carlo on the primitive) and
`tests/test_generate_toy.py::test_greedy_speculative_matches_autoregressive`
(bit-for-bit through the full cached loop).

## 2. The acceptance probability equals the distribution overlap

**Claim.** The probability that a single draft token is accepted is

$$\alpha \;=\; \sum_{x} \min(p(x), q(x)) \;=\; 1 - \mathrm{TV}(p, q),
\qquad \mathrm{TV}(p, q) = \tfrac{1}{2}\sum_x |p(x) - q(x)|.$$

**Proof.** Averaging the acceptance probability over the draw $x \sim q$,

$$\Pr[\text{accept}] = \sum_x q(x)\,\min\!\Big(1, \tfrac{p(x)}{q(x)}\Big)
= \sum_x \min(q(x), p(x)) = \beta.$$

For the total-variation form, write $|p(x) - q(x)| = p(x) + q(x) - 2\min(p(x), q(x))$
and sum: $\sum_x |p - q| = 2 - 2\beta$, hence $\beta = 1 - \tfrac{1}{2}\sum_x |p - q|
= 1 - \mathrm{TV}(p, q)$. $\qquad\blacksquare$

So the acceptance rate is high exactly when the draft is close to the target in
total variation, which is the quantitative content of "a better draft accepts
more often." This $\alpha$ is the per-position acceptance probability; the
harness estimates it as `accepted / proposed`, which equals $\alpha$ in the limit
of one proposal per round (see Section 3).

*Verified by:*
`tests/test_acceptance_theory.py::test_marginal_acceptance_probability_equals_overlap`.

### Temperature dependence of $\alpha$

Because $\alpha = 1 - \mathrm{TV}(p, q)$ depends only on how close the two
distributions are, the softmax temperature $T$ has a predictable effect. Write the
temperature-scaled distributions $p_T(x) \propto \exp(\ell^p_x / T)$ and
$q_T(x) \propto \exp(\ell^q_x / T)$ from the target and draft logits $\ell^p, \ell^q$.
The two limits are exact:

- **$T \to \infty$:** both $p_T$ and $q_T$ converge to the uniform distribution, so
  $\mathrm{TV}(p_T, q_T) \to 0$ and $\alpha \to 1$. Every draft token is accepted in
  the limit, because the draft and target agree (both are uniform).
- **$T \to 0^+$:** both collapse onto their argmaxes. A single draft token is then
  accepted iff $\arg\max q = \arg\max p$, so $\alpha \to \mathbb{1}[\arg\max q = \arg\max p]$
  at a fixed context, and averaged over contexts the sampling acceptance approaches
  the greedy argmax-agreement rate.

The two limits are typically *different* values ($\alpha \to \text{argmax-agreement}$
versus $\alpha \to 1$), so $\alpha(T)$ need not be monotone: it is high at both ends
and dips in between, where the temperature-scaled distributions are sharp enough to
concentrate mass on a few tokens yet not so sharp that they have collapsed onto a
shared argmax. The toy temperature sweep in
[experiment 03](../experiments/03-toy-acceptance-sweep/) measures exactly this
U-shape: $\alpha$ bottoms out at $0.27$ near $T = 0.2$, rises back toward the
greedy argmax-agreement rate ($\approx 0.97$ for this pair) as $T \to 0$, and climbs
toward $1$ as $T$ grows (reaching $0.93$ at $T = 4$). Losslessness (Section 1) is
independent of $T$: the emitted token is exactly $p_T$-distributed at every
temperature, so temperature trades output diversity against acceptance without ever
biasing the output distribution.

## 3. Expected confirmed tokens per round

Assume the standard analysis model of Leviathan et al. (2023): within a round the
$K$ positions are accepted i.i.d. with the same probability $\alpha$. A round
proposes $K$ draft tokens and stops at the first rejection. Let $N \in \{0, \dots, K\}$
be the number of accepted tokens before the first rejection. The round emits
$T = N + 1$ tokens: the accepted prefix plus exactly one more token (the
corrected token at the first rejection, or the bonus token when all $K$ are
accepted).

Since $\Pr[N \ge j] = \alpha^{j}$ for $1 \le j \le K$,

$$\mathbb{E}[N] = \sum_{j=1}^{K} \Pr[N \ge j] = \sum_{j=1}^{K} \alpha^{j}
= \frac{\alpha(1 - \alpha^{K})}{1 - \alpha},$$

and therefore the expected confirmed tokens per round is the geometric sum

$$\boxed{\;\mathbb{E}[T] = 1 + \mathbb{E}[N] = \sum_{j=0}^{K} \alpha^{j}
= \frac{1 - \alpha^{K+1}}{1 - \alpha}.\;}$$

**Relation to the $1 + K\alpha$ estimate.** A looser count applies linearity of
expectation to the $K$ positions *without* the stop-at-first-rejection
truncation: each position is accepted with probability $\alpha$, giving an
untruncated expectation of $K\alpha$ accepted tokens and $\mathbb{E}[T] \approx 1 + K\alpha$
(the form quoted in the reading notes). Because positions after the first
rejection can never actually be accepted, this over-counts: with truncation
$\mathbb{E}[N] = \sum_{j=1}^{K} \alpha^{j} \le K\alpha$, so $1 + K\alpha$ is an
**upper bound** on the exact $\mathbb{E}[T]$, tight as $\alpha \to 1$ and equal at
$K = 1$. Both expressions reduce to $1$ as $\alpha \to 0$. The geometric form is
the quantity to use when reasoning about realized throughput.

**Why the harness `accepted / proposed` falls with $K$.** The harness counts
$K$ proposals every round, so its acceptance estimate is

$$\widehat{\alpha}(K) = \frac{\mathbb{E}[N]}{K}
= \frac{1}{K}\sum_{j=1}^{K} \alpha^{j},$$

which equals the true $\alpha$ at $K = 1$ and decreases in $K$ (each added term
$\alpha^{j} \le \alpha$). This is the analytical counterpart of the declining
`accepted/proposed` column measured in
[experiment 02](../experiments/02-draft-length-sweep/); the quantitative gap
between the constant-$\alpha$ prediction and the measurement reflects that real
per-position acceptance is not constant and tends to decline at deeper positions.

*Verified by:*
`tests/test_acceptance_theory.py::test_expected_tokens_per_round_matches_formula`
(controlled i.i.d. setting, empirical $\mathbb{E}[T]$ vs. the geometric form).

## References

- Y. Leviathan, M. Kalman, Y. Matias. *Fast Inference from Transformers via
  Speculative Decoding.* ICML 2023. arXiv:2211.17192.
- C. Chen, S. Borgeaud, G. Irving, et al. *Accelerating Large Language Model
  Decoding with Speculative Sampling.* 2023. arXiv:2302.01318.
