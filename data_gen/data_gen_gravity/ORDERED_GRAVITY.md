# Fixed-order objects for the scalar–gravity benchmarks

This note specifies two different objects: a physically motivated ordering of
the **scalar flavour traces**, and smaller **auxiliary permutation seeds** whose
sums reproduce the two existing benchmark expressions. The latter have explicit
compact formulas, exact reconstruction proofs, and an implementation in
ordered_benchmarks.py. They provide a concrete possible target for a
simplification dataset; they are not universal colour-ordered gravity amplitudes.

## 1. Scope and conventions

The benchmarks are the five-point, tree-level, same-positive-helicity
scalar–gravity expressions in Cheung, Dersy and Schwartz, Eqs. (4.7) and (4.8).
The scalar has a cubic interaction. The four-scalar example selects one scalar
exchange channel by its flavour structure (footnote 17). The repository uses

\[
B_3=\overline{\mathcal M}_{\rm paper}(\phi_1\phi_2\phi_3h_4^+h_5^+),
\qquad
B_4=2\overline{\mathcal M}_{\rm paper}(\phi_1\phi_2\phi_3\phi_4h_5^+).
\]

Couplings and flavour tensors are stripped. In this note

\[
d_{ij}=p_i\cdot p_j,\qquad
X_i(a,b)=p_a\cdot F_i\cdot p_b=-X_i(b,a).
\]

Thus \(d_{ij}\) is **half** the usual massless Mandelstam invariant
\((p_i+p_j)^2\). Do not replace it by that invariant without adjusting constants.
Each graviton appears twice in each compact numerator term, consistent with its
factorized polarization tensor. The \(B_3\) and \(B_4\) stripped dimensions are
zero and minus two.

The two-piece identities below are rational identities using only commutativity
of scalar factors and antisymmetry of \(F_i\). Their interpretation as the paper's
physical benchmarks uses the stated on-shell and helicity conditions.

## 2. A natural physical ordering orders the scalars, not the gravitons

The benchmark paper says it obtains the scalar–gravity expressions by applying
the transmutation operators of Cheung, Shen and Wen to gravity amplitudes.
These operators create ordered scalar flavour structures. Applying a scalar
trace operator to each polarization copy gives a concrete candidate definition

\[
\mathcal B(\alpha\mid\beta;H)
 = {\cal N}\,\mathcal T[\alpha]\,
   \widetilde{\mathcal T}[\beta]\,\mathcal M_{\rm grav},
\]

where \(\alpha,\beta\) are cyclic orderings of the scalar labels and \(H\) is
the unordered set of graviton labels. The mother amplitude
\(\mathcal M_{\rm grav}\) here is unprojected KLT/extended gravity, with two
independent polarization copies, as required by the cited transmutation
construction. The surviving spectators are then selected as positive-helicity
gravitons. The chosen component is the one with the cubic scalar interactions;
this is not a claim about every coupling sector of an arbitrary scalar–gravity
Lagrangian.

The transmutation identity for the CHY integrand gives
\(\operatorname{PT}(\alpha)\operatorname{PT}(\beta)
\operatorname{Pf}\Psi_H\,\operatorname{Pf}\widetilde\Psi_H\).
Here \(\operatorname{PT}(\alpha)=
1/(z_{\alpha_1\alpha_2}\cdots z_{\alpha_m\alpha_1})\), and
\(\Psi_H\) is the spectator graviton matrix; its diagonal polarization–momentum
entries still sum over **all** external particles. With equal polarization
copies, its two Pfaffians multiply to a square.

The independent five-point implementation in verify_ordered_chy.py fixes
the measure and overall sign by the explicit definition

\[
\mathcal B(\alpha\mid\beta;H)
=-\sum_{z:\,E_i=0}
\frac{\operatorname{PT}(\alpha)\operatorname{PT}(\beta)
      (\operatorname{Pf}\Psi_H)^2}
     {{\det}'\Phi},\qquad
E_i=\sum_{j\ne i}\frac{d_{ij}}{z_i-z_j}.
\]

Explicitly, with \(z_{ab}=z_a-z_b\), the spectator matrix is

\[
\Psi_H=\begin{pmatrix}A&-C^{\mathsf T}\\ C&B\end{pmatrix},\quad
A_{ab}=\frac{d_{ab}}{z_{ab}},\quad
B_{ab}=\frac{\epsilon_a\cdot\epsilon_b}{z_{ab}},\quad
C_{ab}=\frac{\epsilon_a\cdot p_b}{z_{ab}}\quad(a\ne b),
\]

where its indices run over \(H\), \(A_{aa}=B_{aa}=0\), and
\(C_{aa}=-\sum_{j\ne a}\epsilon_a\cdot p_j/z_{aj}\), with the sum over
all five legs. The Jacobian has
\(\Phi_{ij}=d_{ij}/z_{ij}^2\) for \(i\ne j\) and
\(\Phi_{ii}=-\sum_{j\ne i}\Phi_{ij}\). In the implementation's puncture
gauge \((z_1,z_2,z_5)=(0,1,-1)\),
\({\det}'\Phi=\det\Phi_{\{3,4\},\{3,4\}}/(z_{12}z_{25}z_{51})^2\).
Both solutions are included with unit multiplicity.

The two five-point scattering-equation solutions give the following
correspondence to the repository normalization, checked independently against
the spinor formulas:

| Process | First scalar order | Second scalar order | Unordered gravitons | Target |
| --- | --- | --- | --- | --- |
| 3s2h | (1,2,3) | (1,2,3) | {4,5} | \(B_3\) |
| 4s1h | (1,2,3,4) | (1,3,2,4) | {5} | \(B_4\) |

For four scalars, these two cycles have only the nontrivial two-versus-two scalar partition
\(\{1,4\}\mid\{2,3\}\) in common. This explains the selected exchange channel.
Graviton emission can attach to the relevant scalar lines; the graviton is not
inserted as another colour-ordered scalar.

This is a literature-based definition with an independent numerical
identification of these two benchmark components. The symbolic proofs in the
next sections concern the explicit compact seed reconstruction, not a general
proof of this CHY identification for every theory or multiplicity.

There is no need to sum all flavour orderings to obtain these **particular**
benchmarks: each is already one selected coefficient. Reconstructing a
flavour-dressed amplitude instead requires the appropriate trace/flavour
weights. An unweighted sum of all orders is not that prescription.

## 3. Explicit fixed-order seed for three scalars and two gravitons

Define the ordered slots \((a,b,c;h,k)\), with three scalar labels followed by
two graviton labels, and set

\[
m_3(a,b,c;h,k)=
-\frac{
X_h(a,b)X_h(a,k)X_k(a,c)X_k(b,c)}
{d_{ah}d_{ak}d_{bh}d_{bk}d_{ck}d_{hk}}.
\]

The reference target is \(m_3(1,2,3;4,5)\). Its reconstruction is

\[
\boxed{B_3=m_3(1,2,3;4,5)+m_3(1,2,3;5,4).}
\]

The first seed is the first signed term of BENCHMARK_3S2H. Swapping \(4,5\)
in every momentum, field strength and denominator produces exactly its second
signed term after commuting scalar factors. This proves the identity without
numerical fitting or any omitted remainder.

As an independent check, with the repository's positive-helicity convention,

\[
m_3(1,2,3;4,5)=
\frac{\langle12\rangle\langle13\rangle\langle23\rangle[14][35]}
{\langle14\rangle\langle24\rangle\langle25\rangle
 \langle35\rangle\langle45\rangle}.
\]

Adding the swapped expression is Eq. (4.7). On conserved massless momenta in
the \(++\) sector, \(B_3\) is symmetric under \(S_3\) on the scalars and \(S_2\)
on the gravitons. Therefore, if an all-allowed-order sum is desired,

\[
B_3=\frac{1}{6}
\sum_{\pi\in S_3}\sum_{\tau\in S_2}
m_3(\pi(1),\pi(2),\pi(3);\tau(4),\tau(5)).
\]

There are twelve labelled summands, and the factor \(1/6\) is essential.
Pair each scalar permutation's two graviton orders to obtain one \(B_3\);
there are six such pairs. The scalar symmetry is proved symbolically on a
generic conserved spinor patch in the tests. It is not a generic mixed-helicity
identity for the same compact expression.

## 4. Explicit fixed-order seed for the selected four-scalar channel

Define ordered slots \((a,b,c,d;h)\), retaining the channel
\(\{a,d\}\mid\{b,c\}\), and set

\[
m_4(a,b,c,d;h)=
\frac{X_h(a,d)X_h(b,c)}
     {2d_{ad}d_{bc}d_{ah}d_{ch}}
+\frac{X_h(a,d)X_h(c,d)}
      {d_{bc}d_{ah}d_{ch}d_{dh}}.
\]

The reference target is \(m_4(1,2,3,4;5)\). Its reconstruction is

\[
\boxed{B_4=m_4(1,2,3,4;5)+m_4(3,4,1,2;5).}
\]

To prove it, denote the three signed terms in BENCHMARK_4S1H by
\(T_0,T_1,T_2\), in file order. The relabeling \(r=(13)(24)\) satisfies

\[
rT_0=T_0,\qquad rT_1=T_2,\qquad
m_4=\tfrac12 T_0+T_1.
\]

These identities follow directly from \(X_h(a,b)=-X_h(b,a)\), so
\(m_4+rm_4=T_0+T_1+T_2\). The seed has two terms and is not \(B_4/2\).
The paper's amplitude itself is \((m_4+rm_4)/2\).

The allowed larger permutation group is the stabilizer of the selected
partition,

\[
H=\operatorname{Stab}_{S_4}(\{\{1,4\},\{2,3\}\})
\cong(S_2\times S_2)\rtimes S_2.
\]

It has eight elements, generated by \((14),(23),(12)(34)\). On shell,
\(B_4\) is invariant under this group, and

\[
B_4=\frac14\sum_{\pi\in H}
m_4(\pi(1),\pi(2),\pi(3),\pi(4);5).
\]

The implementation rejects scalar permutations outside \(H\). Such a
permutation changes the selected channel. In particular,

\[
\sum_{\pi\in S_4}\pi B_4
=8\left(C_{14|23}+C_{12|34}+C_{13|24}\right),
\qquad C_{14|23}=B_4,
\]

where the other two coefficients are its relabelings. This expression includes
three exchange channels and is not the existing single-channel benchmark.
Actual flavour couplings determine which coefficients belong in a physical
amplitude and with which weights.

### A useful emission interpretation, and an alternative seed

Let \(v_i=d_{i5}\), \(e_i=\epsilon_5\cdot p_i\), \(A=d_{14}\), \(B=d_{23}\).
Momentum conservation and transversality imply
\(\sum_{i=1}^4v_i=\sum_{i=1}^4e_i=0\) and \(A=B+v_2+v_3\).
The same channel coefficient is exactly

\[
B_4=
\frac{e_1^2/v_1+e_4^2/v_4}{B}
+\frac{e_2^2/v_2+e_3^2/v_3}{A}
+\frac{(e_1+e_4)^2}{AB}.
\]

This displays radiation associated with the two ends and the internal scalar
line, and explains the channel-preserving symmetries. It also admits a
gauge-invariant one-term seed

\[
g(14|23;5)=
\frac{X_5(1,4)^2}{d_{15}d_{45}d_{23}(d_{15}+d_{45})},
\qquad B_4=g(14|23;5)+g(23|14;5).
\]

This alternative introduces a spurious pole at
\(d_{15}+d_{45}=d_{23}-d_{14}=0\), which cancels in the sum. The implemented
\(m_4\) avoids that additional denominator and stays within the existing
product-of-pair-dots target grammar.

## 5. Dataset contract

A future dataset should state which of these objects it targets.

1. **Physical scalar-order coefficient:** use scalar_order_left,
   scalar_order_right, graviton_legs, scalar coupling sector, normalization and
   channel metadata. These benchmarks already fix the two scalar orders above.
   There is no universal cyclic order on all five legs and no justification for
   removing nonadjacent graviton channels.
2. **Auxiliary ordered seed:** use the explicit m3 or m4 definition, reference
   role order (1,2,3,4,5), and decomposition version. Each row maps a scrambled
   seed to its compact seed. Reconstruction uses the two specified images and
   coefficients. A seed prediction is not compared directly with the full
   benchmark amplitude; reconstruct the sum first.

The seeds are individually gauge invariant because they are products of
field-strength contractions. Their sum and normalization are fixed, but their
decomposition is not unique: a term antisymmetric under the partner involution
can be added to the seed without changing the reconstruction. No claim is made
that these seeds separately satisfy all physical factorization constraints or
the ordered recursion of an unrelated supersymmetric theory.

The random targets in the present gravity training generator are synthetic
gauge-invariant rational expressions. Summing their permutations does not
automatically turn them into amplitudes of the scalar–gravity theory.

For held-out evaluation, reserve the entire parent benchmark family, **all
ordered component seeds**, their relabelings and scramble trajectories.
Excluding only the full benchmark target is insufficient once the model is
trained to simplify its components.

The dedicated entry point [ordered_gravity_gen.py](../ordered_gravity_gen.py)
implements the auxiliary-seed option, with synthetic training targets and
held-out scrambles of both reconstruction components. See the
[generation commands and output contract](README.md#ordered-component-traintest-generation).
Its structural and finite numerical exclusion checks apply to newly generated
outputs; they do not retroactively certify existing training corpora.

## 6. Reproduce the checks

Run from Amplitudes-simplified-ML in the scattering Python environment:

    python -m unittest data_gen.data_gen_gravity.test_ordered_benchmarks -v
    python -m data_gen.data_gen_gravity.ordered_benchmarks \
      --seeds 30 --benchmark-raw data/gravity/benchmarks_raw.csv.gz \
      --output data_gen/data_gen_gravity/ordered_benchmark_verification.json
    python -m data_gen.data_gen_gravity.verify_ordered_chy \
      --seeds 30 --output data_gen/data_gen_gravity/ordered_chy_verification.json

The tests prove the two-piece reconstruction with independent symbolic
antisymmetric variables for all allowed reference orders. They also prove the
larger-group symmetries under the appropriate kinematic assumptions, check
dimensions and field-strength counts, and compare against the paper's
independent spinor formulas.

The numerical report checks both reconstruction sums and gauge invariance at
30 momentum points with five reference/gauge choices per process. The saved
benchmark check confirms that all 200 rows use the two reconstructed targets,
100 each; it does not rerun every saved scrambled input's numerical validation.
The CHY report independently tests the scalar-order identification.

## Sources

- Cheung, Dersy and Schwartz, [Learning the Simplicity of Scattering
  Amplitudes](https://arxiv.org/html/2408.04720v2#S4.SS4), §4.4, Eqs. (4.7–4.8)
  and footnote 17: benchmark theory, normalization reference and flavour choice.
- Cheung, Shen and Wen, [Unifying Relations for Scattering
  Amplitudes](https://arxiv.org/pdf/1705.03025), §3, Eq. (24), Eqs. (55–56):
  trace/insertion operators and scalar ordering; restoring full flavour
  amplitudes requires flavour factors.
- Bollmann and Ferro, [Transmuting CHY
  formulae](https://arxiv.org/pdf/1808.07451), Eq. (4.10):
  ordered trace operators acting on a reduced Pfaffian leave the spectator
  Pfaffian and a Parke–Taylor factor.
- Drummond et al., [Tree-Level Amplitudes in N=8
  Supergravity](https://arxiv.org/pdf/0901.2363), §II.B: an example of a different
  nonunique ordered-subamplitude prescription; its supersymmetric recursion
  is not assumed for the scalar–gravity benchmarks here.
