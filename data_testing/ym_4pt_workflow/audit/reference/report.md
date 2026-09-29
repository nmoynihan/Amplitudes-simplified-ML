# Scientific audit of the supplied four-gluon amplitude

**Verdict: independently validated as the on-shell, tree-level, color-ordered Yang–Mills four-gluon amplitude with ordering (1,2,3,4), with coupling, color factors, and the conventional overall i stripped.** This conclusion follows from exact equality to the Yang–Mills vertex expression, not only from Ward tests or agreement with an existing dataset.

**Historical report, audited on 21 September 2026.** The scientific findings below are retained; local links and reproduction commands have been adapted to this repository package. See [reference provenance](README.md).

 Source: [gluon4feyn1234_model_ready.csv](../../inputs/gluon4feyn1234_model_ready.csv).

The file has one headerless row, ID 1, containing 48 additive terms. Size: 4,473 bytes. SHA256:

```text
f73c1e6b8459876394697b26d4f2585b6f51ed0eee52fa8412dd5ba9fbc6ca0c
```

The source was read without modification. These checked-in results are historical evidence from that audit; rerunning the portable workflow writes fresh evidence to the chosen output directory.

## Assumptions and scope

Interpret the dots as Lorentz contractions, with all external momenta outgoing and

\[
p_i^2=0,\qquad \sum_{i=1}^4p_i=0,\qquad e_i\cdot p_i=0.
\]

The explicit four-vector checks use metric \((+---)\). Define

\[
s=(p_1+p_2)^2=2p_1\cdot p_2,\quad
t=(p_2+p_3)^2=2p_2\cdot p_3=2p_1\cdot p_4,\quad
u=(p_1+p_3)^2=-s-t.
\]

The exact tensor proof uses masslessness, momentum conservation, and transversality only; it does not use four-dimensional Gram identities or a special polarization gauge. The helicity checks are specifically four-dimensional. The result is a partial amplitude; obtaining the full color-dressed scattering amplitude requires the coupling and appropriate color trace sum. It is not an off-shell Green function.

## Decisive equality to the Yang–Mills vertices

Contracting the standard cubic kinematic vertex with two transverse polarizations gives

\[
J_{ij}^{\mu}=(e_i\cdot e_j)(p_i-p_j)^{\mu}
+2(e_i\cdot p_j)e_j^{\mu}-2(e_j\cdot p_i)e_i^{\mu}.
\]

Writing \(X\) for the expression in the CSV, the fresh symbolic calculation proves

\[
\boxed{X=
\frac{J_{12}\cdot J_{34}}{2s}
+\frac{J_{23}\cdot J_{41}}{2t}
+(e_1\cdot e_3)(e_2\cdot e_4)
-\frac{(e_1\cdot e_2)(e_3\cdot e_4)+(e_1\cdot e_4)(e_2\cdot e_3)}{2}.}
\]

The difference is **exactly zero**, as a rational function of independent scalar contractions. No coefficient or normalization was fitted to the CSV. The two exchange terms and the contact term were constructed separately from the cubic current and quartic tensor.

For normalized ordered rules \(V_3=iC/\sqrt2\), propagator \(-i\eta/P^2\), and \(V_4=iQ/2\), this is \(A_4^{\rm tree}=iX\). Here \(Q\) contracts to twice the first contact product minus the other two products in the displayed equation. These are the conventions of Dixon, Figure 5 and equation (40), with \(\mathrm{Tr}(T^aT^b)=\delta^{ab}\); his equations (42) and (44) give the corresponding helicity amplitudes. [Dixon, hep-ph/9601359](https://arxiv.org/pdf/hep-ph/9601359).

## Exact consistency checks

| Test | Fresh result |
|---|---|
| Polarization dependence | Every term has degree one in each external polarization |
| Momentum scaling | Every term has degree zero under a common momentum rescaling |
| Ward identities | \(X(e_i\to p_i)=0\) exactly, for each of the four legs |
| Cyclic symmetry | All three nontrivial cyclic rotations agree exactly |
| Reflection | \(X(1,2,3,4)=X(4,3,2,1)\) exactly |
| Photon decoupling | \(X(1,2,3,4)+X(2,1,3,4)+X(2,3,1,4)=0\) |
| Four-point BCJ relation | \(sX(1,2,3,4)-uX(1,3,2,4)=0\) |
| Locality | Reduced denominator is \(ab\), where \(a=s/2,b=t/2\) |
| Exchange residues | Exactly the contractions of the corresponding cubic currents |
| Independent four-dimensional checks | 48 linear-polarization and 48 circular-polarization evaluations agree exactly with normalized vertices |
| Helicity check | All 16 helicity assignments agree with the four-point tree formulas as exact functions of scattering angle |

All algebra used exact integers, rationals, and symbolic expressions in SymPy 1.14.0. Residuals are algebraically zero; there is no floating-point tolerance. For identically vanishing amplitudes a relative numerical residual is unnecessary and would be undefined.

Gauge invariance alone would not establish that a function is the Yang–Mills amplitude: many gauge-invariant functions have the wrong poles or residues. Here the full vertex equality supplies the stronger identification.

## Apparent poles and factorization

Individual CSV terms contain nonadjacent contractions in denominators and products that appear to generate double poles. These are representation artifacts. After all on-shell relations are applied to the complete sum, the reduced denominator is only \(ab\). In particular, the \(u=0\) limit is finite for \(s\ne0\), and there are no double \(s\) or \(t\) poles.

Direct symbolic residue checks give

\[
\lim_{s\to0}sX=\left.\frac12J_{12}\cdot J_{34}\right|_{s=0},
\qquad
\lim_{t\to0}tX=\left.\frac12J_{23}\cdot J_{41}\right|_{t=0}.
\]

The currents obey \((p_i+p_j)\cdot J_{ij}=0\) on shell. Consequently the longitudinal parts of an internal polarization sum decouple, and these are the standard cubic-vertex factorization residues with the stated phase conventions. Residues were checked algebraically in independent invariants, so they do not rely on a degenerate real collinear numerical configuration.

Direct term-by-term evaluation of the original CSV at \(u=0\) still encounters division by zero before the cancellation. Use the reduced or vertex representation to evaluate that removable singularity or take limits. This is a defect of the numerical representation, not of the underlying amplitude.

## Independent helicity evidence

Take \(x=\tan(\theta/2)>0\), \(c=(1-x^2)/(1+x^2)\), \(q=2x/(1+x^2)\), and

\[
p_1=(-1,0,0,-1),\quad p_2=(-1,0,0,1),\quad
p_3=(1,q,0,c),\quad p_4=(1,-q,0,-c).
\]

Use transverse vectors \(b_1=(0,1,0,0)\), \(b_2=-b_1\), \(b_3=(0,c,0,-q)\), \(b_4=-b_3\), \(y=(0,0,1,0)\), and normalized circular polarizations \(e_i^{h}=(b_i+ihy)/\sqrt2\). A separate script factors the momentum bispinors into explicit spinors and compares the CSV directly with

\[
\frac{\langle ij\rangle^4}{\langle12\rangle\langle23\rangle\langle34\rangle\langle41\rangle}
\]

for each pair of negative helicities \(i,j\), with the global \(i\) omitted. The comparison is symbolic in \(x\), rather than an angular fit. In the specified external-state phases:

| Helicity | CSV value |
|---|---|
| \(--++\), \(++--\) | \(1+x^2\) |
| \(-+-+\), \(+-+-\) | \(x^4/(1+x^2)\) |
| \(-++-\), \(+--+\) | \(1/(1+x^2)\) |
| Other ten assignments | \(0\) |

This includes all-plus, single-minus, and their parity partners. The separate point tests used \(\cos\theta=3/5,5/13,-7/25\); each tested the full 16-product linear basis, all 16 circular states, a generic polarization with longitudinal gauge shifts, and all four Ward replacements.

## Provenance and interpretation

The requested file is byte-identical to [the bundled prepared input](../../inputs/gluon4feyn1234_model_ready.csv). Its preparation completed an earlier seed and then re-expanded an already-derived compact three-term field-strength expression. The portable [workflow README](../../README.md) documents that preparation and its historical provenance.

Thus this audit validates the physical function in this particular file. It does not establish that a model independently discovered the amplitude or its compact form. Agreement with another repository file was not used as the decisive physics test here.

## Reproduction

From the repository root, with the package requirements installed:

```bash
python -m data_testing.ym_4pt_workflow.audit \
  --input data_testing/ym_4pt_workflow/inputs/gluon4feyn1234_model_ready.csv \
  --output-dir /tmp/ym_4pt_audit
```

The [audit README](../README.md) describes each independent stage. The parent [workflow README](../../README.md) also covers regeneration of the prepared input and model-dependent steps.

Historical machine-readable evidence: [symbolic results](symbolic_results.json), [vertex checks](vertices_results.json), and [helicity checks](parke_taylor_results.json).
