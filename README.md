# esQueranto website

**esQueranto** is a JAX-based framework for the differentiable simulation and optimization of quantum-optical experiments. It uses JAX to combine automatic differentiation, compiled execution, vectorized numerical operations, and GPU acceleration [1,2].

The source code is not currently public. Nevertheless, its computational performance is demonstrated through a simple and reproducible task: optimizing a parameterized optical circuit to rediscover a discrete Fourier-transform unitary.

## Fourier-transform optimization

We consider a rectangular Clements interferometer acting on (M) optical modes [3]. It contains

[
\frac{M(M-1)}{2}
]

generalized beam splitters, each controlled by two real parameters, followed by (M) output phases. The complete circuit therefore contains

[
2\frac{M(M-1)}{2}+M=M^2
]

real parameters.

The circuit is optimized to reproduce the (M)-dimensional discrete Fourier transform

[
(F_M)_{jk}
==========

\frac{1}{\sqrt M}
\exp\left(\frac{2\pi i jk}{M}\right).
]

Given the parameterized unitary (U(\boldsymbol{\theta})), the objective function is

[
\mathcal L_U(\boldsymbol{\theta})
=================================

\frac{\left\lVert
U(\boldsymbol{\theta})-F_M
\right\rVert_F^2}{2M}.
]

This provides a controlled benchmark for evaluating the cost of differentiating through increasingly large quantum-optical circuits.

## Gradient-evaluation time

We measure the time required to evaluate the loss and its gradient while varying:

* the number of optical modes (M);
* the number of trainable circuit parameters.

For a scalar loss, JAX uses reverse-mode automatic differentiation to obtain the derivatives with respect to all trainable parameters through a single reverse pass, rather than performing an independent circuit simulation for every parameter.

Consequently, increasing the number of trainable parameters does not necessarily produce a proportional increase in evaluation time. JAX’s compiled and vectorized array operations further reduce the overhead associated with processing many parameters simultaneously.

The evaluation time nevertheless increases with (M), since larger circuits contain more optical elements and require a larger computational graph.

## GPU memory requirements

Execution time is only one limitation of differentiable quantum simulation. GPU memory becomes increasingly important when the number of photons is increased.

For (N) indistinguishable photons distributed among (M) optical modes, the dimension of the fixed-(N) Fock sector is

[
D_{N,M}
=======

{N+M-1\choose N}.
]

When all photon-number sectors from the vacuum up to (N) are stored, the total number of represented basis states is

[
D_{\leq N,M}
============

# \sum_{n=0}^{N}{n+M-1\choose n}

{N+M\choose N}.
]

In the following memory estimate, we denote the number of stored basis states by (D). Thus,

[
D=D_{N,M}
]

for a fixed-photon-number representation, or

[
D=D_{\leq N,M}
]

when every sector up to (N) is retained.

Assuming that each basis state is represented by

* one `complex128` amplitude, requiring (16) bytes;
* an occupation vector containing (M) `int32` indices, requiring (4M) bytes;

the persistent memory required for the amplitudes is

[
B_{\mathrm{amplitudes}}=16D,
]

while the memory required for the basis-state indices is

[
B_{\mathrm{indices}}=4MD.
]

The total persistent memory is therefore

[
\boxed{
B_{\mathrm{persistent}}
=======================

D(16+4M)\ \text{bytes}
}
]

or, in gibibytes,

[
B_{\mathrm{persistent}}^{(\mathrm{GiB})}
========================================

\frac{D(16+4M)}{2^{30}}.
]

This estimate represents only the memory needed to retain the quantum state and its basis description.

The actual peak GPU memory is larger. Applying quantum operations creates temporary output arrays and intermediate quantities, while reverse-mode differentiation may retain additional values and cotangents required during the backward pass. Compiler workspaces and memory-allocation overhead can further increase the peak.

The peak memory therefore depends not only on (M) and (N), but also on the sequence of quantum operations and the differentiation strategy. It must be measured during execution and cannot, in general, be inferred from the persistent-state size alone.

Together, the reported results show how the computational requirements of differentiable quantum-optical optimization are governed by two complementary limits:

[
\boxed{\text{gradient-evaluation time}}
\qquad\text{and}\qquad
\boxed{\text{peak GPU memory}}.
]

## References

[1] J. Bradbury, R. Frostig, P. Hawkins, M. J. Johnson, C. Leary, D. Maclaurin, G. Necula, A. Paszke, J. VanderPlas, S. Wanderman-Milne, and Q. Zhang, *JAX: Composable Transformations of Python+NumPy Programs*, 2018.

[2] JAX Developers, *JAX Documentation: High-Performance Array Computing and Automatic Differentiation*.

[3] W. R. Clements, P. C. Humphreys, B. J. Metcalf, W. S. Kolthammer, and I. A. Walmsley, “Optimal design for universal multiport interferometers,” *Optica* **3**, 1460–1465 (2016). DOI: 10.1364/OPTICA.3.001460.
