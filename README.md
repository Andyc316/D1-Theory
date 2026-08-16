# D1 Unified Field Theory

**Andrew Cottham — Independent Theoretical Physics Researcher**

D1 Field Theory is a proposed fundamental field framework investigating whether spacetime, gravitation, cosmology and particle-physics phenomena can emerge from the dynamics of an underlying D1 field.

This repository contains research papers, computational models, numerical experiments and supporting material developed as part of the D1 research programme.

---

## Research Programme

The D1 research programme investigates a number of connected questions in fundamental physics.

### Spacetime and Gravitation

Can the observed structure of spacetime and gravitational dynamics emerge from the underlying D1 field?

A major current objective is to investigate whether Einstein's gravitational equations can be obtained as an effective description of the D1 field.

### Black Holes

D1 research investigates the behaviour of the field in the extreme-energy regime associated with gravitational collapse, including possible non-singular descriptions of black-hole interiors.

### Cosmology

The D1 cosmological programme investigates the evolution of the universe from the early universe through to the present epoch, including inflation, baryogenesis and cosmic acceleration.

### Particle Physics

The D1 framework is also being developed as a possible foundation for particle physics, including investigations of composite Higgs models and discrete vacuum structures.

### Quantum Foundations

A longer-term objective is to investigate whether the D1 framework can provide a common foundation connecting gravitational and quantum descriptions.

---

# Computational Research

This repository contains computational work supporting the D1 research programme.

## GPS Clock Injection and Recovery

### `gps_injection_recovery.py`

This script performs signal injection and recovery testing using real GPS clock data.

The calculation injects a sinusoidal signal into clock residuals, fits a model to the resulting data and evaluates the ability of the analysis to recover the injected signal.

The code provides a computational framework for testing signal-recovery methods against noisy timing data.

---

## Cosmological Reconstruction

### `reconstruction.py`

This script implements a D1 cosmological reconstruction calculation based on the proposed relationship between present-day cosmological parameters and the history of the universe.

The calculation includes the determination of inflationary e-folds from specified model parameters.

This work is associated with the D1 cosmological investigation of whether information about the early universe can be connected to present-day measured quantities.

---

## Inflation and Slow-Roll Parameters

### `slow_roll_parameters`

This calculation investigates standard inflationary slow-roll quantities within the relevant D1 cosmological framework.

The calculations include quantities such as:

- scalar spectral index \(n_s\)
- tensor-to-scalar ratio \(r\)
- running of the spectral index

These quantities provide a connection between the theoretical inflationary model and observational cosmology.

---

# Research Papers

The D1 research programme has produced a series of theoretical and computational papers covering cosmology, gravitation, black holes and particle physics.

Research outputs are archived through Zenodo and Figshare.

### Selected research

- *Time as the First Dimension and as Freedom of Movement, and the Emergence of Space.*
- *The D1 Unified Field Theory*
- *The D1 Theory of Baryogenesis and Cosmic Acceleration*
- *The D1 Field: Unified Dynamics and the Complete Cosmological History (Big Bang to Today)*
- *The D1 Theory of Black Holes*
- *The D1 Cosmological Model: Unification of Baryogenesis and Cosmic Acceleration*
- *The Universe from One Number: How Today's Measured Dark-Energy Density Encodes 63 e-folds of Starobinsky-like Inflation and the Entire Cosmic History*
- *D1: Deforming the Mexican Hat – Four Discrete Crystalline Vacua in a Natural Composite Higgs Framework*
- *Unification of General Relativity and Quantum Mechanics via Foundational D1 Field Theory*

The complete research archive is available through Zenodo.

---

# Published Research

## D1 Composite Higgs: Discrete Z₄ Crystalline Vacua – Numerical Confirmation and LHC Phenomenology

Published in the *Journal of Theoretical, Experimental, and Applied Physics*, Volume 2, Issue 3, 2026.

The paper develops a D1 Composite Higgs framework involving discrete \(Z_4\) crystalline vacua and investigates its numerical structure and phenomenological implications for LHC physics.

---

# Repository Structure

```text
D1-Theory/
│
├── Version_2_November_2025/
│   └── paper 4 v2.2.pdf
│
├── gps_injection_recovery.py
├── reconstruction.py
├── slow_roll_parameters
├── requirement.txt
└── README.md
