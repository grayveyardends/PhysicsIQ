"""physicsiq_pinn — physics-informed neural networks that find the weak
point of a CAD part, with no dataset.

The mental model in 30 seconds
------------------------------
A PINN is a tiny neural net u(x) trained so that the PHYSICS EQUATIONS
are its loss function. No training data: we sample random points inside
the part ("collocation points"), plug them into the PDE, and the residual
(how much the equations are violated) is the loss. Boundary conditions
(fixed face, loaded face) become extra loss terms. When the loss is low,
u(x) IS the displacement field, and stress falls out of its derivatives —
which JAX gives us exactly, via autodiff, no meshing, no FEM.

Package map (each folder is an extension point):
    geometry/   CAD file -> collocation + boundary points   (swap: better samplers)
    problems/   one file = one PDE                          (swap/add: your physics)
    models/     the neural net                              (swap: fancier nets)
    training/   the optimize loop                           (swap: LBFGS, schedules)
    postproc/   displacement -> stress -> weak points
    viz/        matplotlib plots
    cli.py      what FreeCAD actually invokes
"""

__version__ = "0.1.0"
