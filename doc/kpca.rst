.. _kpca:

Principal Components Analysis
=============================

Principal component analysis is a statistical technique that uses an orthogonal
transformation to expressa set of possibly correlated variables a set of 
variables that are linearly uncorrelated known as principal components. The 
orthogonal transformation is defined in such a way that the first principal
component captures the greatest amount of variation of the original set of
variables. 

Suppose that :math:`\mathbf{x} \in \mathbb{R}^{p}` is a random vector with
covariance matrix :math:`\mathbf{\Sigma}` Each of the components of 
:math:`\mathbf{x}` may be correlated.

.. math::
    \operatorname{V}\left(\mathbf{w}^{\intercal}\mathbf{x}\right)
    = \mathbf{w}^{\intercal}\operatorname{V}(\mathbf{x})\mathbf{w}
    = \mathbf{w}^{\intercal}\mathbf{\Sigma}\mathbf{w}

This is a test citation [fukanaga]_.

Kernel Principal Components Analysis
====================================
