# Deep Koopman and HAVOC Optimization Toolkit

## What is Koopman and what shall one even care about such another useless cryptic python library?

Koopman analysis is looking into quasi-chaotic, non-linear system simulation and its propagation through time.

In other words, we could just forecast non-linear dynamic systems and describe it efficiently, thought its frequency behaviour, up a a specific degree of accuraccy, other classical systems couldn't!

### How does it work?

When looking into stadard Fourier Analysis, we describe a linear system between the actual oscillators and the desired output space:

$\matcal{L}\left(\Theta, \omega\right) = - \ln E\left(f_{\Theta'}(\Omega(\omega t)) | x, \Theta'\right)$

With $E$ being the actual error function between our ground truth $x$ and the oscillator $\Omega(\omega, t)$.

Usually this is the square error (analogous to the Fourier Transform):

$E\left(f_{\Theta'}(\Omega(\omega, t)) | x, \Theta'\right) = \sum_{t \in [1, .., T]} \left(x - f_{\Theta}(\Omega(\omega t)) \right)^2$

