# Deep Koopman and HAVOC Optimization Toolkit

## What is Koopman and what shall one even care about such another useless cryptic python library?

Koopman analysis is looking into quasi-chaotic, non-linear system simulation and its propagation through time.

In other words, we could just forecast non-linear dynamic systems and describe it efficiently, thought its frequency behaviour, up a a specific degree of accuraccy, other classical systems couldn't!


## *Classical* Koopman Analysis

When looking into stadard Fourier Analysis, we describe a linear system between the actual oscillators and the desired output space, minimizing the squared error:

$\begin{equation}E\left(K, \omega\right) = \sum_{t=0}^{T-1} \left(x - K \Omega(\omega t) \right)^2\end{equation}$

With $E$ being the actual error function between our ground truth $x$ and the oscillator $\Omega(\omega', t)$.


The oscillator $\Omega(\omega t)$ is defined in the same manner as in the Fourier wavelets:

$
\begin{equation}
\Omega(\omega t) = \begin{pmatrix} 
    \sin(\omega_1 t) \\
    \vdots \\
    \sin(\omega_N t) \\
    \cos(\omega_1 t) \\
    \vdots \\
    \cos(\omega_N t)
\end{pmatrix}
\end{equation}
$

In Koopman Analysis, however, we look into any type of function $f_{\Theta}$, mostly non-linear or at least quasi-linear:

$\begin{equation}E\left(f_{\Theta'}(\Omega(\omega', t)) | x, \Theta', \omega'\right) = \sum_{t=0}^{T-1} \left(x - f_{\Theta}(\Omega(\omega' t), \Theta') \right)^2\end{equation}$

Furthermore, from this error function, we can derive some type of pseudo $log$-likelihood, using any kind of error function for oscillators and periodic frequency elements:

$\begin{equation}\log L\left(\Theta', \omega'\right) = - E\left(f_{\Theta'}(\Omega(\omega' t)) | x, \Theta', \omega'\right) \end{equation}$

In this case we optimize our non-liearity $\Theta' \rightarrow \Theta$ and our frequencies $\omega' \rightarrow \omega$.

Here for such a pseudo-likelihood, we would use a softmax function over all samples $n$ and for all target dimensions $d$ to guarantee a distribution:

$\begin{equation}L\left(\Theta', \omega'\right) = \frac{\exp\left(\log L_{n, d}\left(\Theta', \omega'\right)\right)}
{\sum_{n=0}^{N-1} \sum_{\forall d} \exp\left(\log L_{n, d}\left(\Theta, \omega'\right)\right)} \end{equation}$



