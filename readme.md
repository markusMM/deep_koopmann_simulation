# Deep Koopman and HAVOK Optimization Toolkit

## What is Koopman and what shall one even care about such another useless cryptic python library?

Koopman analysis is looking into quasi-chaotic, non-linear system simulation and its propagation through time.

In other words, we could just forecast non-linear dynamic systems and describe it efficiently, thought its frequency behaviour, up a a specific degree of accuraccy, other classical systems couldn't!


## *Classical* Koopman Analysis

When looking into stadard Fourier Analysis, we describe a linear system between the actual oscillators and the desired output space, minimizing the squared error:

$$E\left(K, \omega\right) = \sum_{t=0}^{T-1} \left(x - K \Omega(\omega t) \right)^2$$

With $E$ being the actual error function between our ground truth $x$ and the oscillator $\Omega(\omega', t)$.


The oscillator $\Omega(\omega t)$ is defined in the same manner as in the Fourier wavelets:

$$
\Omega(\omega t) = \begin{pmatrix} 
    \sin(\omega_1 t) \\
    \vdots \\
    \sin(\omega_N t) \\
    \cos(\omega_1 t) \\
    \vdots \\
    \cos(\omega_N t)
\end{pmatrix}
$$

In Koopman Analysis, however, we look into any type of function $f_{\Theta}$, mostly non-linear or at least quasi-linear:

$$E\left(f_{\Theta'}(\Omega(\omega', t)) | x, \Theta', \omega'\right) = \sum_{t=0}^{T-1} \left(x - f_{\Theta}(\Omega(\omega' t), \Theta') \right)^2$$

Furthermore, from this error function, we can derive some type of pseudo $log$-likelihood, using any kind of error function for oscillators and periodic frequency elements:

$$\log L\left(\Theta', \omega'\right) = - E\left(f_{\Theta'}(\Omega(\omega' t)) | x, \Theta', \omega'\right)$$

In this case we optimize our non-liearity $\Theta' \rightarrow \Theta$ and our frequencies $\omega' \rightarrow \omega$.

Here for such a pseudo-likelihood, we would use a softmax function over all samples $n$ and for all target dimensions $d$ to guarantee a distribution:

$$L\left(\Theta', \omega'\right) = \frac{\exp\left(\log L_{n, d}\left(\Theta', \omega'\right)\right)}
{\sum_{n=0}^{N-1} \sum_{\forall d} \exp\left(\log L_{n, d}\left(\Theta, \omega'\right)\right)}$$


## Hankel Alternative View Of Koopman (HAVOK)

HAVOK aims to have an expressively non-linear view onto the recombination function within the Koopman framework. <br>
Thus, similar to the Deep Koopman, where we approximate our function with a Deep Neural Network, we here try to adapt our function given the Hankel criterium. <br>


### Koopman Operator $\mathcal{K}$

So, this whole Koopman process can be expressed in a simple Koopman operator which describes *finite*, *linear* transition from our dynamical *non-linear function $g(x_{k})\rightarrow{\mathcal{K}} g(x_{k+1})$*, as described below:

$\frac{d}{dt} x = f(x)$<br>
$\rightarrow \mathbf{F}_t(x(t_0)) = x(t_0+t) = x(t_0) + \int_{t_0}^{t_0+t} f(x(\tau))d\tau$<br>
$\rightarrow x_{k+1} = \mathbf{F}_t(x_k)$,    discrete time update
<br>

$\mathcal{K}_{t} g = g \circ \mathbf{F}_t$<br>
$\rightarrow \mathcal{K}_t g(X_k) = g(\mathbf{F}_t(x_k)) = g(x_{k+1})$<br>
$\rightarrow g(x_{k+1}) = \mathcal{K}_t g(X_k)$, discrete time update
<br>

Thus, if we would find a subspace for $g(x_k)$, where function $g(x_{k+1})$ stays within that subspace, we'd have an effective description of that system, at least for timestep $t_k \rightarrow t_{k+1}$.
<br>
$\mathbf{F}_t: x_k \rightarrow x_{k+1}$<br>
 $g_x: x_k \rightarrow y_k$<br>
$\mathcal{K}_t: y_k \rightarrow y_{k+1}$<br>

### Hankel Alternative View

Now to not sit in a cave for about eternity, trying to find such subspace $g$, one might look into more clever methods.<br>
Thus, we are using the Hankel matrix

$
\begin{bmatrix}
x(t_1) & x(t_2) & x(t_3) & \cdots & x(t_p) \\
x(t_2) & x(t_3) & x(t_4) & \cdots & x(t_{p+1}) \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
x(t_q) & x(t_{q+1}) & x(t_{q+2}) & \cdots & x(t_m)
\end{bmatrix}
$

Thus, we have DMD like setting, using the SVD of this matrix to directly have the Koopman-invariant measurement system on the attractor. ([Giannakis 2015](doi:10.48550/arXiv.1507.02338))

This diagram illustrates how the regression model connects to methods like Dynamic Mode Decomposition (DMD) ([Rowley 2009](https://doi.org/10.1017/S0022112009992059), [Schmid 2010](https://doi.org/10.1017/S0022112010001217), [Kutz 2016](https://doi.org/10.1137/16M1059396)) and Sparse Identification of Nonlinear Dynamics (SINDy) ([Brunton 2016](https://doi.org/10.1073/pnas.1517384113)). The linear part is captured in matrix $A$, while the bad fit or unmodeled nonlinear effects are symbolized as matrix $B$.

$$
\frac{d}{dt}
\begin{bmatrix}
v_1 \\
v_2 \\
v_3 \\
\vdots \\
v_r
\end{bmatrix}
=
\begin{bmatrix}
A & B \\
\text{- Bad -} & \text{- Fit -}
\end{bmatrix}
\begin{bmatrix}
v_1 \\
v_2 \\
v_3 \\
\vdots \\
v_r
\end{bmatrix}
$$


### Deep HAVOK

So, when looking into the aforementioned methology, whe can easily see that there is a specific range, HAVOK could hadle, finding the most optimal operator. <br>
When we want to have a better fit on $v_{\tau}(t+1)$ using any small number of previous $v_{\tau}(t) ... v_{\tau}(t-k)$, we need to further approximate it using Deep Learning, for instance,  as illustrated if the figure below ([Yang et al 2022](https://doi.org/10.3390/e24030408)): 

![illustation of ML in HVOK](https://cdn.ncbi.nlm.nih.gov/pmc/blobs/2775/8947207/8a98f3b48c2c/entropy-24-00408-g001.jpg)<br>
*Figure: Machine Learning filling the computational gab (bad fit) in HAVOK analysis. ([Yang et al 2022](https://doi.org/10.3390/e24030408))*

Here, we will try to generalize this finite evolution using many small time whindows of a given observation and even future predicted observations, given the HAVOK singular value decomposition.



