\page technicalAutodiff Automatic Differentiation (Autodiff)
\tableofcontents

Optimization in the context of deep learning often means finding a local minimum of the loss function via some form of
optimization routine, which usually needs to automatically compute the derivative of the model's output function (e.g.
gradient descend needs the first order derivative and the Newton method even the second order). Thus, the support for
automatic differentiation (autodiff), i.e., evaluating the gradient at a desired coordinate without manually calculating
the derivative, is central to any deep learning library that actually wants to support the **learning** part.


# How it works
## Mathematical Background
\note This section is largely based on an awesome survey by Baydin et al. \cite autodiff_survey .
### Forward Mode
In a twist of sheer elegance, consider the following observation. For \f(a, b\in\mathbb R\f), we call
\f(z=a+b\varepsilon\f) a *dual number*, where \f(\varepsilon \neq 0\f) is a new symbol defined such that
\f(\varepsilon^2 = 0\f). We will now use dual numbers to calculate **both**, the evaluation of the function and the
evaluation of its derivative. First, observe that for arbitary dual numbers
\f(u+\dot u\varepsilon, v+\dot v\varepsilon\f) it holds that
\f{align}{
    (u+\dot u\varepsilon) + (v+\dot v\varepsilon) &= (u+v) + (\dot u + \dot v)\varepsilon, \text{and}\\
    (u+\dot u\varepsilon) \cdot (v+\dot v\varepsilon) &= uv + (u\dot v + \dot uv)\varepsilon.
\f}
Notice how this expresses the linearity of differentiation and the product rule, if we interpret \f(\dot u\f) and
\f(\dot v\f) as the derivatives at point \f(u\f) and \f(v\f) respectively. Now let \f(f\colon \mathbb{R}\to\mathbb{R}\f)
be a differentiable function. We extend it into the dual numbers via
\f[f(x + \dot x \varepsilon) := f(x) + f'(x)\dot x\varepsilon.\f]
Note that this definition makes sense with our previous interpretation. Further, we can now observe that the chain rule
also holds:
\f{align}{
    f(g(x + \dot x\varepsilon) &= f(g(x) + g'(x)\dot x\varepsilon))\\
        &= f(g(x)) + f'(g(x))g'(x)\varepsilon.
\f}
Given that this actually holds in general (which it does), we can now simultaneously evaluate \f(f(x)\f) and \f(f'(x)\f)
by calculating \f(f(x+1\varepsilon)\f) and taking the real part for \f(f(x)\f) and the coefficient of \f(\varepsilon\f)
is the result of \f(f'(x)\f).

But what about functions \f(f\colon \mathbb{R}^n \to \mathbb{R}\f) with more inputs? Since the forward pass can only
calculate the derivative in one direction per pass, we need \f(n\f) passes to compute the gradient \f(\nabla f\f), which
is quite inefficient and, where the *reverse mode* automatic differentiation shines.

### Reverse Mode
Consider the composite function \f(h := f\circ g\f), the *chain rule* states that the derivative \f(h\f) at point
\f(x\f) can be calculated via
\f[\left.\frac{\partial h}{\partial x}\right|_{x} = \left.\frac{\partial f}{\partial u}\right|_{u=g(x)} \cdot \left.\frac{\partial u}{\partial x}\right|_{x}.\f]
This is crucial since it allows us to compute the gradient at a single point without first calculating the first order
derivative over all points (as would be done with symbolic differentiation). Instead, we have to overload the semantics
of each function to calculate the value (as it usually would) and additionally store the computation information
somewhere to allow computing the gradient in the **reverse** order of the functions. That is, in the example above, the
forward pass would first compute \f(y = f(x)\f), then \f(z = g(y)\f) and then go backwards and compute the gradients
\f(\frac{\partial z}{\partial y}\f) and then \f(\frac{\partial y}{\partial x}\f).

This way, we can compute all derivatives at the same time (and thus are **more efficient** than forward mode) but need to
store the intermediate results and the computation order for the backwards pass (and thus are **less performant** than
forward mode).

## Implementation in libdl      {#autodiffimpl}
\todo WIP


# Adding Autodiff Support to a Function
This example with equip the logarithm, `dl::log(dl::TensorPtr x)` with support for automatic differentiation. We will
start of with the basic stub
```{cpp}
TensorPtr dl::log(TensorPtr x) noexcept {
    auto tensor = x->log();
    if (tensor->requiresGrad()) {
	    /* implement derivative here */
    }
	return tensor;
}
```
Remember from the [description above](@ref autodiffimpl) that, to enable a function for the backwards pass, it only
needs to register a callback for calculating the derivative to `dl::TensorImpl::gradfn`.
```{cpp}
tensor->grad = [x = std::move(x)](dl::TensorPtr agg) {
    // Compute the gradient
    auto grad = 1/x;
    // Update the gradient of x
    x->grad = (x->grad == nullptr)? grad : (x->grad + grad);
    // Compute the gradients for the value that x depends on (go one step further back in the chain rule) or, if x is
    // the last element in the chain (i.e., it has no gradfn to go back further), it must require a gradient itself
    if (x->gradfn)
        x->gradfn(grad);
    else
        assert(x->requiresGrad());
};
```
\todo WIP

## Functions with Multiple Inputs
\todo WIP

## Adding Autodiff to a Function that Composites autodiff enabled functions
\todo This is not currently supported


# Gradients
## Gradients of Matrix Operations
### Matrix Product
\todo WIP