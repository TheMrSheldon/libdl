\page technicalTensors Tensors
\tableofcontents

# Design Decisions
Tensors are at the heart of deep learning and one has to strike a good balance between performance, flexibility and ease of use when implementing them. In the context of C++, there are many design decisions going into how tensors could be implemented. For example:
 1. Should a tensor interface be defined explicitly (e.g., via an abstract base class) or should tensors be defined as types that fulfill general concepts?
 2. Who should own the tensor's memory? Should it be shared or uniquely owned?

With respect to *1.*, we ultimately decided on an abstract base class for general pointer implementations, `dl::TensorImpl`. And, to reduce the API's complexity, any concrete implementations of the base class are hidden behind device specific factories implementing `dl::Device`. Some key points in favor if this implementation are:
 - Ease of use -- the user only has to deal with a single datatype, `dl::TensorImpl` (or rather `dl::TensorPtr`) as outlined below. It is also semantically clear that all tensors are equal (no matter on which device they are stored)
 - Compile time -- templating all operations on tensors and defining tensors as a concept (as opposed to a type) could certainly be clean but would also result in long compile times
 - Control and integration -- we often need certain assumptions on the tensor operations to hold (e.g., about memory ownership). These may not be captured by concepts but can be expressed through API documentation and a common base class.

The correct choice for *2.* was a lot more tricky but after initially going with unique ownership, tensors now share their memory. The key question behind this is, what behavior is natural for the user and efficient? For example consider the following code:
```{cpp}
dl::Tensor a = {1, 2, 3};
dl::Tensor b = a;
b[0] = 4;
```
which value should `a` hold? *Unique ownership* of memory would dictate that, since `b` cannot "co-own" the memory used by `a`, it must have created a copy of `a`'s memory and in the third line only the copy is modified such that `a` still holds the value `{1, 2, 3}` by the end. We don't always want for `b` to copy the members of `a`. For example, the return-type of the subscript operator (`[]`) can't be `dl::Tensor` since it must reference the memory of `a`. The best way to solve this, would be an additional datatype, `dl::TensorView`:
```{cpp}
dl::Tensor a = {1, 2, 3};
dl::TensorView b = a;
b[0] = 4;
// a is now {4, 2, 3}
```
With *shared ownership*, however, `dl::Tensor b = a;` could mean both: creating a copy of `a` (it is the "copy constructor" after all) or simply adding a reference to the memory that `a` references as well. For libdl, we chose *shared ownership* and solved this ambiguity problem by naming the datatyoe `dl::TensorPtr`, which could generally be thought of as a `std::shared_ptr<dl::TensorImpl>` with some tensor-specific API. It should now be clear that in
```{cpp}
dl::TensorPtr a = {1, 2, 3};
dl::TensorPtr b = a;
```
`b` is a (reference counted) pointer to the same memory that `a` also points to. The copy constructor copies the pointer and not the memory. There certainly are good points for both shared and unique ownership. Unique ownership for example (in the context of `std::unique_ptr` vs `std::shared_ptr`) is more performant since no atomic reference counter is needed and allows/forces the user of the API to think more about how they manage their memory. E.g., the user has control over if they want to copy or move memory. In practice, however, this also means that the exact same function must be overloaded differently depending on memory ownership:
```{cpp}
dl::Tensor matmul(dl::Tensor& x, dl::Tensor& y) {
    // *Reference* both tensors in the computation graph
    // ...
}
dl::Tensor matmul(dl::Tensor&& x, dl::Tensor& y) {
    // *Move* x into and **reference** y in the computation graph
    // ...
}
dl::Tensor matmul(dl::Tensor& x, dl::Tensor&& y) {
    // *Reference* x in and **move** y into the computation graph
    // ...
}
dl::Tensor matmul(dl::Tensor&& x, dl::Tensor&& y) {
    // *Move* both tensors into the computation graph
    // ...
}
```
and we did not even mention that we also need to add overloads for `dl::TensorView` yet! Further consider the following case:
```{cpp}
{
    dl::Tensor tmp = somefunc();
    otherfunc(tmp, tmp);                        // Option 1
    otherfunc(std::move(tmp), tmp);             // Option 2
    otherfunc(tmp, std::move(tmp));             // Option 3
    otherfunc(std::move(tmp), std::move(tmp));  // Option 4
}
```
Option 2 and 3 clearly don't work since `tmp` is used after it was moved away. Option 3 does not work since the first parameter causes the computation graph to hold a *reference* to `tmp`, which is moved away right after. Option 1 does work but is not ideal since it needs to unnecessarily copy the tensor. This problem could be avoided by something like
```{cpp}
{
    dl::Tensor tmp = somefunc();
    // Move tmp into the computation graph and return a reference to it
    dl::Tensor& ref = dl::remember(std::move(tmp));
    otherfunc(ref, ref);
}
```
but simplicity should be key here.