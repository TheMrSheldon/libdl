\page basicArithmetics Arithmetics
\tableofcontents

# What is the difference between const Tensor&, Tensor&, Tensor&&?
A tensor can always only belong to exactly one object. That means that if we look at the following code snipped, where a
tensor is simply assigned to another tensor, we will observe that it gets **copied**.
```cpp
dl::Tensor tensora = {1, 2, 3, 4};
dl::Tensor tensorb = tensora;
tensorb[2] = 42;
std::cout << tensora << std::endl;
// {1, 2, 3, 4}
std::cout << tensorb << std::endl;
// {1, 42, 3, 4}
```
But quite often we do not want to copy the tensor but merely **reference** it. This is, where the C++ references come
into play. A reference to a pointer is described by the type `Tensor&`, such that above code would look like this:
```cpp
dl::Tensor tensora = {1, 2, 3, 4};
dl::Tensor& tensorb = tensora;
tensorb[2] = 42;
std::cout << tensora << std::endl;
// {1, 42, 3, 4}
std::cout << tensorb << std::endl;
// {1, 42, 3, 4}
```
\attention `tensorb` only *references* the memory used by `tensora`. The reference should not live longer than the memory it is referencing. To illustrate, can you spot the problem with the following code snippet?
```cpp
class MyClass {
private:
    dl::Tensor& tensor;
public:
    MyClass(dl::Tensor tensor) : tensor(tensor) {}
};
```
`MyClass` stores a reference to its constructor argument. Since the constructor argument is destroyed after the
constructor exits, we reference deleted memory.

One can express it like this: each tensor *uniquely* owns their memory. If a tensor (not a reference to a tensor!) gets
destroyed, it destroys its memory. The alternative would be *shared* ownership, where multiple tensors can own the same
memory and the memory is only destroyed after all tensors are destroyed that reference this memory. This is usually
implemented using an atomic (i.e., thread safe) reference counter (see, e.g., std::shared_ptr). However, in the design
of this library we went for unique ownership to remove may time-costly atomic increments and decrements. If you need
shared ownership, you could however always use `std::shared_ptr<dl::Tensor>`.

Finally, you may have noticed that arithmetic functions usually come with multiple overloads. For example for `dl::pow`
you can choose between
```cpp
dl::Tensor pow(const dl::Tensor& base, float exponent); // 1)
dl::Tensor pow(dl::Tensor& base, float exponent);       // 2)
dl::Tensor pow(dl::Tensor&& base, float exponent);      // 3)
```
but what is the difference between them and which one to use? Option 2) is straight forward: to avoid copying the
tensor, we only pass a reference to it to the function. Option 1) generally does the same thing, but is not allowed to
modify the tensor. You may wonder why dl::pow needs to modify its input tensors at all but consider this: If `base`
required a gradient, a reference of it would be *stored for the backwards pass* in the computation graph and in the
backwards pass we need to modify `base` to update its gradient. The first option is not allowed to modify `base` and
thus won't be allowed to update the gradient for `base`. As such, be sure that `base` does not need a gradient if you
use option 1). Option 3) on the other hand is a bit more interesting since it uses "move semantics". Concretely, this
means that ownership of memory is moved from one object to another:
```cpp
dl::Tensor tensora = {1, 2, 3, 4};
dl::Tensor tensorb = std::move(tensora);
std::cout << tensora << std::endl;
// null
std::cout << tensorb << std::endl;
// {1, 2, 3, 4}
```
It follows that option 2) and 3) only differentiate in **who owns the memory**. In option 2), the computation graph
holds a reference to the tensor that should be updated, wheras in option 3) the computation graph *owns* the tensor. So
which of these two options is correct?
```cpp
// Compute (2x)²
dl::Tensor funcA(dl::Tensor& x) {
    dl::Tensor b = 2*x;
    return dl::pow(b, 2);
}
dl::Tensor funcB(dl::Tensor& x) {
    dl::Tensor b = 2*x;
    return dl::pow(std::move(b), 2);
}
```
Answer: `funcB` is correct since the memory used by `b` is moved into the computation graph. In `funcA` the memory used
by `b` will be destroyed after the function returns and the computation graph has a dangling reference.

To conclude (this of course holds for all other gradient enabled functions as well):
 - Use dl::pow(const dl::Tensor& base, float exponent) if no gradient should be calculated (even if
    `base.requireGrad()` was true). This is useful to allow compiler optimizations for faster **inference**.
 - Use dl::pow(dl::Tensor& base, float exponent) if a gradient may be calculated and you can ensure that the tensor
    you give a reference to in the function call is alive long enough. This is useful for **model-parameters**.
 - Use dl::pow(dl::Tensor&& base, float exponent) if a gradient may be calculated but you can not (or do not want to)
    ensure that the data lives long enough. This is useful for **temporary variables**.


## Small Exercise
Can you spot the problem with the following code?
```cpp
dl::Tensor a = {1, 2, 3, 4};
a->setRequiresGrad(true);
dl::Tensor b = dl::pow(a, 2);
dl::Tensor c = dl::pow(std::move(a), 4);
dl::Tensor result = b + c; // a²+a⁴
result.backward();
```