\page basicRearrange Rearrange
\tableofcontents

The `dl::rearrange` function is a more versatile alternative to `dl::reshape`, `dl::transpose`, `dl::permute`, `dl::expand`, `dl::unsqueeze`, `dl::squeeze`, and `dl::chunk`, in that it combines all their functionality and remains more readable!

To achieve this, the rearrange operation is specified through a string as shown in the following example (our implementation of rearrange is heavily influed by einops \cite einops):

```{cpp}
dl::TensorPtr x = dl::random({2.0f, 3.0f, 4.0f});
dl::rearrange("a b c -> b 1 (a c)", x);
```

And all of this without additional overhead over calling `dl::reshape` et al. yourself because the rearrange specifier is evaluated **at compile time** (for now at runtime but compile time is technically possible :P).


# Syntax
Every rearrange statement takes the form `<leftexpr> -> <rightexpr>` where `leftexpr` and `rightexpr` are space-separated sequences of
- named dimensions `[A-Za-z]+`
- unnamed dimensions `_`
- constant dimensions `[0-9]+`
- at most one elipsis `...`
- chunk expression `/[0-9]+`
- merged dimension `(<expr>)` where `<expr>` is itself a space-separated sequence of named, constant dimensions, or ellipsis only


## Examples
1. Unsqueeze at -2: `... a -> ... 1 a`
2. Squeeze at -2: `... 1 a -> ... a`
3. Reshape from (12, 3) to (3, 4, 3): `a b -> a/3 b`
4. Transpose the last two dimensions: `... a b -> ... b a`
5. Repeat the 2nd dimension 10 times: `_ a -> _ (10 a)`
6. Merge all dimensions except the last: `... a -> (...) a`