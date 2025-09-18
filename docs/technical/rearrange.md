\page technicalRearrange Rearrange
\tableofcontents

\note This page describes the technical side of rearrange. If you would like to read a user guide, please go to the page \ref basicRearrange .

# Parsing

The parsing may be split into few distinct steps:
1. Split the specification at `->`, we call these substrings `left` and `right`
2. For `left` and `right`, determine the order of named dimensions and their signed indices
3. If the index lists do not match (the order is changed between `left` and `right`)

\paragraph Example
Lets walk the algorithm above through with this complex specifier: `a ... b 1 c -> (b a)/2 1 ... (10 c)`.
1. Splitting the specification yields `a ... b 1 c ` for `left` and ` (b a)/2 1 ... (10 c)` for `right`
2. The order of named dimensions in `left` is `a b c` and `b a c` for `right` with indices (listed in order `a b c`) `0 -2 -1` and `1 0 -1` respectively.
3. Squeeze dimensions labeled `1`. At this point, the tensor has shape `a ... b c`
4. Since the index lists are different for `left` and `right`, we `dl::permute` the tensor. At this point, the tensor has shape `b a ... c`
5. Apply unsqueezes. At this point, the tensor has shape `b a 1 ... 10 c`
6. Apply reshape. At this point, the tensor has shape `(b a) 1 ... (10 c)`
7. Apply chunking. At this point, the tensor has shape `(b a)/2 1 ... (10 c)`

\paragraph rearrangecases Hard cases
1. E.g., `a b c -> a b 1 c` should both produce the same index lists since no permutation necessary. Similarly for `a b 1 c -> a b c` or `a b (10 c)`

\paragraph rearrangerot Rules of thumb
1. Only named dimensions can be matched. E.g., `10 42 -> 42 10` is invalid.