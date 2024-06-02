\page technicalTransformer Transformer
\tableofcontents

\todo WIP

\f[
    \begin{bmatrix}0 & 1 & 2\\3 & 4 & 5\\6 & 7 & 8\\9 & 10 & 11\end{bmatrix}
    \overset{\text{reshape}(2, 2, 3)}{\Rightarrow}
    \begin{bmatrix}\begin{bmatrix}0 & 1 & 2\\3 & 4 & 5\end{bmatrix} & \begin{bmatrix}6 & 7 & 8\\9 & 10 & 11\end{bmatrix}\end{bmatrix}
\f]
\f{eqnarray*}{
    \begin{bmatrix}\begin{bmatrix}0 & 1 & 2\\3 & 4 & 5\end{bmatrix} & \begin{bmatrix}6 & 7 & 8\\9 & 10 & 11\end{bmatrix}\end{bmatrix}
    &\overset{\text{transpose}(0, 1)}{\Rightarrow}&
    \begin{bmatrix}\begin{bmatrix}0 & 1 & 2\\3 & 4 & 5\end{bmatrix} & \begin{bmatrix}6 & 7 & 8\\9 & 10 & 11\end{bmatrix}\end{bmatrix}\\
    &\overset{\text{transpose}(0, 2)}{\Rightarrow}&
    \begin{bmatrix}\begin{bmatrix}0 & 6\\3 & 9\\\end{bmatrix} & \begin{bmatrix}1 & 7\\4 & 10\\\end{bmatrix} & \begin{bmatrix}2 & 8\\5 & 11\\\end{bmatrix}\end{bmatrix}\\
    &\overset{\text{transpose}(1, 2)}{\Rightarrow}&
    \begin{bmatrix}\begin{bmatrix}0 & 3\\ 1 & 4\\2 & 5\end{bmatrix} & \begin{bmatrix}6 & 9\\7 & 10\\ 8 & 11\end{bmatrix}\end{bmatrix}
\f}



# Background and Implementation

\f[\text{MHA}(Q, K, V) = \text{Concat}(\text{Attention}(QW_1^Q+b_1^Q, KW_1^K+b_1^K, VW_1^V+b_1^V), \dots)\cdot W^O + b^O.\f]

\note The original proposal for transformers does not include the bias terms \f(b_i^Q, b_i^K, b_i^V\f), and \f(b^O\f).
We mention them here anyway since some transformer architectures (most notably BERT) added these in.

\f[\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.\f]

## Optimization
\todo WIP

# Architectures
## BERT
\todo WIP

# Tokenization
## WordPiece
\todo WIP