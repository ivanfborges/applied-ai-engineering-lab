# Senior Interview Questions: Vector Spaces, Bases, and Projections

## 1. What is the difference between span and basis?

The span is every linear combination obtainable from a set of vectors. A basis is a spanning set with no redundancy: its vectors are linearly independent. A spanning set can contain extra vectors; a basis cannot.

## 2. Why is a vector's representation unique in a basis?

Assume \(x=\sum_i a_i b_i=\sum_i c_i b_i\). Subtracting gives \(0=\sum_i(a_i-c_i)b_i\). Linear independence implies every \(a_i-c_i=0\), so the two coordinate representations are identical.

## 3. What is the difference between dimension and rank?

Dimension belongs to a vector space and equals the size of any basis. Rank belongs to a matrix and equals the dimension of its column space, equivalently its row space.

## 4. What does lower intrinsic dimension mean?

It means observations use many ambient coordinates but their meaningful variation can be described by fewer independent directions, at least approximately. For example, points stored in \(\mathbb{R}^{100}\) may lie near a 10-dimensional subspace or manifold.

## 5. Derive the projection of \(x\) onto a nonzero vector \(u\).

Write the projection as \(p=\alpha u\). Orthogonality of the residual requires \(u^\top(x-\alpha u)=0\), which gives

\[
\alpha=\frac{u^\top x}{u^\top u},
\qquad
\operatorname{proj}_u(x)=\frac{u^\top x}{u^\top u}u.
\]

If \(u\) has unit norm, the formula is \((u^\top x)u\).

## 6. Why is \(QQ^\top\) an orthogonal projection matrix?

For orthonormal columns, \(Q^\top Q=I\). Thus \(P=QQ^\top\) is symmetric because \(P^\top=P\), and idempotent because \(P^2=Q(Q^\top Q)Q^\top=QQ^\top=P\). Its range is the column space of \(Q\), and the residual is orthogonal to that space.

## 7. How is linear regression a projection problem?

OLS chooses \(X\hat\beta\) in the column space of \(X\) that minimizes \(\lVert y-X\beta\rVert_2^2\). Therefore, fitted values are the orthogonal projection of \(y\) onto that column space, and the residual satisfies \(X^\top(y-X\hat\beta)=0\).

## 8. Why should you avoid explicitly computing \((X^\top X)^{-1}\)?

It is less stable and often less efficient than solving the least-squares problem directly. Forming \(X^\top X\) squares the condition number and can magnify numerical error. Prefer QR, SVD, `numpy.linalg.lstsq`, or an appropriate tested solver.

## 9. What happens when the basis columns are rank deficient?

Coordinates are not unique because redundant columns describe the same directions. \(A^\top A\) is singular, so the inverse-based formula fails. A pseudoinverse or SVD-based least-squares solver can return a minimum-norm solution, while the projected point remains well-defined for the column space.

## 10. Why can classical Gram-Schmidt be numerically unstable?

With nearly dependent inputs, subtracting almost equal floating-point components causes cancellation. Rounding errors can accumulate and the resulting columns may lose orthogonality. Modified Gram-Schmidt improves the procedure, while Householder QR and SVD are generally stronger library choices.

## 11. How is PCA related to projection?

After centering the data, PCA learns orthonormal directions that maximize retained variance. If \(W_k\) contains the first \(k\) directions, \(XW_k\) gives reduced coordinates and \(XW_kW_k^\top\) reconstructs the projection in the principal subspace. Equivalently, PCA minimizes squared reconstruction error among \(k\)-dimensional linear subspaces.

## 12. Why do embeddings work as vectors?

An embedding model maps an object to \(\mathbb{R}^d\). Its training objective makes selected geometric relations—distance, angle, inner product, or neighborhood—useful for the task. The array type alone does not create semantics; the learned mapping and objective do.

## 13. Can embeddings from different models be compared if their dimensions match?

Usually not. Equal dimension only makes the array shapes compatible. Coordinates from independently trained models refer to different learned spaces unless the models or an explicit alignment procedure place them in a shared representation.

## 14. When are cosine similarity and Euclidean distance equivalent for ranking?

For L2-normalized vectors, \(\lVert x-y\rVert_2^2=2-2x^\top y\), and the dot product equals cosine similarity. Their rankings are then equivalent. Without normalization, vector magnitude affects Euclidean distance and dot product differently.

## 15. How would you evaluate embedding dimensionality reduction in a RAG system?

Measure efficiency outcomes such as index size, memory, retrieval latency, ingestion time, and cost, together with quality outcomes such as recall@k, MRR, nDCG, relevant-context coverage, and downstream answer correctness. Fit on representative training data, compare against an uncompressed baseline, and test relevant slices and distribution shifts.

## 16. Why is explained variance insufficient for evaluating embedding compression?

PCA optimizes global variance retention and reconstruction error, not semantic relevance. A low-variance direction may contain important task signal, so high explained variance can coexist with damaged nearest-neighbor rankings or downstream quality.

## 17. What must be versioned when projecting production embeddings?

Version the embedding model, input preprocessing, normalization, projection parameters, similarity configuration, and vector index. Apply a compatible pipeline to documents and queries; otherwise they no longer inhabit the same operational representation space.

## 18. Does removing one direction remove a semantic concept from an embedding?

Not necessarily. It removes only the component aligned with that linear direction. The concept may be distributed across other directions, represented nonlinearly, or recoverable through correlated information. Validate the desired effect and collateral quality loss empirically.

## 19. Compare PCA and random projection.

PCA learns data-dependent directions that maximize variance and must be fitted and versioned. Random projection is cheaper to construct, does not optimize variance, and can approximately preserve pairwise distances with enough target dimensions. Choose according to reconstruction, task quality, scalability, and operational constraints.

## 20. How would you explain embedding dimensions to a nontechnical stakeholder?

They are coordinates in a map learned by a model. Individual coordinates usually have no simple standalone meaning; the full pattern places related content near each other. More coordinates can provide capacity but also increase storage and computation.
