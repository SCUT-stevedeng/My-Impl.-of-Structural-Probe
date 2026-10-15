# Annotations  
(2026/10/4 14:48:54)

“Tree structure is embedded if the transformed space has the property that squared L2 distance between two words’ vectors corresponds to the number of edges between the words in the parse tree.” ([Hewitt and Manning, p. 1](zotero://select/library/items/A65ATJ95)) ([pdf](zotero://open-pdf/library/items/7S43QUGM?page=1&annotation=CLIL359D)) The paper hypothesizes that if word embeddings encode dependency-tree distances, the number of edges between two words in the dependency tree corresponds to the squared L2 distance between their transformed embeddings.

“o reconstruct edge directions, we hypothesize a linear transformation under which the squared L2 norm corresponds to the depth of the word in the parse tree.” ([Hewitt and Manning, p. 1](zotero://select/library/items/A65ATJ95)) ([pdf](zotero://open-pdf/library/items/7S43QUGM?page=1&annotation=2SE8MHYX)) The paper further hypothesizes that the squared L2 norm of a transformed word embedding corresponds to its distance from the ROOT, i.e., its depth in the dependency tree.

“Intuitively, why do parse tree distances and depths matter to syntax? The distance metric—the path length between each pair of words—recovers the tree T simply by identifying that nodes u, v with distance dT (u, v) = 1 are neighbors. The node with greater norm—depth in the tree—is the child.” ([Hewitt and Manning, p. 2](zotero://select/library/items/A65ATJ95)) ([pdf](zotero://open-pdf/library/items/7S43QUGM?page=2&annotation=WQWRFT7P)) If two word embeddings transformed by the probe have a tree distance of 1, the corresponding words are neighbors in the tree (i.e., they have a parent–child relationship).  
The word with the larger squared L2 norm is the child (since the authors hypothesize that this quantity corresponds to the word’s distance from the ROOT).

“For example, the ability to perform the classic hierarchy test of subject-verb number agreeement (Linzen et al., 2016) in the presence of “attractors” can be explained as the verb (V) being closer in the tree to its subject (S) than to any of the attactor nouns:” ([Hewitt and Manning, p. 2](zotero://select/library/items/A65ATJ95)) ([pdf](zotero://open-pdf/library/items/7S43QUGM?page=2&annotation=27IND59F)) This refers to subject–verb agreement.  
For example: The key to the cabinets is ...  
Here, the distance between “key” and “is” in the parse tree is shorter than that between “cabinets” and “is.”

“Intuitively, if a neural network embeds parse trees, it likely will not use its entire representation space to do so, since it needs to encode many kinds of information. Our probe learns a linear transformation of a word representation space such that the transformed space embeds parse trees across all sentences. This can be interpreted as finding the part of the representation space that is used to encode syntax; equivalently, it is finding the distance on the original space that best fits the tree metrics.” ([Hewitt and Manning, p. 2](zotero://select/library/items/A65ATJ95)) ([pdf](zotero://open-pdf/library/items/7S43QUGM?page=2&annotation=V85J4DNX)) The assumption here is that syntactic information is encoded in only a low-rank subspace of the original word representation space.

“hT Ah,” ([Hewitt and Manning, p. 2](zotero://select/library/items/A65ATJ95)) ([pdf](zotero://open-pdf/library/items/7S43QUGM?page=2&annotation=5XYRNZGE)) For the standard Euclidean inner product, A is the identity matrix, so the expression reduces to hᵀh.  
For a generalized inner product, A is a symmetric positive-definite matrix.  
Since vector norms and distances can be induced by inner products, standard and generalized inner products define different norms and distances, thereby capturing different geometric properties.  
In this paper, however, A is required to be symmetric and positive semidefinite. The reason is discussed in the notes below.

“we can view this as specifying a linear transformation B ∈ Rk×m, such that A = BT B. The inner product is then (Bh)T (Bh), the norm of h once transformed by B.” ([Hewitt and Manning, p. 2](zotero://select/library/items/A65ATJ95)) ([pdf](zotero://open-pdf/library/items/7S43QUGM?page=2&annotation=3FMVTELQ)) Any symmetric positive-semidefinite matrix admits such a factorization. This allows a quadratic form of the type hᵀAh to be rewritten as the squared Euclidean norm ‖Bh‖².  
In other words, “A is symmetric and positive semidefinite” ⇔ “A can be factored as BᵀB.” This reparameterizes the constraint on A in terms of the product of an unconstrained matrix B and its transpose.

“dB (h\` i , h\` j)2 = (B(h\` i − h\` j))T (B(h\` i − h\` j ))” ([Hewitt and Manning, p. 2](zotero://select/library/items/A65ATJ95)) ([pdf](zotero://open-pdf/library/items/7S43QUGM?page=2&annotation=A5WFLT65)) If A is symmetric and positive semidefinite, the induced distance and norm-like quantity are the Mahalanobis distance and Mahalanobis (semi)norm, respectively.  
The symmetry of A ensures that dist(a, b) = dist(b, a).  
The positive semidefiniteness of A ensures that dist(a, b) ≥ 0.  
Requiring the metric matrix A to be symmetric and positive semidefinite is one of the most common constraints in metric learning.

“mBin ∑ \` 1 |s\`|2 ∑ i,j ∣ ∣dT \` (w\` i , w\` j) − dB(h\` i , h\` j )2∣” ([Hewitt and Manning, p. 2](zotero://select/library/items/A65ATJ95)) ([pdf](zotero://open-pdf/library/items/7S43QUGM?page=2&annotation=ZWUBMF6C)) The squared Mahalanobis distance is used to approximate the dependency-tree distance.  
The reason is explained in footnote 2.

“|s\`|” ([Hewitt and Manning, p. 2](zotero://select/library/items/A65ATJ95)) ([pdf](zotero://open-pdf/library/items/7S43QUGM?page=2&annotation=NGDEVPDH)) The square of the sentence length is on the same order as the number of word pairs, C(n, 2), but squaring the sentence length is easier to compute.

“distances are guaranteed nonnegative and symmetric,” ([Hewitt and Manning, p. 2](zotero://select/library/items/A65ATJ95)) ([pdf](zotero://open-pdf/library/items/7S43QUGM?page=2&annotation=EY34FHES))

“an inner product” ([Hewitt and Manning, p. 2](zotero://select/library/items/A65ATJ95)) ([pdf](zotero://open-pdf/library/items/7S43QUGM?page=2&annotation=PPNXQI4I)) That is, an inner product-like form defined by a symmetric, positive-semidefinite matrix A.

“encodes syntax tree distance” ([Hewitt and Manning, p. 2](zotero://select/library/items/A65ATJ95)) ([pdf](zotero://open-pdf/library/items/7S43QUGM?page=2&annotation=BAMGJL6V))

“We replace the vector distance function dB(hi, hj) with the squared vector norm ‖hi‖2 B, replacing Equation 1 with ‖hi‖A = (Bhi)T (Bhi) and training B to recreate ‖wi‖.” ([Hewitt and Manning, p. 3](zotero://select/library/items/A65ATJ95)) ([pdf](zotero://open-pdf/library/items/7S43QUGM?page=3&annotation=EX6VNWRT)) Here, the squared Mahalanobis (semi)norm is defined as (Bhᵢ)ᵀ(Bhᵢ), and it is used to approximate the word’s depth in the dependency tree.

“i.e., B is square.” ([Hewitt and Manning, p. 3](zotero://select/library/items/A65ATJ95)) ([pdf](zotero://open-pdf/library/items/7S43QUGM?page=3&annotation=PFPUFZTT)) I have a question: The authors define an inner product of the form xᵀAy, where A is a square matrix whose dimensions equal the model’s embedding dimension. They then factor A into BᵀB, but here B is also required to be square, so both dimensions of B must equal the embedding dimension. If B does not explicitly reduce the dimensionality of the word embeddings, how can we investigate whether the encoded information lies in a low-rank subspace?  
My tentative answer: Perhaps we do not need to enforce low rank through explicit dimensionality reduction. If the information genuinely lies in a low-rank subspace, would examining the rank of B reveal that low-rank structure naturally?