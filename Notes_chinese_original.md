# 注释  
(2026/10/4 14:48:54)

“Tree structure is embedded if the transformed space has the property that squared L2 distance between two words’ vectors corresponds to the number of edges between the words in the parse tree.” ([Hewitt和Manning, p. 1](zotero://select/library/items/A65ATJ95)) ([pdf](zotero://open-pdf/library/items/7S43QUGM?page=1&annotation=CLIL359D)) 本文假设如果word embedding编码了依存距离树，那么两个词在依存树中边的数量等同于两个词对应embedding的L2距离

“o reconstruct edge directions, we hypothesize a linear transformation under which the squared L2 norm corresponds to the depth of the word in the parse tree.” ([Hewitt和Manning, p. 1](zotero://select/library/items/A65ATJ95)) ([pdf](zotero://open-pdf/library/items/7S43QUGM?page=1&annotation=2SE8MHYX)) 此外，本文还假设word embedding自身的L2范数就是其到ROOT的距离，即它在依存树中的深度

“Intuitively, why do parse tree distances and depths matter to syntax? The distance metric—the path length between each pair of words—recovers the tree T simply by identifying that nodes u, v with distance dT (u, v) = 1 are neighbors. The node with greater norm—depth in the tree—is the child.” ([Hewitt和Manning, p. 2](zotero://select/library/items/A65ATJ95)) ([pdf](zotero://open-pdf/library/items/7S43QUGM?page=2&annotation=WQWRFT7P)) 若两个经过probe变换的word embedding的距离为1，则他们是parent-child（即neighbour）。  
而那个具有更大L2范数的词即为child（作者假设L2范数约等于这个词到ROOT的距离）

“For example, the ability to perform the classic hierarchy test of subject-verb number agreeement (Linzen et al., 2016) in the presence of “attractors” can be explained as the verb (V) being closer in the tree to its subject (S) than to any of the attactor nouns:” ([Hewitt和Manning, p. 2](zotero://select/library/items/A65ATJ95)) ([pdf](zotero://open-pdf/library/items/7S43QUGM?page=2&annotation=27IND59F)) 即主谓一致  
比如: The key to the cabinets is....  
这里key到is的距离比cabinets到is的距离更近

“Intuitively, if a neural network embeds parse trees, it likely will not use its entire representation space to do so, since it needs to encode many kinds of information. Our probe learns a linear transformation of a word representation space such that the transformed space embeds parse trees across all sentences. This can be interpreted as finding the part of the representation space that is used to encode syntax; equivalently, it is finding the distance on the original space that best fits the tree metrics.” ([Hewitt和Manning, p. 2](zotero://select/library/items/A65ATJ95)) ([pdf](zotero://open-pdf/library/items/7S43QUGM?page=2&annotation=V85J4DNX)) 这里假设编码syntax信息的只占原word representation的一个low rank sub-space

“hT Ah,” ([Hewitt和Manning, p. 2](zotero://select/library/items/A65ATJ95)) ([pdf](zotero://open-pdf/library/items/7S43QUGM?page=2&annotation=5XYRNZGE)) 狭义的向量内积中间的A就是单位矩阵，即等效于hTh。  
广义的向量内积，中间的A是个对称正定矩阵。  
由于向量的范数和距离可以由内积诱导得到，因此标准内积和广义内积可以定义出不同的范数与距离，并体现出不同的几何性质。  
不过这里本文要求A是正半定对称矩阵，原因在后面的笔记讲解

“we can view this as specifying a linear transformation B ∈ Rk×m, such that A = BT B. The inner product is then (Bh)T (Bh), the norm of h once transformed by B.” ([Hewitt和Manning, p. 2](zotero://select/library/items/A65ATJ95)) ([pdf](zotero://open-pdf/library/items/7S43QUGM?page=2&annotation=3FMVTELQ)) 对称半正定矩阵一定可以这么分解。这样就能把形如h^TAh的内积，转换为一个普通的L2范数的平方||Bx||^2  
即"A是对称半正定矩阵"<=>"A可分解为B^TB"，这样，就将"A是对称半正定矩阵"的约束分解为了一般矩阵B的乘积

“dB (h\` i , h\` j)2 = (B(h\` i − h\` j))T (B(h\` i − h\` j ))” ([Hewitt和Manning, p. 2](zotero://select/library/items/A65ATJ95)) ([pdf](zotero://open-pdf/library/items/7S43QUGM?page=2&annotation=A5WFLT65)) 若A对称且半正定，基于此定义内积的距离和范数分别是Mahalanobis距离和范数。  
A对称 说明dist(a,b) = dist(b,a)  
A半正定 说明 dist(a,b)>=0  
度量矩阵A对称且半正定度量学习最常用的度量矩阵约束。

“mBin ∑ \` 1 |s\`|2 ∑ i,j ∣ ∣dT \` (w\` i , w\` j) − dB(h\` i , h\` j )2∣” ([Hewitt和Manning, p. 2](zotero://select/library/items/A65ATJ95)) ([pdf](zotero://open-pdf/library/items/7S43QUGM?page=2&annotation=ZWUBMF6C)) 用平方Mahalanobis距离来approximate依存距离  
原因在脚注2

“|s\`|” ([Hewitt和Manning, p. 2](zotero://select/library/items/A65ATJ95)) ([pdf](zotero://open-pdf/library/items/7S43QUGM?page=2&annotation=NGDEVPDH)) 句子长度的平方约等于句子词对数量C_n^2，但是句子长度平方更好算。

“distances are guaranteed nonnegative and symmetric,” ([Hewitt和Manning, p. 2](zotero://select/library/items/A65ATJ95)) ([pdf](zotero://open-pdf/library/items/7S43QUGM?page=2&annotation=EY34FHES))

“an inner product” ([Hewitt和Manning, p. 2](zotero://select/library/items/A65ATJ95)) ([pdf](zotero://open-pdf/library/items/7S43QUGM?page=2&annotation=PPNXQI4I)) 即满足A对称且半正定的内积

“encodes syntax tree distance” ([Hewitt和Manning, p. 2](zotero://select/library/items/A65ATJ95)) ([pdf](zotero://open-pdf/library/items/7S43QUGM?page=2&annotation=BAMGJL6V))

“We replace the vector distance function dB(hi, hj) with the squared vector norm ‖hi‖2 B, replacing Equation 1 with ‖hi‖A = (Bhi)T (Bhi) and training B to recreate ‖wi‖.” ([Hewitt和Manning, p. 3](zotero://select/library/items/A65ATJ95)) ([pdf](zotero://open-pdf/library/items/7S43QUGM?page=3&annotation=EX6VNWRT)) 这里给出了Mahalanobis Norm的定义：(Bh_i)^T(Bh_i)，并用它来逼近这个词在依存树中的深度

“i.e., B is square.” ([Hewitt和Manning, p. 3](zotero://select/library/items/A65ATJ95)) ([pdf](zotero://open-pdf/library/items/7S43QUGM?page=3&annotation=PFPUFZTT)) 我有个问题：作者用的x^Ay内积定义，那么A是一个方阵，宽高等于模型embedding dim，然后将A分解为B转置和B的乘积，但这里要求B也是方阵，那B的宽高也只能等于embedding dim。既然B不能把word embedding的维度降低，那还怎么研究其encoded的信息在不在低秩空间。  
我想到的回答：就是说可以不强制降维来限制低秩，如果信息确实在low rank sub-space，直接计算B的秩自然会得到low rank?