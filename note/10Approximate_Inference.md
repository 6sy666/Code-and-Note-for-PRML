# 10 近似推断
在概率模型的应用中，一个中心任务是在给定观测（可见）数据变量$X$的条件下，计算潜在变量$Z$的后验概率分布$p(Z | X)$，以及计算关于这个概率分布的期望。
对于实际应用中的许多模型来说，计算后验概率分布或者计算关于这个后验概率分布的期望是不可行的。这可能是由于潜在空间的维度太高，以至于无法直接计算，或者由于后验概率分布的形式特别复杂，从而期望无法解析地计算。在连续变量的情形中，需要求解的积分可能没有解析解，而空间的维度和被积函数的复杂度可能使得数值积分变得不可行。对于离散变量，求边缘概率的过程涉及到对隐含变量的所有可能的配置进行求和。这个过程虽然原则上总是可以计算的，但是我们在实际应用中经常发现，隐含状态的数量可能有指数多个，从而精确的计算所需的代价过高。
根据近似方法依赖于随机近似还是确定近似，方法大体分为两大类。
## 10.1 变分推断
变分方法本质上没有任何近似的东西，但是它们通常会被用于寻找近似解。寻找近似解的过程可以这样完成：限制需要最优化算法搜索的函数的范围，例如只考虑二次函数，或者考虑由固定的基函数线性组合而成的函数，其中只有线性组合的系数可以发生变化。
假设我们有一个纯粹的贝叶斯模型，其中每个参数都有一个先验概率分布。这个模型也可以有潜在变量以及参数，我们会把所有潜在变量和参数组成的集合记作$Z$。类似地，我们会把所有观测变量的集合记作$X$.对于N个独立同分布的数据，其中$X=\{x_1,\dotsb,x_N\},Z=\{z_1,\dotsb,z_N\}$.概率模型确定了联合概率分布$p(X, Z)$，我们的目标是找到对后验概率分布$p(Z | X)$以及模型证据$p(X)$的近似。
可以将对数边缘概率分解，得到：
$$\begin{aligned}
\ln{p(x)}&=\mathcal L(q)+\mathbf {KL}(q||p) \\
\mathcal L(q)&=\int q(Z)\ln\Bigg\{\frac{p(X,Z)}{q(Z)}\Bigg\}{\rm d}Z\\
\mathbf{KL}(q||p)&=-\int q(Z)\ln\Bigg\{\frac{p(Z|X)}{q(Z)}\Bigg\}{\rm d}Z
\end{aligned}$$
与EM算法相比，参数向量$\theta$不再出现，被整合到$Z$当中。与之前一样，可以通过关于$q(Z)$优化下界$\mathcal L(q)$使之达到最大值，这等价于最小化$\mathbf{KL}$散度。若可任意选择$q(Z)$，则下界最大值出现在$\mathbf{KL}$散度等于0的时候，此时$q(Z)$等于后验概率分布$p(Z|X)$.所以对于$q(Z)$的受限类型，在该范围内找到使得$\mathbf{KL}$散度最小的概率分布。所以要限制$q(Z)$使得它易于处理，也要使得这个范围足够大，充分灵活，使得它可以对真实后验概率分布足够好的近似。
需要强调的是，施加限制条件的唯一目的是为了计算方便，并且在这个限制条件下，我们应该使用尽可能丰富的近似概率分布。特别地，对于高度灵活的概率分布来说，没有“过拟合”现象。使用灵活的近似仅仅使得我们更好地近似真实的后验概率分布。
### 10.1.1 分解概率分布
我们限制概率分布$q(Z)$的范围。假设我们将$Z$的元素划分成若干个互不相交的组，记作$Z_i,i=1,\dotsb,M$,我们假定q分布关于这些分组可以分解：
$$q(Z)=\prod_{i=1}^M q_i(Z_i)$$

我们希望对于$\mathcal L(q)$关于所有概率分布$q_i(Z_i)$进行自由形式的（变分）最优化，将$q_i(Z_i)$记作$q_i$：
$$\begin{aligned}
\mathcal L(q)&=\int \prod_{i}q_i\Bigg\{\ln{p(X,Z)-\sum_i \ln{q_i}}\Bigg\}{\rm d}Z \\
&=\int q_j \Bigg\{\int \ln{p(X,Z)}\prod_{i\neq j}q_i{\rm d}Z_i\Bigg\}{\rm d}Z_j-\int q_j\ln{q_j}{\rm d}Z_j+\text{const}\\
&=\int q_j \ln{\tilde p(X,Z_j)}{\rm d}Z_j-\int q_j\ln{q_j}{\rm d}Z_j+\text{const}\\
&=\mathbf{KL}(q_j||\tilde{p}(X,Z_j))+\text{const}
\end{aligned}$$
其中：
$$\begin{aligned}
 \ln{\tilde p(X,Z_j)}&=\mathbb E_{i\neq j}[\ln{p(X,Z)}]+\text{const}\\
 \mathbb E_{i\neq j}[\ln{p(X,Z)}]&=\int \ln{p(X,Z)}\prod_{i\neq j}q_j{\rm d}Z_i
\end{aligned}$$
可以得到最优解$q^*_j(Z_j)$的一般表达式为：
$$\ln{q^*_j(Z_j)}=\mathbb E_{i\neq j}[\ln{p(X,Z)}]+\text{const}$$

这个解表明，为了得到因子$q_j$ 的最优解的对数，我们只需考虑所有隐含变量和可见变量上的联合概率分布的对数，然后关于所有其他的因子$\{q_i\}$取期望即可，其中$i\neq j$.
其中可加性常数可以通过对概率分布 $q^*_j(Z_j)$进行归一化的方式得到：
$$q^*_j(Z_j)=\frac{\exp(\mathbb E_{i\neq j}[\ln{p(X,Z)}])}{\int \exp(\mathbb E_{i\neq j}[\ln{p(X,Z)}]){\rm d}Z_j}$$

这些方程并没有给出一个显式的解，因为最优化$q^*_j(Z_j)$的公式（10.9）的右侧表达式依赖于关于其他的因子$q_i(Z_i),i\neq j$计算的期望。于是，我们会用下面的方式寻找出一个相容的解：首先，恰当地初始化所有的因子$q_i(Z_i)$,然后在各个因子上进行循环，每一轮用一个修正后的估计来替换当前因子。
### 10.1.2 分解近似的性质