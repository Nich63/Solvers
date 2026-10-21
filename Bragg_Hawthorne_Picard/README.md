# Bragg–Hawthorne 型算子：欠松弛 Picard + 稀疏 LU

独立的非线性椭圆方程教学例子。使用 Dust Devil 原网格和离散组装实现，新增通用源项接口及制造解；无需原项目的数据表或 MATLAB 工具箱。建议 MATLAB R2019b 或更新版本。

在 MATLAB 中进入本目录：

```matlab
out = run_bragg_hawthorne(); % 求解、绘图，保存 output/solution.png
% out = run_bragg_hawthorne(false); % 不绘图
```

## 方程与制造解

所有变量无量纲。在 0≤r,z≤1 上求解

\[
\Delta^*\psi=\psi_{rr}-\frac1r\psi_r+\psi_{zz}=S(r,z,\psi).
\]

轴 r=0 和底面 z=0：ψ=0；侧面 r=1：ψ=sin(πz/2)；顶面 z=1：ψz=0。角点以 Dirichlet 条件优先；算子不在 r=0 求值。

设 ψe=r²sin(πz/2)，采用

\[
S(r,z,\psi)=-\frac{\pi^2}{4}r^2\sin(\pi z/2)
+\frac12\left[\psi^2-r^4\sin^2(\pi z/2)\right].
\]

这样 ψe 精确满足连续非线性方程和全部边界条件。它是**制造解基准**，不是 Dust Devil 碰撞区物理解；源项也不是原研究中只依赖 ψ 的 m(ψ)。

## 数值方法

`solve_bh_picard` 冻结源项求线性解，再按 ω=0.7 欠松弛。`decomposition(A,'lu')` 只执行一次，此后重复回代。非均匀三点差分组装内部算子，顶面使用三点单边导数。

每次更新后重新计算非线性 PDE 残差和边界残差，二者均低于容差才接受。输出同时保留 `converged`、迭代次数和残差历史；达到迭代上限不代表收敛。默认演示的原始残差门槛为 1e−9。

该平滑弱非线性算例不验证原 Dust Devil 非光滑源项的 Picard 收敛性。更换源项、网格或非线性强度后需重新选择松弛参数并验证。

来源见根目录 `DUST_DEVIL_PROVENANCE.md`。
