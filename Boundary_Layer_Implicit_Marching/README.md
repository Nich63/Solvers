# 隐式边界层推进：Keller–box 型离散 + Newton

从 Dust Devil 主边界层代码提取的独立短程算例。全部依赖均在本目录，启动相似解也有独立副本；无需其他例子或原项目数据。

需要 MATLAB R2019b 或更新版本及 **Optimization Toolbox**（`fsolve`）。

在 MATLAB 中进入本目录：

```matlab
result = run_boundary_layer();  % 求解、绘图，保存 output/profiles.png
% result = run_boundary_layer(false); % 不绘图
```

配置入口是 `example_bl_config.m`。默认 Ri=Pr=1，ζ0=0.001，推进至 r=0.85；61 个径向站、81 个垂向节点、zmax=6。选择短程附着流是为了演示离散与求解，不覆盖近轴极限或全盘生产计算。

## 坐标、方程与边界

所有变量无量纲：ζ=1−r 是推进坐标，z 是边界层法向高度，s=−log r 只用于生成网格。守恒变量 U=ru、V=rv、W=rw；压力 p=−1/(2r²)+pb。状态排列为 `[U; A; V; B; W; Theta; H; pb]`，其中 A=Uz、B=Vz、H=Θz。

\[
W_z=U_\zeta,\qquad (p_b)_z=Ri\Theta,
\]
\[
rA_z+UU_\zeta-WA+\frac{U^2}{r}+r^2(p_b)_\zeta-\frac{1-V^2}{r}=0,
\]
\[
rB_z+UV_\zeta-WB=0,\qquad
\frac r{Pr}H_z+U\Theta_\zeta-WH=0.
\]

z=0：U=V=W=0、Θ=1；z=zmax：U=0、V=1、Θ=0、pb=0。远场 W 由求解确定，不额外设成零。

初始站由本目录的相似解生成。令 η=Ri^(1/5)ζ0^(−2/5)z，则 U=r Ri^(2/5)ζ0^(1/5)F'，pb=Ri^(4/5)ζ0^(2/5)Π，W=r Ri^(1/5)ζ0^(−2/5)(3F/5−2ηF'/5)，V 与 Θ 直接取相似解；导数量由链式法则得到。相似方程见 `private/solve_rim_bvp.m`（同批次 `Similarity_BVP/README.md` 也有说明，但不是运行依赖）。

## 数值方法

- z 方向使用单元中点平均与差分；ζ 方向使用全隐式后向 Euler。不能称为两个方向均二阶的经典 Keller–box。
- 单站由 `fsolve` 使用解析稀疏 Jacobian 求解；行列尺度调整改善条件数。
- 原核心保留失败步二分功能；本例设 `maxStepHalvings=0`，便于明确比较同一组目标站。
- 原核心的接受条件为：边界残差通过，且原始或行归一化单元残差之一通过。
- 原核心保留 FLARE/DUFLARE 接口，但默认仅使用 `standard`，本例未验证反向流处理。

`private/` 保留已用过的求解实现，外层入口为本次整理新增。

## 适用范围

本例用于短程附着流的数值方法演示，不覆盖全盘、近轴、反向流或任意参数。网格、截断高度和有限 ζ0 相似启动都会影响精度；小方程残差不代表物理验证。

来源见根目录 `DUST_DEVIL_PROVENANCE.md`。
