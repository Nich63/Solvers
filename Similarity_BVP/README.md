# 相似解边值问题：加热圆盘近边缘流动

从 Dust Devil 项目的近边缘相似解提取，保留原 `bvp4c` 方程。目录可单独复制运行，不需要原项目、输入数据或其他示例目录。使用 MATLAB R2019b 或更新版本；无需额外工具箱。

在 MATLAB 中进入本目录：

```matlab
rim = run_similarity_bvp();   % 求解、绘图，保存 output/profiles.png
% rim = run_similarity_bvp(false); % 不绘图
```

## 方程与边界条件

所有变量无量纲，η 是相似坐标，不能直接当作物理高度 z。撇号表示对 η 求导。未知量为 F、Π、Θ、V：

\[
F'''=\frac35FF''-\frac15(F')^2-\frac25(\Pi-\eta\Theta),\quad
\Pi'=\Theta,\quad
\Theta''=\frac35Pr F\Theta',\quad
V''=\frac35F V'.
\]

壁面 η=0：F=F'=V=0，Θ=1；截断远场 η=ηmax：F'=Π=Θ=0，V=1。默认 Pr=1，ηmax=20；Pr=1 时有 Θ+V=1。

`private/solve_rim_bvp.m` 把方程转为八个一阶 ODE，并用 `bvp4c` 自适应配点求解。修改 `run_similarity_bvp.m` 内的配置即可改变 Pr 或截断高度。

## 适用范围

边界残差为零不代表无穷远截断正确。改变 Pr 后需要重新检查 ηmax 和求解容差；Pr=1 时可用 Θ+V=1 检查数值一致性。

来源见根目录 `DUST_DEVIL_PROVENANCE.md`。
