### DDL、工作、作业，拥有巨大压力的学生YC_H，他在临刑前的一句话让人们趋之若鹜奔向大海。“想要我的solver吗？想要的话可以全部给你，去找吧！我把所有代码都放在那里！”

## 独立 MATLAB 求解器示例

| 示例 | 方法 | 依赖 |
|---|---|---|
| [边界层隐式推进](Boundary_Layer_Implicit_Marching/README.md) | 单元中点离散、后向 Euler、解析稀疏 Jacobian、Newton | MATLAB + Optimization Toolbox |
| [非线性 Bragg–Hawthorne 型方程](Bragg_Hawthorne_Picard/README.md) | 欠松弛 Picard、稀疏 LU 重用、制造解验证 | MATLAB |
| [近边缘相似解 BVP](Similarity_BVP/README.md) | bvp4c 自适应配点、远场截断检查 | MATLAB |

各目录可独立复制运行，入口和方程见各自说明。输出图片写入各目录 `output/`，不纳入 Git。

来源：[提取记录](DUST_DEVIL_PROVENANCE.md)。这些是数值方法示例，不能将求解收敛等同于完整物理验证。
