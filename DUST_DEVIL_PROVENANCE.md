# Dust Devil 示例来源

本次整理日期：2026-10-20（America/Los_Angeles）。三个目录均为可单独运行的本地示例，无外部数据依赖。

源项目：用户的 Dust Devil BL / Code 本地仓库。
提取时 HEAD：`d3537f533436dd0b56d52231f2335a2b6ba78984`；本次复制的源文件在源仓库中无未提交修改。这里只记录来源，不假定该仓库有可公开访问的 URL。

| 目标 | 原始路径与处理 |
|---|---|
| Boundary_Layer_Implicit_Marching/private | src/global 中的网格、残差、Jacobian、对流处理、单站/二分/全局推进七个文件；src/rim 中的相似求解与映射两个文件。精简注释，保留计算逻辑 |
| Similarity_BVP/private | src/rim/solve_rim_bvp.m，独立副本，精简注释 |
| Bragg_Hawthorne_Picard/private | src/collision/inviscid_core 中 assemble_bh_system.m 和 build_collision_grid.m，精简注释，保留计算逻辑 |
| 三个目录的公开入口和文档 | 本次新增；BH 使用新通用 Picard 驱动和解析制造解，不复制原项目源项数据链 |

`dust_devil_source_manifest.json` 记录提取时、精简注释前的原始副本 SHA-256，用于追溯源文件，不是当前文件的校验值。注释精简前后已逐行确认可执行代码未变。

没有引入 ff-bifbox、FreeFEM 或其他第三方求解框架源码；BVP 使用 MATLAB 自带 bvp4c，边界层使用 Optimization Toolbox 的 fsolve。未为原仓库新增或更改许可证。
