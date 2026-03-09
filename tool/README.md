# Output_r 数据处理脚本（Tool）

本目录包含对 **Output_r** 数据目录进行收集、聚类、合并、校验等处理的全套脚本。  
所有脚本的**数据根目录**均指向与 `Script` 同级的 **Output_r**（即 `sediment_wzx_1111/Output_r`），与当前工作目录无关。

## 目录结构

- `01_collect/` — 从 Output_r 下所有 qc 文件夹扫描 .nc，输出 collected_stations.csv
- `02_cluster/` — 按经纬度聚类，输出 clustered_stations.csv 等
- `03_merge/` — 按位置合并时间序列、处理重叠、生成 merged_all.nc
- `04_overlap/` — 按 source 拆分 overlap CSV
- `05_plot/` — 从 merged_all.nc 绘图
- `06_verify/` — 时间分辨率校验与按分辨率重组

## 运行方式

在 **sediment_wzx_1111** 目录下（或任意目录，脚本内部会解析 Output_r 路径）运行，例如：

```bash
cd /path/to/sediment_wzx_1111

# 步骤 1：收集站点
python Script/tool/01_collect/s1_collect_qc_stations.py

# 步骤 2：聚类
python Script/tool/02_cluster/s2_cluster_qc_stations.py

# 步骤 6：时间分辨率校验
python Script/tool/06_verify/verify_time_resolution.py
python Script/tool/06_verify/reorganize_qc_by_resolution.py --clear -j 8
```

各脚本的 `--root`、`--out`、`--verify-csv` 等参数默认均指向 Output_r 下的路径；如需覆盖，可传入绝对路径或相对 Output_r 的路径（视脚本说明而定）。
