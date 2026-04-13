代码整理计划
阶段一：完成 tool.py 瘦身（工作量：小）
目标：把 tool.py 中剩余的实现体全部迁移到 /code/

步骤	操作
1	将 convert_ssl_units_if_needed() 移入 code/units.py
2	将 check_variable_metadata_tiered() 移入 code/metadata.py
3	将 check_nc_completeness() 移入 code/validation.py
4	新建 code/plot.py，移入 plot_ssc_q_diagnostic()
5	新建 code/output.py，移入3个 generate_*_csv() 函数
6	tool.py 最终只剩 import + 重新导出，约 50 行
阶段二：统一数据集脚本的 import 风格（工作量：中）
目标：所有脚本直接从 code/ 导入，不再经过 tool.py 中转

优先改那些已经部分修改的数据集（ALi_De_Boer、HYBAM、Rhine、USGS、GSED），其余数据集逐步跟进。每改一个数据集，验证其 NetCDF 输出不变即可。

阶段三：推广 runtime.py 路径解析（工作量：中）
目标：消灭所有硬编码的 Output_r 路径

在每个数据集脚本顶部统一替换：

# 旧写法（硬编码）
OUTPUT_DIR = Path(__file__).parent.parent / "Output_r"

# 新写法（runtime.py）
from code.runtime import resolve_output_root
OUTPUT_DIR = resolve_output_root() / "DatasetName"

阶段四：整理文档（工作量：小）
步骤	操作
1	将 README_CN.md 内容合并到 readme.md 中作为中文章节，删除重复文件
2	在 readme.md 中补充 run_pipeline.py 用法（参考 PIPELINES.md）
3	可考虑删除 PIPELINES.md，将其内容合并到 readme.md 避免文档分散
