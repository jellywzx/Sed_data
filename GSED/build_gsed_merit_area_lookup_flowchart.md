# `build_gsed_merit_area_lookup.py` Flowchart

这个脚本的目标是：

1. 读取 GSED 的 `R_ID`
2. 从 `GSED_Reach.shp/.dbf` 恢复每条 reach 的 centroid、端点候选和层级信息
3. 先把端点候选匹配到 MERIT reach，并据此推断下游锚点
4. 再从已知 MERIT reach 直接追溯上游 basin，避免重复做第二次 reach 匹配
5. 把匹配到的 `uparea` 写成 `R_ID -> upstream_area_km2` lookup 表
6. 同时输出一组审计字段，供后续筛选和人工检查

## Mermaid

```mermaid
flowchart TD
    A[启动脚本] --> B[parse_args 解析参数]
    B --> C[确定默认路径<br/>GSED CSV / GSED shp / MERIT dir / basin_tracer_dir / output csv]

    C --> D[读取 GSED_Reach_Monthly_SSC.csv<br/>只取 R_ID]
    D --> E[标准化 R_ID<br/>_normalize_gsed_rid]

    E --> F[读取 GSED_Reach.dbf]
    F --> G[按记录顺序读取 GSED_Reach.shp]
    G --> H[提取 centroid 和所有端点候选<br/>_extract_polyline_representatives]
    H --> I[组合 reach 元数据<br/>R_ID / R_level / Length / centroid / endpoints / basin_code_l1-l4]

    I --> J[动态导入 basin_tracer.py 和 basin_policy.py]
    J --> K[实例化 UpstreamBasinTracer]

    K --> L{遍历每个 GSED R_ID}
    L --> M[读取 centroid 与端点候选]
    M --> N{是否至少有一个可用锚点}
    N -- 否 --> O[写入空匹配结果<br/>missing_gsed_coords]
    N -- 是 --> P[遍历端点候选<br/>逐个调用 tracer.find_best_reach]
    P --> Q{是否至少有一个端点匹配成功}
    Q -- 是 --> R[选 uparea 最大的端点<br/>若并列取距离更近]
    Q -- 否 --> S[回退 centroid<br/>调用 tracer.find_best_reach]
    R --> T[调用 tracer.get_upstream_basin_from_reach]
    S --> T
    T --> U[得到 MERIT basin 结果<br/>basin_id/COMID<br/>uparea_merit<br/>distance<br/>pfaf_code<br/>point_in_local<br/>point_in_basin<br/>method]
    U --> V[调用 classify_basin_result]
    V --> W{是否满足接受条件}

    W -- 否 --> X[merit_lookup_accept = False]
    W -- 是 --> Y[merit_lookup_accept = True]

    X --> Z[写入 lookup 行<br/>含 gsed_anchor_source / gsed_endpoint_match_count]
    Y --> Z
    O --> Z

    Z --> AA{是否处理完全部 R_ID}
    AA -- 否 --> L
    AA -- 是 --> AB[汇总为 DataFrame]
    AB --> AC[写出 GSED_Reach_upstream_area.csv]
    AC --> AD[打印匹配统计与 match_quality 分布]
```

## 接受条件

脚本里当前把一条 MERIT 面积标记为可正式回填的条件定义为：

1. `merit_basin_status == "resolved"`
2. `merit_method == "upstream_traced"`

也就是：

1. 必须通过 `scripts_basin_test` 共享的 basin policy
2. 必须来自真实上游流域追溯结果
3. 不能只是 fallback buffer

## 输出表的关键字段

最终输出的 `GSED_Reach_upstream_area.csv` 里，最关键的是这些列：

1. `R_ID`
2. `upstream_area_km2`
3. `merit_comid`
4. `merit_pfaf_code`
5. `merit_distance_m`
6. `merit_match_quality`
7. `merit_method`
8. `merit_point_in_local`
9. `merit_point_in_basin`
10. `merit_basin_status`
11. `merit_basin_flag`
12. `merit_lookup_accept`
13. `gsed_anchor_source`
14. `gsed_endpoint_match_count`

## 与后续 GSED 主流程的关系

`process_gsed_cf18.py` 现在会自动读取这张 lookup 表，并且优先只使用：

1. `merit_lookup_accept = True`

的记录，把它们写入 GSED 输出 NetCDF 的 `upstream_area`。
