# 用现代 C++ 学习 C4.5 决策树

一个 C++20 教学项目，演示数据读取、分裂选择、递归建树、缺失值处理、后剪枝和预测。

**算法、演示和评测都在原有的 [`DecisionTree_C4.5.cpp`](DecisionTree_C4.5.cpp) 中**，按顺序阅读即可。独立测试放在 [`tests.cpp`](tests.cpp)，继续使用原有的 `Golf.dat` 示例。代码只依赖 C++ 标准库。

## 运行

需要 GCC 11+、GNU Make 和 Python 3.9+；本机验证环境是 Linux / GCC 14.2。Python 只用于数据校验和转换，模型和评测均由 C++ 执行。

```bash
# 在项目根目录执行：校验数据、构建、运行测试
./scripts/setup.sh

# 查看 Golf 决策树、逐条预测与概率
./build/c45 Golf.dat
./build/c45 Golf.dat --no-prune

# 检查内存错误和未定义行为
make sanitize

# 评测三个标准数据集
make benchmark
```

Debian / Ubuntu 缺少工具时，可安装 `build-essential python3`。日常构建和评测不需要联网或后台服务。`./scripts/setup.sh --sanitize` 可以一次完成构建、测试和内存检查；编译并行度默认是 2，可用 `BUILD_JOBS` 调整。

产物都在 `build/`，`make clean` 可清理。更换编译器或选项时使用独立目录，例如 `make BUILD=build/debug CXXFLAGS='-O0 -g' test`。也可以直接编译原有源文件：

```bash
g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic DecisionTree_C4.5.cpp -o build/c45
```

## 按什么顺序读代码

先运行 Golf 示例，再从文件末尾的 `main`、`demo` 进入 `DecisionTree::fit`。训练的主线是：

```text
fit → build → find_split → discrete_split / continuous_split
          ↓
        prune
```

| 阅读内容 | 实现入口 | 要理解的决定 |
|---|---|---|
| 数据与类型 | `Dataset`、`read_dataset` | 数值只解析一次；用 `Cell` 区分数值、类别和缺失 |
| 分裂评分 | `entropy`、`find_split` | 先筛信息增益，再比较增益率 |
| 离散 / 连续特征 | `discrete_split`、`continuous_split` | 离散多路分支，连续二分阈值 |
| 递归生长 | `build`、`branch_for` | 何时停止、如何分配样本、哪些特征能继续使用 |
| 后剪枝 | `estimated_errors`、`prune` | 比较保留子树与改为叶子的估计错误 |
| 预测 | `predict_at`、`predict_proba` | 走确定分支，或对多个分支的概率加权 |
| 评测 | `stratified_folds`、`benchmark`、`macro_f1` | 保证训练与测试隔离，报告效果和耗时 |

测试通过 `C45_TESTING` 省略源文件的 `main`，直接复用同一份实现。因此不需要另外维护一份算法或测试接口。

## 算法设计与取舍

### 分裂评分

对类别权重计算 Shannon 熵，再计算分裂的信息增益和增益率：

```text
H(S) = -Σ p(c) log2 p(c)
Gain = H(S) - Σ P(branch) H(branch)
SplitInfo = -Σ P(branch) log2 P(branch)
GainRatio = Gain / SplitInfo
```

`SplitInfo` 使用**实际分支的比例**。例如连续特征切成左右各一半，其值为 1，与原始数值有多少个不同取值无关。

直接选择最大增益率，可能选中分母很小、实际收益也很小的分裂。因此先对有效的正增益候选求平均增益，只在达到平均值的候选中比较增益率。没有正增益就停止。评分近似相等时保留先出现的特征，预测概率相等时选择编码顺序靠前的类别。

### 连续特征与缺失值

连续特征在每个节点排序一次，扫描时更新左右类别计数，只在不同取值之间评估阈值。阈值按信息增益选择，预测使用 `<=` 和 `>`。`std::midpoint` 避免极大值溢出；中点若舍入到右端点，则改用左端点。

默认启用 MDL 校正：从已知样本上的增益减去 `log2(M) / K`，其中 `M` 是合法切分数量，`K` 是已知样本总权重，用于抵消搜索多个阈值带来的乐观偏差。

对于缺失值，设节点总权重为 `W`：

- 先在已知样本上计算增益，再乘 `K / W`；`SplitInfo` 另计缺失值组的比例。
- 建树时，把缺失样本按已知分支的比例分配。例如两个分支占 60% 和 40%，该样本就分别贡献 0.6 和 0.4 的权重。
- 预测时遇到缺失值或新类别，按相同分支比例混合子树概率。

`build` 只传递行号和权重，不复制原始数据。离散特征分裂后不再使用；连续特征可以在更深层再次分裂。全缺失列、常量列不会产生分裂。

### 剪枝与参数

树先生长，再自底向上比较错误估计：如果一个多数类叶子的估计错误不高于整个子树，就用叶子替换子树。零误差做二项式修正，少于一次错误时插值，其余情况用带连续性修正的 Wilson 上界。

| `Options` 参数 | 默认值 | 作用 |
|---|---:|---|
| `min_leaf_weight` | 2 | 连续分裂两侧均达到该已知权重；离散分裂至少两个分支达到 |
| `max_depth` | 64 | 限制递归深度，根深度为 0，配置上限为 256 |
| `mdl_correction` | true | 校正数值阈值搜索的乐观偏差 |
| `pruning` | true | 启用后剪枝 |
| `confidence` | 0.25 | 错误上界的单侧尾概率，越小越保守；允许 `[1e-6, 0.5]` |

本项目实现 C4.5 核心流程，剪枝采用**子树替换**，省略子树提升和规则提取，以保持递归逻辑直接可读。阈值候选计数、平均增益筛选等约定也可能不同于原版 C4.5 / Weka J48，因此不保证生成完全相同的树。

### C++ 实现

`std::variant` 表达特征类型，`std::unique_ptr` 管理树节点，`std::span` 接收只读输入，`std::vector` 保存动态计数。训练数据不被模型持有；重新训练、剪枝和析构都会自动释放节点。非法输入抛出异常，命令行将其转换为错误信息和退出码 1。

这里保留字符串类别、节点内排序和小概率向量，方便对应算法步骤。代价是额外的存储和分配开销。单节点连续搜索约为 `O(d × (s log s + sC))`，`s` 是到达节点的样本条目数，`d` 是连续特征数，`C` 是类别数。整树成本还取决于深度；缺失样本的多分支传播会增加条目数量。当前规模不需要全局预排序缓存或并行训练。

## 数据输入与预测概率

沿用两行表头：第一行是类型，第二行是列名，最后一列为类别。

```text
Continuous, Discrete, Discrete
temperature, outlook, decision
20, sunny, Play
?, rain, Don't Play
25, , Play
```

支持空行、CRLF、字段前后空白；特征中的 `?` 或空字段表示缺失。类别不能为空，数值必须完整且有限，拒绝 `nan`、`inf`、`12kg`。为聚焦算法，不支持带引号、字段内逗号或多行字段的 CSV；需要时先转换输入。

`demo` 展示最小使用流程：`load_dataset` → `tree.fit` → `tree.predict_proba`。`predict` 返回类别编码，`predict_proba` 返回按 `data.labels` 顺序排列的全部类别概率。`fit(data, training_indices)` 可以只用指定行训练。

命令行的 `predicted_probability` 是所预测类别的加权经验概率，**与剪枝参数 `confidence` 不同，也未做概率校准**。Golf 的 14/14 是训练结果，泛化效果见交叉验证。

## 测试与标准数据

`make test` 运行 24 项具名检查，覆盖熵和分裂评分、浮点阈值、缺失权重守恒、未见类别、剪枝、输入验证、训练/测试隔离及指标计算；另外检查命令行的正常输出和错误退出码。`make sanitize` 用 AddressSanitizer、UndefinedBehaviorSanitizer 检查内存与未定义行为，两种构建均通过。

CI 运行同样的测试和三个数据集评测。本机只验证了 Linux / GCC，Windows/MSVC 尚未实测，VS2013 不满足 C++20 要求。

| 数据集 | 样本 × 特征 | 类别 | 来源与版本说明 |
|---|---:|---:|---|
| Iris | 150 × 4 | 3 | Fisher / [UCI](https://archive.ics.uci.edu/dataset/53/iris)，[版本说明](https://github.com/scikit-learn/scikit-learn/blob/1.7.2/sklearn/datasets/descr/iris.rst) |
| Wine | 178 × 13 | 3 | Forina 等 / [UCI](https://archive.ics.uci.edu/dataset/109/wine)，[版本说明](https://github.com/scikit-learn/scikit-learn/blob/1.7.2/sklearn/datasets/descr/wine_data.rst) |
| Breast Cancer Wisconsin Diagnostic | 569 × 30 | 2 | Wolberg、Street、Mangasarian / [UCI](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)，[版本说明](https://github.com/scikit-learn/scikit-learn/blob/1.7.2/sklearn/datasets/descr/breast_cancer.rst) |

数据来自 scikit-learn **1.7.2** 的固定文件，已附在 `data/`；不依赖其 Python 库。Iris 使用该版本的两处数值修正。转换只增加表头并还原类别名，不筛选样本或特征；分发许可见 [`data/LICENSE-scikit-learn.txt`](data/LICENSE-scikit-learn.txt)。

[`scripts/prepare_data.py`](scripts/prepare_data.py) 固定上游和转换后文件的 SHA-256：`--check` 离线校验，`--download` 通过 HTTPS 从 `raw.githubusercontent.com` 校验并重建。

## 效果与性能

固定分层 **5 折 × 5 次重复**，种子 `2026..2030`。每次重复的所有样本恰好测试一次；阈值、剪枝和多数类基线都只使用训练折。两种模型使用相同划分，区别只有是否剪枝，未根据这些结果调参。

准确率和 Macro-F1 先按每次重复的全部折外预测计算，再汇总均值与总体标准差。标准差描述划分敏感性，不是置信区间。

训练耗时排除读取、划分和预热；预测包含输入检查与概率计算，每折循环 50 次。表中为 25 折计时的中位数。环境为 Debian 13、AMD EPYC 9V74 云主机、GCC 14.2、`-O2`，程序单线程；共享主机的耗时会波动。

| 数据集 | 模型 | 准确率（均值 ± SD） | Macro-F1 | 训练 ms | 预测 μs/条 | 平均叶子数 |
|---|---|---:|---:|---:|---:|---:|
| iris | 不剪枝 | 94.27% ± 0.80% | 0.9426 | 0.041 | 0.038 | 6.00 |
| iris | 剪枝 | 95.07% ± 0.90% | 0.9506 | 0.045 | 0.035 | 4.44 |
| wine | 不剪枝 | 92.81% ± 1.79% | 0.9283 | 0.331 | 0.052 | 6.40 |
| wine | 剪枝 | 93.15% ± 2.20% | 0.9321 | 0.337 | 0.048 | 5.32 |
| breast_cancer | 不剪枝 | 93.32% ± 1.27% | 0.9286 | 3.938 | 0.088 | 12.28 |
| breast_cancer | 剪枝 | 93.36% ± 1.25% | 0.9289 | 4.155 | 0.075 | 10.96 |

多数类基线准确率分别为 33.33%、39.89%、62.74%。不剪枝版本用于观察剪枝的影响，当前结果不能推导对任意数据集优于其他模型。

完整结果和复现信息见 [`results/benchmark.csv`](results/benchmark.csv)、[`results/environment.txt`](results/environment.txt)。等待编译和测试结束后，可重新生成 CSV：

```bash
./build/c45 --benchmark data/iris.dat data/wine.dat data/breast_cancer.dat > results/benchmark.csv
```

## 历史 PR 与 issues

| 讨论 | 结论与当前处理 |
|---|---|
| [PR #1](https://github.com/zgyao/DecisionTree_C4.5/pull/1) | 感谢 aardvarkk 提供编译、空行、裁剪和参数输入等修复思路。已吸收这些意图，但其版本仍有空指针调用和泄漏，因此未直接合并旧补丁。 |
| [#2 编译问题](https://github.com/zgyao/DecisionTree_C4.5/issues/2) | 已用标准容器和节点所有权消除变长数组、未初始化指针等根因。仅把旧指针设为 `NULL` 仍不能合法调用非静态成员函数。 |
| [#3 预测置信度](https://github.com/zgyao/DecisionTree_C4.5/issues/3) | 已提供概率 API、命令行概率输出和回归测试，区别于剪枝置信参数。 |
| [#4 增益率分母](https://github.com/zgyao/DecisionTree_C4.5/issues/4) | 报告正确；已按阈值形成的分支计算 SplitInfo，并用相同分区、不同原值数量的案例防止回归。 |

建议新版本合入后，附实现和测试链接回复相关讨论，再由维护者关闭已解决事项。本地改动未执行远端合并、回复或关闭。

## 参考

- J. R. Quinlan, *C4.5: Programs for Machine Learning*, 1993。
- J. R. Quinlan, “Improved Use of Continuous Attributes in C4.5”, 1996，[DOI: 10.1613/jair.279](https://doi.org/10.1613/jair.279)。
