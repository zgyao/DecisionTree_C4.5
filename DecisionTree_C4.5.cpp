// C++20 教学示例：数据读取 → 分裂选择 → 建树 → 剪枝 → 预测。
// main 和单机评测放在文件末尾；tests.cpp 复用本文件检查算法。
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace c45 {

// 1. 数据格式与读取

enum class Kind { continuous, discrete };
using Cell = std::variant<std::monostate, double, std::string>; // monostate 表示缺失值

struct Feature {
    std::string name;
    Kind kind;
};

struct Sample {
    std::vector<Cell> values;
    std::size_t label;
};

struct Dataset {
    std::vector<Feature> features;
    std::vector<std::string> labels;
    std::vector<Sample> rows;
};

std::string trim(const std::string &text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

// 输入采用简单的逗号分隔格式；带引号的字段需要先转换。
std::vector<std::string> split_fields(const std::string &line) {
    if (line.find('"') != std::string::npos) {
        throw std::invalid_argument("quoted CSV fields are not supported");
    }
    std::vector<std::string> result;
    std::size_t begin = 0;
    for (;;) {
        const auto end = line.find(',', begin);
        result.push_back(trim(line.substr(begin, end - begin)));
        if (end == std::string::npos) {
            return result;
        }
        begin = end + 1;
    }
}

// 两行表头：类型、列名。空行忽略，特征中的 ? 或空字段表示缺失。
Dataset read_dataset(std::istream &input) {
    Dataset data;
    std::vector<std::string> types;
    std::string line;
    std::size_t line_number = 0;
    std::size_t header_rows = 0;

    while (std::getline(input, line)) {
        ++line_number;
        if (trim(line).empty()) {
            continue;
        }
        try {
            const auto cells = split_fields(line);

            // 第一行声明类型，最后一列是类别。
            if (header_rows == 0) {
                if (cells.size() < 2 || cells.back() != "Discrete") {
                    throw std::invalid_argument("need feature columns and a Discrete class column");
                }
                for (const auto &type : cells) {
                    if (type != "Continuous" && type != "Discrete") {
                        throw std::invalid_argument("unknown column type: " + type);
                    }
                }
                types = cells;
                ++header_rows;
                continue;
            }
            if (cells.size() != types.size()) {
                throw std::invalid_argument("column count does not match schema");
            }

            // 第二行声明列名；重名会让打印出的决策规则产生歧义。
            if (header_rows == 1) {
                for (std::size_t j = 0; j < cells.size(); ++j) {
                    const auto previous_end = cells.begin() + static_cast<std::ptrdiff_t>(j);
                    const bool duplicate =
                        std::find(cells.begin(), previous_end, cells[j]) != previous_end;
                    if (cells[j].empty() || duplicate) {
                        throw std::invalid_argument("column names must be nonempty and unique");
                    }
                    if (j + 1 < cells.size()) {
                        const Kind kind =
                            types[j] == "Continuous" ? Kind::continuous : Kind::discrete;
                        data.features.push_back({cells[j], kind});
                    }
                }
                ++header_rows;
                continue;
            }

            // 后续每行是一条样本。数值在这里解析一次，建树时直接使用。
            Sample row;
            for (std::size_t j = 0; j < data.features.size(); ++j) {
                if (cells[j].empty() || cells[j] == "?") {
                    row.values.emplace_back(std::monostate{});
                } else if (data.features[j].kind == Kind::discrete) {
                    row.values.emplace_back(cells[j]);
                } else {
                    std::size_t consumed = 0;
                    const double value = std::stod(cells[j], &consumed);
                    if (consumed != cells[j].size() || !std::isfinite(value)) {
                        throw std::invalid_argument("numeric feature must be a finite number");
                    }
                    row.values.emplace_back(value);
                }
            }
            if (cells.back().empty() || cells.back() == "?") {
                throw std::invalid_argument("class label must not be missing");
            }
            const auto found = std::find(data.labels.begin(), data.labels.end(), cells.back());
            row.label = static_cast<std::size_t>(found - data.labels.begin());
            if (found == data.labels.end()) {
                data.labels.push_back(cells.back());
            }
            data.rows.push_back(std::move(row));
        } catch (const std::exception &error) {
            throw std::invalid_argument("line " + std::to_string(line_number) + ": " +
                                        error.what());
        }
    }
    if (input.bad()) {
        throw std::runtime_error("failed while reading dataset");
    }
    if (data.rows.empty()) {
        throw std::invalid_argument("dataset has no samples");
    }
    return data;
}

Dataset load_dataset(const std::string &path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot open dataset: " + path);
    }
    return read_dataset(file);
}

// 2. 分裂、建树、剪枝与预测
double entropy(std::span<const double> counts) {
    const double total = std::accumulate(counts.begin(), counts.end(), 0.0);
    if (total <= 0.0) {
        return 0.0;
    }
    double result = 0.0;
    for (double count : counts) {
        if (count <= 0.0) {
            continue;
        }
        const double p = count / total;
        result -= p * std::log2(p);
    }
    return result;
}

struct Options {
    double min_leaf_weight = 2.0;
    std::size_t max_depth = 64; // 限制递归深度与缺失值的分支扩张
    bool mdl_correction = true; // 对搜索多个数值阈值进行惩罚
    bool pruning = true;
    double confidence = 0.25; // 越小，剪枝时的错误估计越保守
};

struct Node {
    std::vector<double> counts;         // 各类别的训练权重，允许分数
    std::optional<std::size_t> feature; // 无分裂特征时是叶子节点
    double threshold = 0.0;
    std::vector<std::string> categories; // 离散分支的有序取值；连续分裂时为空
    std::vector<double> priors;          // 已知特征值中的分支比例，与 children 顺序一致
    std::vector<std::unique_ptr<Node>> children;

    [[nodiscard]] bool is_leaf() const {
        return !feature.has_value();
    }
};

class DecisionTree {
public:
    explicit DecisionTree(Options options = {}) : options_(options) {
        if (!std::isfinite(options.min_leaf_weight) || options.min_leaf_weight < 1.0 ||
            options.max_depth > 256 || !std::isfinite(options.confidence) ||
            options.confidence < 1e-6 || options.confidence > 0.5) {
            throw std::invalid_argument(
                "invalid tree options (weight >= 1, depth <= 256, confidence in [1e-6, .5])");
        }
    }

    void fit(const Dataset &data) {
        std::vector<std::size_t> indices(data.rows.size());
        std::iota(indices.begin(), indices.end(), 0);
        fit(data, indices);
    }

    // 指定训练行，保证交叉验证的测试行不参与分裂选择或剪枝。
    void fit(const Dataset &data, std::span<const std::size_t> indices) {
        if (data.features.empty() || data.labels.empty() || indices.empty()) {
            throw std::invalid_argument(
                "training set, features and class vocabulary must be nonempty");
        }
        std::vector<bool> seen(data.rows.size(), false);
        std::vector<WeightedRow> rows;
        rows.reserve(indices.size());
        for (auto index : indices) {
            if (index >= data.rows.size() || seen[index]) {
                throw std::invalid_argument("training indices must be unique and in range");
            }
            seen[index] = true;
            validate_values(data.rows[index].values, data.features);
            if (data.rows[index].label >= data.labels.size()) {
                throw std::invalid_argument("class index out of range");
            }
            rows.push_back({index, 1.0});
        }
        // 新树建好后再替换旧树，非法输入不会破坏已训练的模型。
        auto root = build(data, rows, std::vector<bool>(data.features.size(), true), 0);
        if (options_.pruning) {
            prune(*root);
        }
        features_ = data.features;
        labels_ = data.labels;
        root_ = std::move(root);
    }

    [[nodiscard]] std::vector<double> predict_proba(std::span<const Cell> values) const {
        require_fitted();
        validate_values(values, features_);
        return predict_at(*root_, values);
    }

    [[nodiscard]] std::size_t predict(std::span<const Cell> values) const {
        const auto probabilities = predict_proba(values);
        const auto best = std::max_element(probabilities.begin(), probabilities.end());
        return static_cast<std::size_t>(best - probabilities.begin());
    }

    [[nodiscard]] const Node &root() const {
        require_fitted();
        return *root_;
    }

    [[nodiscard]] const std::vector<std::string> &labels() const {
        require_fitted();
        return labels_;
    }

    void print(std::ostream &out) const {
        require_fitted();
        print_node(out, *root_, "");
    }

private:
    static constexpr double epsilon = 1e-12;

    struct WeightedRow {
        std::size_t index;
        double weight;
    };

    struct Split {
        std::size_t feature = 0;
        double threshold = 0.0;
        double gain = 0.0;
        double split_info = 0.0;
        std::vector<std::string> categories;
        std::vector<double> weights;
    };

    Options options_;
    std::vector<Feature> features_;
    std::vector<std::string> labels_;
    std::unique_ptr<Node> root_;

    void require_fitted() const {
        if (!root_) {
            throw std::logic_error("fit the tree before prediction or inspection");
        }
    }

    static void validate_values(std::span<const Cell> values,
                                const std::vector<Feature> &features) {
        if (values.size() != features.size()) {
            throw std::invalid_argument("feature count mismatch");
        }
        for (std::size_t j = 0; j < values.size(); ++j) {
            if (std::holds_alternative<std::monostate>(values[j])) {
                continue;
            }
            if (features[j].kind == Kind::continuous) {
                const auto *number = std::get_if<double>(&values[j]);
                if (!number || !std::isfinite(*number)) {
                    throw std::invalid_argument("expected a finite numeric feature");
                }
            } else if (!std::holds_alternative<std::string>(values[j])) {
                throw std::invalid_argument("expected a categorical feature");
            }
        }
    }

    static double total(std::span<const double> counts) {
        return std::accumulate(counts.begin(), counts.end(), 0.0);
    }

    // SplitInfo 按分裂后的分支计算。连续特征使用左右两组，
    // 不能按原始数值各出现几次来算；另加入缺失值组的权重。
    static double split_information(const std::vector<double> &weights, double missing) {
        auto groups = weights;
        if (missing > epsilon) {
            groups.push_back(missing);
        }
        return entropy(groups);
    }

    Split discrete_split(const Dataset &data, const std::vector<WeightedRow> &rows,
                         std::size_t feature, double weight) const {
        std::map<std::string, std::vector<double>> groups;
        std::vector<double> known_counts(data.labels.size(), 0.0);
        for (const auto &row : rows) {
            const auto &sample = data.rows[row.index];
            if (const auto *value = std::get_if<std::string>(&sample.values[feature])) {
                auto &counts = groups.try_emplace(*value, data.labels.size(), 0.0).first->second;
                counts[sample.label] += row.weight;
                known_counts[sample.label] += row.weight;
            }
        }
        Split split;
        split.feature = feature;
        const double known = total(known_counts);
        if (known <= epsilon || groups.size() < 2) {
            return split;
        }
        double after = 0.0;
        std::size_t substantial_branches = 0;
        for (const auto &[category, counts] : groups) {
            const double branch_weight = total(counts);
            split.categories.push_back(category);
            split.weights.push_back(branch_weight);
            after += branch_weight / known * entropy(counts);
            if (branch_weight + epsilon >= options_.min_leaf_weight) {
                ++substantial_branches;
            }
        }
        // 至少两个分支达到最小权重，其他稀有类别可以有更小的分支。
        if (substantial_branches < 2) {
            return Split{};
        }
        split.gain = (known / weight) * (entropy(known_counts) - after);
        split.split_info = split_information(split.weights, weight - known);
        return split;
    }

    Split continuous_split(const Dataset &data, const std::vector<WeightedRow> &rows,
                           std::size_t feature, double weight) const {
        struct Entry {
            double value;
            double weight;
            std::size_t label;
        };

        std::vector<Entry> sorted;
        std::vector<double> right(data.labels.size(), 0.0);
        std::vector<double> left(data.labels.size(), 0.0);
        for (const auto &row : rows) {
            const auto &sample = data.rows[row.index];
            if (const auto *value = std::get_if<double>(&sample.values[feature])) {
                sorted.push_back({*value, row.weight, sample.label});
                right[sample.label] += row.weight;
            }
        }
        std::stable_sort(sorted.begin(), sorted.end(),
                         [](const Entry &a, const Entry &b) { return a.value < b.value; });
        Split best;
        best.feature = feature;
        const double known = total(right);
        const double before = entropy(right);
        double left_weight = 0.0;
        double best_raw_gain = -1.0;
        std::size_t candidates = 0;
        // 从左到右移动样本，一次扫描更新两侧类别计数。相同取值不拆开。
        for (std::size_t i = 0; i + 1 < sorted.size(); ++i) {
            left[sorted[i].label] += sorted[i].weight;
            right[sorted[i].label] -= sorted[i].weight;
            left_weight += sorted[i].weight;
            const double right_weight = known - left_weight;
            if (sorted[i].value == sorted[i + 1].value ||
                left_weight + epsilon < options_.min_leaf_weight ||
                right_weight + epsilon < options_.min_leaf_weight) {
                continue;
            }
            ++candidates;
            const double gain = before - left_weight / known * entropy(left) -
                                right_weight / known * entropy(right);
            if (gain > best_raw_gain + epsilon) {
                best_raw_gain = gain;
                // midpoint 避免溢出；若中点舍入到右端点，改用左端点。
                best.threshold = std::midpoint(sorted[i].value, sorted[i + 1].value);
                if (best.threshold >= sorted[i + 1].value) {
                    best.threshold = sorted[i].value;
                }
                best.weights = {left_weight, right_weight};
            }
        }
        if (candidates == 0) {
            return best;
        }
        const double penalty =
            options_.mdl_correction ? std::log2(static_cast<double>(candidates)) / known : 0.0;
        best.gain = (known / weight) * (best_raw_gain - penalty);
        best.split_info = split_information(best.weights, weight - known);
        return best;
    }

    std::optional<Split> find_split(const Dataset &data, const std::vector<WeightedRow> &rows,
                                    const std::vector<bool> &available, double weight) const {
        std::vector<Split> candidates;
        double gain_sum = 0.0;
        for (std::size_t j = 0; j < data.features.size(); ++j) {
            if (!available[j]) {
                continue;
            }
            Split split;
            if (data.features[j].kind == Kind::continuous) {
                split = continuous_split(data, rows, j, weight);
            } else {
                split = discrete_split(data, rows, j, weight);
            }
            if (split.gain > epsilon && split.split_info > epsilon) {
                gain_sum += split.gain;
                candidates.push_back(std::move(split));
            }
        }
        if (candidates.empty()) {
            return std::nullopt;
        }
        const double average_gain = gain_sum / static_cast<double>(candidates.size());
        std::optional<Split> best;
        double best_ratio = -1.0;
        // 先过滤低于平均增益的候选，防止极小分母让低收益分裂胜出。
        for (const auto &split : candidates) {
            const double ratio = split.gain / split.split_info;
            if (split.gain + epsilon >= average_gain && ratio > best_ratio + epsilon) {
                best = split;
                best_ratio = ratio;
            }
        }
        return best;
    }

    static std::optional<std::size_t> branch_for(const Cell &value, double threshold,
                                                 const std::vector<std::string> &categories) {
        if (const auto *number = std::get_if<double>(&value)) {
            return *number <= threshold ? 0 : 1;
        }
        if (const auto *category = std::get_if<std::string>(&value)) {
            const auto found = std::lower_bound(categories.begin(), categories.end(), *category);
            if (found != categories.end() && *found == *category) {
                return static_cast<std::size_t>(found - categories.begin());
            }
        }
        return std::nullopt; // 缺失值，或训练时未见过的类别
    }

    std::unique_ptr<Node> build(const Dataset &data, const std::vector<WeightedRow> &rows,
                                std::vector<bool> available, std::size_t depth) const {
        auto node = std::make_unique<Node>();
        node->counts.assign(data.labels.size(), 0.0);
        for (const auto &row : rows) {
            node->counts[data.rows[row.index].label] += row.weight;
        }
        const double weight = total(node->counts);
        if (depth >= options_.max_depth || weight < 2 * options_.min_leaf_weight ||
            entropy(node->counts) <= epsilon) {
            return node;
        }
        // 先判断停止条件，再选分裂；没有正增益时保留多数类叶子。
        const auto split = find_split(data, rows, available, weight);
        if (!split) {
            return node;
        }
        node->feature = split->feature;
        node->threshold = split->threshold;
        node->categories = split->categories;
        const double known = total(split->weights);
        for (double branch_weight : split->weights) {
            node->priors.push_back(branch_weight / known);
        }
        std::vector<std::vector<WeightedRow>> partitions(node->priors.size());
        for (const auto &row : rows) {
            const auto branch = branch_for(data.rows[row.index].values[split->feature],
                                           split->threshold, split->categories);
            if (branch) {
                partitions[*branch].push_back(row);
            } else {
                // 缺失样本按已知分支比例分配，所有分支的权重之和保持不变。
                for (std::size_t j = 0; j < partitions.size(); ++j) {
                    partitions[j].push_back({row.index, row.weight * node->priors[j]});
                }
            }
        }
        // 离散特征已经按所有类别分支，不再复用；连续特征仍可继续分裂。
        if (data.features[split->feature].kind == Kind::discrete) {
            available[split->feature] = false;
        }
        for (const auto &partition : partitions) {
            node->children.push_back(build(data, partition, available, depth + 1));
        }
        return node;
    }

    // 估计叶子的错误次数上界：零误差修正、少于一次错误时插值，
    // 其余情况用带连续性修正的 Wilson 上界。
    double estimated_errors(double n, double errors) const {
        // n 是样本总权重。zero_errors 是观测到零次错误时的错误次数上界。
        const double zero_errors = n * -std::expm1(std::log(options_.confidence) / n);
        if (errors < 1.0) {
            if (n <= 1.0) {
                return std::min(n, zero_errors + errors * (n - zero_errors));
            }
            return zero_errors + errors * (estimated_errors(n, 1.0) - zero_errors);
        }
        if (errors + 0.5 >= n) {
            return n;
        }
        // 求单侧正态分位数 z：P(Z > z) = confidence。
        double lo = 0.0;
        double hi = 8.0;
        for (int i = 0; i < 60; ++i) {
            const double mid = std::midpoint(lo, hi);
            if (0.5 * std::erfc(mid / std::sqrt(2.0)) > options_.confidence) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        const double z = std::midpoint(lo, hi);
        const double z2 = z * z;
        // 加 0.5 做连续性修正，然后代入 Wilson 上界。
        const double f = (errors + 0.5) / n;
        return n * (f + z2 / (2 * n) + z * std::sqrt(f * (1 - f) / n + z2 / (4 * n * n))) /
               (1 + z2 / n);
    }

    double prune(Node &node) const {
        const double n = total(node.counts);
        const double errors = n - *std::max_element(node.counts.begin(), node.counts.end());
        const double leaf_errors = estimated_errors(n, std::max(0.0, errors));
        if (node.is_leaf()) {
            return leaf_errors;
        }
        double subtree_errors = 0.0;
        for (auto &child : node.children) {
            subtree_errors += prune(*child);
        }
        // 若多数类叶子的估计错误更少，就释放子树并替换为叶子。
        if (leaf_errors <= subtree_errors + epsilon) {
            node.feature.reset();
            node.children.clear();
            node.categories.clear();
            node.priors.clear();
            return leaf_errors;
        }
        return subtree_errors;
    }

    std::vector<double> predict_at(const Node &node, std::span<const Cell> values) const {
        if (node.is_leaf()) {
            auto probabilities = node.counts;
            const double n = total(probabilities);
            for (double &probability : probabilities) {
                probability /= n;
            }
            return probabilities;
        }
        const auto branch = branch_for(values[*node.feature], node.threshold, node.categories);
        if (branch) {
            return predict_at(*node.children[*branch], values);
        }
        // 无法确定分支时，按训练时的分支比例混合所有子树的概率。
        std::vector<double> probabilities(node.counts.size(), 0.0);
        for (std::size_t j = 0; j < node.children.size(); ++j) {
            const auto child = predict_at(*node.children[j], values);
            for (std::size_t k = 0; k < probabilities.size(); ++k) {
                probabilities[k] += node.priors[j] * child[k];
            }
        }
        return probabilities;
    }

    void print_node(std::ostream &out, const Node &node, const std::string &indent) const {
        const auto majority = static_cast<std::size_t>(
            std::max_element(node.counts.begin(), node.counts.end()) - node.counts.begin());
        if (node.is_leaf()) {
            out << indent << "=> " << labels_[majority] << " (weight=" << total(node.counts)
                << ")\n";
            return;
        }
        for (std::size_t j = 0; j < node.children.size(); ++j) {
            out << indent << features_[*node.feature].name;
            if (node.categories.empty()) {
                out << (j == 0 ? " <= " : " > ")
                    << std::setprecision(std::numeric_limits<double>::max_digits10)
                    << node.threshold;
            } else {
                out << " = " << node.categories[j];
            }
            out << '\n';
            print_node(out, *node.children[j], indent + "  ");
        }
    }
};

} // namespace c45

// 3. 单机评测与命令行演示
namespace {
using Clock = std::chrono::steady_clock;

struct Shape {
    std::size_t leaves = 0;
    std::size_t depth = 0;
};

Shape shape(const c45::Node &node) {
    if (node.is_leaf()) {
        return {1, 0};
    }
    Shape result;
    for (const auto &child : node.children) {
        const auto below = shape(*child);
        result.leaves += below.leaves;
        result.depth = std::max(result.depth, below.depth + 1);
    }
    return result;
}

double mean(const std::vector<double> &values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

double deviation(const std::vector<double> &values) {
    const double average = mean(values);
    double sum = 0;
    for (double value : values) {
        sum += (value - average) * (value - average);
    }
    return std::sqrt(sum / static_cast<double>(values.size()));
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto mid = values.size() / 2;
    return values.size() % 2 ? values[mid] : std::midpoint(values[mid - 1], values[mid]);
}

using Confusion = std::vector<std::vector<std::size_t>>;

double macro_f1(const Confusion &matrix) {
    double sum = 0;
    for (std::size_t k = 0; k < matrix.size(); ++k) {
        std::size_t actual = 0;
        std::size_t predicted = 0;
        for (std::size_t j = 0; j < matrix.size(); ++j) {
            actual += matrix[k][j];
            predicted += matrix[j][k];
        }
        if (actual + predicted) {
            sum +=
                2.0 * static_cast<double>(matrix[k][k]) / static_cast<double>(actual + predicted);
        }
    }
    return sum / static_cast<double>(matrix.size());
}

// 固定 Fisher–Yates 洗牌的取样方式，使划分不受标准库实现差异影响。
// 丢弃不能均分到 i 组的末端随机数，避免直接取模带来的偏差。
void shuffle(std::vector<std::size_t> &values, std::mt19937 &random) {
    for (std::size_t i = values.size(); i > 1; --i) {
        const std::uint64_t range = std::uint64_t{1} << 32;
        const std::uint64_t limit = range - range % i;
        std::uint64_t value;
        do {
            value = random();
        } while (value >= limit);
        std::swap(values[i - 1], values[static_cast<std::size_t>(value % i)]);
    }
}

std::vector<std::vector<std::size_t>> stratified_folds(const c45::Dataset &data,
                                                       std::uint32_t seed) {
    constexpr std::size_t count = 5;
    std::vector<std::vector<std::size_t>> groups(data.labels.size());
    std::vector<std::vector<std::size_t>> folds(count);
    for (std::size_t i = 0; i < data.rows.size(); ++i) {
        groups[data.rows[i].label].push_back(i);
    }
    std::mt19937 random(seed);
    std::size_t offset = 0;
    for (auto &group : groups) {
        if (group.size() < count) {
            throw std::invalid_argument("5-fold evaluation requires at least 5 samples per class");
        }
        shuffle(group, random);
        for (std::size_t i = 0; i < group.size(); ++i) {
            folds[(offset + i) % count].push_back(group[i]);
        }
        offset = (offset + group.size()) % count;
    }
    return folds;
}

void benchmark(const std::string &path) {
    constexpr std::size_t repeats = 5;
    constexpr std::size_t prediction_rounds = 50;
    const auto data = c45::load_dataset(path);
    for (bool pruning : {false, true}) {
        c45::Options options;
        options.pruning = pruning;
        std::vector<double> accuracies, f1s;
        std::vector<double> baselines, baseline_f1s;
        std::vector<double> fit_times, predict_times;
        std::vector<double> leaves;
        std::size_t max_depth = 0;
        for (std::uint32_t repeat = 0; repeat < repeats; ++repeat) {
            const auto folds = stratified_folds(data, 2026 + repeat);
            Confusion confusion(data.labels.size(),
                                std::vector<std::size_t>(data.labels.size(), 0));
            Confusion baseline_confusion = confusion;
            std::size_t correct = 0;
            std::size_t baseline_correct = 0;
            std::size_t tested = 0;
            std::vector<bool> evaluated(data.rows.size(), false);
            for (std::size_t fold = 0; fold < folds.size(); ++fold) {
                std::vector<std::size_t> train;
                for (std::size_t j = 0; j < folds.size(); ++j) {
                    if (j != fold) {
                        train.insert(train.end(), folds[j].begin(), folds[j].end());
                    }
                }
                std::vector<std::size_t> counts(data.labels.size(), 0);
                for (auto index : train) {
                    ++counts[data.rows[index].label];
                }
                const auto majority = static_cast<std::size_t>(
                    std::max_element(counts.begin(), counts.end()) - counts.begin());
                // 预热不计时；正式计时只包含 fit，不包含读取和划分。
                {
                    c45::DecisionTree warmup(options);
                    warmup.fit(data, train);
                }
                c45::DecisionTree tree(options);
                const auto begin = Clock::now();
                tree.fit(data, train);
                const auto end = Clock::now();
                fit_times.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
                const auto dimensions = shape(tree.root());
                leaves.push_back(static_cast<double>(dimensions.leaves));
                max_depth = std::max(max_depth, dimensions.depth);
                // 每条样本在本次重复中恰好测试一次；指标由全部折外预测合并计算。
                for (auto index : folds[fold]) {
                    if (evaluated[index] ||
                        std::find(train.begin(), train.end(), index) != train.end()) {
                        throw std::logic_error("cross-validation overlap");
                    }
                    evaluated[index] = true;
                    const auto &row = data.rows[index];
                    const auto prediction = tree.predict(row.values);
                    ++confusion[row.label][prediction];
                    ++baseline_confusion[row.label][majority];
                    correct += prediction == row.label;
                    baseline_correct += majority == row.label;
                    ++tested;
                }
                // 多次预测摊薄计时开销；保留校验和，避免编译器删除计算。
                std::size_t checksum = 0;
                const auto prediction_begin = Clock::now();
                for (std::size_t round = 0; round < prediction_rounds; ++round) {
                    for (auto index : folds[fold]) {
                        checksum += tree.predict(data.rows[index].values);
                    }
                }
                const auto prediction_end = Clock::now();
                static volatile std::size_t prediction_sink = 0;
                prediction_sink = checksum;
                (void)prediction_sink;
                predict_times.push_back(
                    std::chrono::duration<double, std::micro>(prediction_end - prediction_begin)
                        .count() /
                    static_cast<double>(prediction_rounds * folds[fold].size()));
            }
            if (tested != data.rows.size()) {
                throw std::logic_error("cross-validation did not cover every row");
            }
            accuracies.push_back(static_cast<double>(correct) / static_cast<double>(tested));
            baselines.push_back(static_cast<double>(baseline_correct) /
                                static_cast<double>(tested));
            f1s.push_back(macro_f1(confusion));
            baseline_f1s.push_back(macro_f1(baseline_confusion));
        }
        std::cout << std::filesystem::path(path).stem().string() << ',' << data.rows.size() << ','
                  << data.features.size() << ',' << data.labels.size() << ','
                  << (pruning ? "pruned" : "unpruned") << ",5,5,2026," << std::fixed
                  << std::setprecision(6) << mean(accuracies) << ',' << deviation(accuracies) << ','
                  << mean(f1s) << ',' << deviation(f1s) << ',' << mean(baselines) << ','
                  << mean(baseline_f1s) << ',' << median(fit_times) << ',' << median(predict_times)
                  << ',' << mean(leaves) << ',' << max_depth << '\n';
    }
}

int demo(const std::string &path, bool pruning) {
    const auto data = c45::load_dataset(path);
    c45::Options options;
    options.pruning = pruning;
    c45::DecisionTree tree(options);
    tree.fit(data);
    tree.print(std::cout);
    std::size_t correct = 0;
    std::cout << "\nrow,actual,predicted,predicted_probability\n";
    for (std::size_t i = 0; i < data.rows.size(); ++i) {
        const auto probabilities = tree.predict_proba(data.rows[i].values);
        const auto prediction = static_cast<std::size_t>(
            std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin());
        correct += prediction == data.rows[i].label;
        std::cout << i + 1 << ',' << data.labels[data.rows[i].label] << ','
                  << data.labels[prediction] << ',' << std::fixed << std::setprecision(6)
                  << probabilities[prediction] << '\n';
    }
    std::cout << "Training accuracy (not a generalization estimate): " << correct << '/'
              << data.rows.size() << '\n';
    return 0;
}

void usage() {
    std::cout << "Usage:\n  c45 [dataset.dat] [--no-prune]\n"
                 "  c45 --benchmark dataset.dat [dataset.dat ...]\n"
                 "  c45 --help\n\nDefault dataset: Golf.dat relative to the working directory.\n"
                 "Benchmark: stratified 5-fold CV, 5 repeats, seeds 2026..2030; CSV on stdout.\n";
}
} // namespace

// 测试复用本文件，只替换 main，避免复制算法或另外维护测试接口。
#ifndef C45_TESTING
int main(int argc, char *argv[]) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--help") {
            if (argc != 2) {
                throw std::invalid_argument("--help takes no additional arguments");
            }
            usage();
            return 0;
        }
        if (argc > 1 && std::string(argv[1]) == "--benchmark") {
            if (argc < 3) {
                throw std::invalid_argument("--benchmark needs at least one dataset");
            }
            std::cout << "dataset,n,features,classes,model,folds,repeats,seed,accuracy_mean,"
                         "accuracy_sd,macro_f1_mean,macro_f1_sd,"
                         "majority_accuracy,majority_macro_f1,fit_ms_median,predict_us_per_row_"
                         "median,leaves_mean,depth_max\n";
            for (int i = 2; i < argc; ++i) {
                benchmark(argv[i]);
            }
            return 0;
        }
        std::string path = "Golf.dat";
        bool pruning = true;
        bool path_set = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--no-prune") {
                pruning = false;
            } else if (arg.starts_with("--") || path_set) {
                throw std::invalid_argument("invalid argument: " + arg);
            } else {
                path = arg;
                path_set = true;
            }
        }
        return demo(path, pruning);
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
#endif
