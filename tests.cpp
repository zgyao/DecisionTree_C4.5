#define C45_TESTING
#include "DecisionTree_C4.5.cpp"

#include <functional>

namespace {
void check(bool condition, const std::string &message = "check failed") {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near(double actual, double expected, double tolerance = 1e-10) {
    check(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
          "expected " + std::to_string(expected) + ", got " + std::to_string(actual));
}

template <class Function> void rejects(Function &&function) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception &) {
        rejected = true;
    }
    check(rejected, "invalid input was accepted");
}

c45::Dataset parse(const std::string &text) {
    std::istringstream stream(text);
    return c45::read_dataset(stream);
}

c45::Options grow_options() {
    c45::Options options;
    options.min_leaf_weight = 1;
    options.mdl_correction = false;
    options.pruning = false;
    return options;
}

c45::Dataset numeric(const std::vector<double> &values, const std::vector<std::size_t> &labels) {
    c45::Dataset data{{{"x", c45::Kind::continuous}}, {"A", "B"}, {}};
    for (std::size_t i = 0; i < values.size(); ++i) {
        data.rows.push_back({{values[i]}, labels[i]});
    }
    return data;
}

void conservation(const c45::Node &node) {
    if (node.is_leaf()) {
        return;
    }
    near(std::accumulate(node.priors.begin(), node.priors.end(), 0.0), 1.0);
    for (std::size_t k = 0; k < node.counts.size(); ++k) {
        double sum = 0;
        for (const auto &child : node.children) {
            sum += child->counts[k];
        }
        near(sum, node.counts[k]);
    }
    for (const auto &child : node.children) {
        conservation(*child);
    }
}
} // namespace

int main() {
    std::size_t passed = 0, failed = 0;
    const auto test = [&](const std::string &name, const std::function<void()> &run) {
        try {
            run();
            ++passed;
            std::cout << "PASS " << name << '\n';
        } catch (const std::exception &error) {
            ++failed;
            std::cerr << "FAIL " << name << ": " << error.what() << '\n';
        }
    };
    test("entropy uses class proportions and handles zeros", [] {
        near(c45::entropy(std::vector<double>{}), 0);
        near(c45::entropy(std::vector<double>{0, 5}), 0);
        near(c45::entropy(std::vector<double>{9, 5}), 0.9402859586706311);
        near(c45::entropy(std::vector<double>{2, 2, 2}), std::log2(3.0));
        near(c45::entropy(std::vector<double>{0.5, 0.5}), 1);
    });
    test("parser handles CRLF, whitespace, blank lines and short values", [] {
        const auto data =
            parse(" Continuous , Discrete , Discrete\r\nx, a, class\r\n1, y, A\r\n\n2, n, B\r\n\n");
        check(data.rows.size() == 2 && data.features.size() == 2);
        check(std::get<std::string>(data.rows[0].values[1]) == "y");
        near(std::get<double>(data.rows[1].values[0]), 2);
    });
    test("parser supports explicit and empty missing features", [] {
        const auto data = parse("Continuous,Discrete,Discrete\nx,y,class\n?,,A\n1,b,B");
        check(std::holds_alternative<std::monostate>(data.rows[0].values[0]));
        check(std::holds_alternative<std::monostate>(data.rows[0].values[1]));
    });
    test("parser rejects malformed schema and records with line context", [] {
        for (const auto &text :
             {"", "Unknown,Discrete\nx,c\n1,A", "Continuous,Continuous\nx,c\n1,A",
              "Continuous,Discrete\nx,x\n1,A", "Continuous,Discrete\nx,c\n1,2,A",
              "Continuous,Discrete\nx,c\n1,?", "Continuous,Discrete\nx,c\n1,",
              "Continuous,Discrete\nx,c\n\"1\",A"}) {
            rejects([&] { (void)parse(text); });
        }
        try {
            (void)parse("Continuous,Discrete\nx,c\n1bad,A");
            check(false);
        } catch (const std::invalid_argument &error) {
            check(std::string(error.what()).find("line 3:") == 0);
        }
        rejects([] { (void)c45::load_dataset("build/no-such-dataset-for-test"); });
    });
    test("parser rejects nonfinite, overflowing and partially numeric values", [] {
        for (const auto &value : {"nan", "inf", "-inf", "1e999", "3.2kg"}) {
            rejects([&] { (void)parse(std::string("Continuous,Discrete\nx,c\n") + value + ",A"); });
        }
    });
    test("original Golf example learns all 14 training cases", [] {
        const auto data = c45::load_dataset("Golf.dat");
        check(data.rows.size() == 14);
        c45::Options options;
        options.pruning = false;
        c45::DecisionTree tree(options);
        tree.fit(data);
        check(!tree.root().is_leaf() && *tree.root().feature == 0);
        for (const auto &row : data.rows) {
            check(tree.predict(row.values) == row.label);
        }
        conservation(tree.root());
    });
    test("threshold equality goes left; duplicate values are never separated", [] {
        const auto data = numeric({1, 1, 3, 3}, {0, 0, 1, 1});
        c45::DecisionTree tree(grow_options());
        tree.fit(data);
        near(tree.root().threshold, 2);
        check(tree.predict(std::vector<c45::Cell>{2.0}) == 0);
        check(tree.predict(std::vector<c45::Cell>{std::nextafter(2.0, 3.0)}) == 1);
    });
    test("numeric midpoint is finite at extremes and works for adjacent doubles", [] {
        for (const auto &bounds :
             {std::pair{-1e308, 1e308}, std::pair{1.0, std::nextafter(1.0, 2.0)}}) {
            const auto data = numeric({bounds.first, bounds.second}, {0, 1});
            c45::DecisionTree tree(grow_options());
            tree.fit(data);
            check(std::isfinite(tree.root().threshold));
            for (const auto &row : data.rows) {
                check(tree.predict(row.values) == row.label);
            }
        }
    });
    test("continuous feature can split again farther down the tree", [] {
        const auto data = numeric({1, 2, 3, 4, 5, 6}, {0, 0, 1, 1, 0, 0});
        c45::DecisionTree tree(grow_options());
        tree.fit(data);
        check(shape(tree.root()).depth == 2);
        for (const auto &row : data.rows) {
            check(tree.predict(row.values) == row.label);
        }
    });
    test("average-gain filter rejects a rare low-gain high-ratio feature", [] {
        c45::Dataset data{
            {{"rare", c45::Kind::discrete}, {"useful", c45::Kind::discrete}}, {"A", "B"}, {}};
        for (std::size_t i = 0; i < 100; ++i) {
            const auto label = (i < 30 || (i >= 50 && i < 70)) ? 0u : 1u;
            data.rows.push_back(
                {{std::string(i == 0 ? "rare" : "common"), std::string(i < 50 ? "left" : "right")},
                 label});
        }
        c45::DecisionTree tree(grow_options());
        tree.fit(data);
        check(*tree.root().feature == 1);
    });
    test("numeric MDL correction prevents noisy threshold memorization", [] {
        const auto data = numeric({1, 2, 3, 4, 5, 6, 7, 8}, {0, 1, 0, 1, 0, 1, 0, 1});
        auto options = grow_options();
        c45::DecisionTree uncorrected(options);
        uncorrected.fit(data);
        options.mdl_correction = true;
        c45::DecisionTree corrected(options);
        corrected.fit(data);
        check(!uncorrected.root().is_leaf());
        check(corrected.root().is_leaf());
    });
    test("issue #4: numeric gain ratio uses branch entropy, not raw-value entropy", [] {
        // Both features partition 4 A / 4 B perfectly: gain=1, split_info=1.
        // H(raw numeric values)=3 would incorrectly make the second feature win.
        c45::Dataset data{
            {{"number", c45::Kind::continuous}, {"category", c45::Kind::discrete}}, {"A", "B"}, {}};
        for (std::size_t i = 0; i < 8; ++i) {
            data.rows.push_back(
                {{static_cast<double>(i), std::string(i < 4 ? "left" : "right")}, i < 4 ? 0u : 1u});
        }
        c45::DecisionTree tree(grow_options());
        tree.fit(data);
        check(*tree.root().feature == 0,
              "identical partitions must tie regardless of raw feature cardinality");
        near(tree.root().threshold, 3.5);
    });
    test("issue #3: multiclass probabilities expose the weighted leaf distribution", [] {
        const auto data = parse("Discrete,Discrete\nx,class\na,A\na,B\na,B\na,C\na,C\na,C");
        c45::DecisionTree tree;
        tree.fit(data);
        const auto probabilities = tree.predict_proba(data.rows[0].values);
        check(probabilities.size() == 3);
        near(probabilities[0], 1.0 / 6);
        near(probabilities[1], 2.0 / 6);
        near(probabilities[2], 0.5);
        check(tree.predict(data.rows[0].values) == 2);
    });
    test("missing training values conserve fractional class weights", [] {
        const auto data = parse("Discrete,Discrete\nx,class\na,A\na,A\nb,B\nb,B\n?,A");
        c45::DecisionTree tree(grow_options());
        tree.fit(data);
        conservation(tree.root());
        near(tree.root().children[0]->counts[0], 2.5);
        near(tree.root().children[1]->counts[0], 0.5);
        near(tree.root().children[1]->counts[1], 2.0);
        const auto missing = tree.predict_proba(std::vector<c45::Cell>{std::monostate{}});
        near(missing[0], 0.6);
        near(missing[1], 0.4);
        const auto unseen = tree.predict_proba(std::vector<c45::Cell>{std::string("new")});
        near(unseen[0], missing[0]);
        near(unseen[1], missing[1]);
    });
    test("missing numeric features use the same probabilistic routing", [] {
        auto data = numeric({0, 0, 2, 2}, {0, 0, 1, 1});
        data.rows.push_back({{std::monostate{}}, 1});
        c45::DecisionTree tree(grow_options());
        tree.fit(data);
        conservation(tree.root());
        const auto probabilities = tree.predict_proba(std::vector<c45::Cell>{std::monostate{}});
        near(probabilities[0], 0.4);
        near(probabilities[1], 0.6);
    });
    test("all-missing and constant features terminate with majority class", [] {
        for (const auto &values : {"?,A\n?,B\n?,B", "a,A\na,B\na,B"}) {
            const auto data = parse(std::string("Discrete,Discrete\nx,class\n") + values);
            c45::DecisionTree tree(grow_options());
            tree.fit(data);
            check(tree.root().is_leaf());
            check(tree.predict(data.rows[0].values) == 1);
        }
    });
    test("pure classes, depth limits and minimum branch weights stop growth", [] {
        auto options = grow_options();
        c45::DecisionTree pure(options);
        pure.fit(numeric({1, 2}, {1, 1}));
        check(pure.root().is_leaf());
        options.max_depth = 0;
        c45::DecisionTree shallow(options);
        shallow.fit(numeric({1, 2}, {0, 1}));
        check(shallow.root().is_leaf());
        options.max_depth = 64;
        options.min_leaf_weight = 2;
        c45::DecisionTree small(options);
        small.fit(numeric({1, 2}, {0, 1}));
        check(small.root().is_leaf());
    });
    test("ties are deterministic and refitting replaces the previous tree", [] {
        c45::DecisionTree tree(grow_options());
        tree.fit(numeric({1, 1}, {0, 1}));
        check(tree.predict(std::vector<c45::Cell>{1.0}) == 0);
        tree.fit(numeric({1, 2}, {1, 1}));
        check(tree.predict(std::vector<c45::Cell>{1.0}) == 1);
        rejects([&] { tree.fit(c45::Dataset{}); });
        check(tree.predict(std::vector<c45::Cell>{1.0}) == 1);
    });
    test("post-pruning removes memorized noise but keeps a strong split", [] {
        const auto noise =
            parse("Discrete,Discrete\nx,class\na,A\nb,A\nc,A\nd,A\ne,A\nf,A\ng,A\nh,B");
        auto options = grow_options();
        c45::DecisionTree grown(options);
        grown.fit(noise);
        options.pruning = true;
        c45::DecisionTree pruned(options);
        pruned.fit(noise);
        check(shape(grown.root()).leaves == 8);
        check(pruned.root().is_leaf());
        pruned.fit(numeric({1, 2, 3, 4, 5, 6, 7, 8}, {0, 0, 0, 0, 1, 1, 1, 1}));
        check(!pruned.root().is_leaf());
    });
    test("held-out values and labels do not influence fitting", [] {
        auto data = numeric({1, 2, 3, 4, 1000}, {0, 0, 1, 1, 0});
        const std::vector<std::size_t> train{0, 1, 2, 3};
        c45::DecisionTree first(grow_options());
        first.fit(data, train);
        data.rows.back() = {{-1e200}, 1};
        c45::DecisionTree second(grow_options());
        second.fit(data, train);
        std::ostringstream a, b;
        first.print(a);
        second.print(b);
        check(a.str() == b.str());
        rejects([&] { first.fit(data, std::vector<std::size_t>{0, 0}); });
        rejects([&] { first.fit(data, std::vector<std::size_t>{999}); });
        rejects([&] { first.fit(data, std::vector<std::size_t>{}); });
    });
    test("public API validates types, dimensions, options and fit state", [] {
        c45::DecisionTree tree;
        rejects([&] { (void)tree.predict(std::vector<c45::Cell>{1.0}); });
        auto data = numeric({1, 2, 3, 4}, {0, 0, 1, 1});
        tree.fit(data);
        rejects([&] { (void)tree.predict(std::vector<c45::Cell>{}); });
        rejects([&] { (void)tree.predict(std::vector<c45::Cell>{std::string("x")}); });
        rejects([&] {
            (void)tree.predict(std::vector<c45::Cell>{std::numeric_limits<double>::quiet_NaN()});
        });
        data.rows[0].label = 999;
        rejects([&] { tree.fit(data); });
        for (double confidence : {0.0, 0.6, std::numeric_limits<double>::quiet_NaN()}) {
            c45::Options options;
            options.confidence = confidence;
            rejects([&] { c45::DecisionTree invalid(options); });
        }
    });
    test("fixed-seed stratification covers every row once without leakage", [] {
        const auto data = c45::load_dataset("data/iris.dat");
        const auto folds = stratified_folds(data, 2026);
        check(folds == stratified_folds(data, 2026));
        check(folds != stratified_folds(data, 2027));
        std::vector<unsigned> appearances(data.rows.size(), 0);
        for (const auto &fold : folds) {
            std::vector<unsigned> counts(data.labels.size(), 0);
            for (auto index : fold) {
                ++appearances[index];
                ++counts[data.rows[index].label];
            }
            for (auto count : counts) {
                check(count == 10);
            }
        }
        for (auto count : appearances) {
            check(count == 1);
        }
        rejects([] { (void)stratified_folds(numeric({1, 2}, {0, 1}), 2026); });
    });
    test("evaluation metrics match hand-calculated multiclass cases", [] {
        near(macro_f1({{8, 2}, {1, 9}}), (16.0 / 19 + 18.0 / 21) / 2);
        near(macro_f1({{0, 0}, {0, 4}}), 0.5);
        near(mean({1, 2, 3}), 2);
        near(median({1, 2, 3, 4}), 2.5);
        near(deviation({1, 3}), 1);
    });
    test("demo and benchmark output are executable end-to-end", [] {
        std::ostringstream output;
        auto *previous = std::cout.rdbuf(output.rdbuf());
        try {
            usage();
            demo("Golf.dat", false);
            benchmark("data/iris.dat");
        } catch (...) {
            std::cout.rdbuf(previous);
            throw;
        }
        std::cout.rdbuf(previous);
        check(output.str().find("14/14") != std::string::npos);
        check(output.str().find("row,actual,predicted,predicted_probability") != std::string::npos);
        check(output.str().find("1,Don't Play,Don't Play,1.000000") != std::string::npos);
        check(output.str().find("iris,150,4,3,pruned,5,5,2026,") != std::string::npos);
    });
    std::cout << passed << " passed, " << failed << " failed\n";
    return failed ? 1 : 0;
}
