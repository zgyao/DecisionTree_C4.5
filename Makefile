CXX ?= g++
CPPFLAGS ?=
CXXFLAGS ?= -O2 -g
WARNINGS = -Wall -Wextra -Wpedantic -Wconversion -Wshadow
STANDARD = -std=c++20
BUILD ?= build

.PHONY: all test sanitize benchmark clean
all: $(BUILD)/c45 $(BUILD)/tests

$(BUILD):
	mkdir -p $@

$(BUILD)/c45: DecisionTree_C4.5.cpp | $(BUILD)
	$(CXX) $(CPPFLAGS) $(STANDARD) $(WARNINGS) $(CXXFLAGS) $< $(LDFLAGS) -o $@

$(BUILD)/tests: tests.cpp DecisionTree_C4.5.cpp | $(BUILD)
	$(CXX) $(CPPFLAGS) $(STANDARD) $(WARNINGS) $(CXXFLAGS) $< $(LDFLAGS) -o $@

test: all
	./$(BUILD)/tests
	./$(BUILD)/c45 Golf.dat > $(BUILD)/golf-output.txt
	./$(BUILD)/c45 --help > /dev/null
	@status=0; ./$(BUILD)/c45 --unknown > /dev/null 2>&1 || status=$$?; test "$$status" -eq 1
	@status=0; ./$(BUILD)/c45 $(BUILD)/nonexistent-data-file > /dev/null 2>&1 || status=$$?; test "$$status" -eq 1
	@status=0; ./$(BUILD)/c45 --benchmark > /dev/null 2>&1 || status=$$?; test "$$status" -eq 1
	@status=0; ./$(BUILD)/c45 --help extra > /dev/null 2>&1 || status=$$?; test "$$status" -eq 1

sanitize:
	$(MAKE) BUILD=build/sanitize CXXFLAGS='-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all' LDFLAGS='-fsanitize=address,undefined' test

benchmark: $(BUILD)/c45
	python3 scripts/prepare_data.py --check
	./$(BUILD)/c45 --benchmark data/iris.dat data/wine.dat data/breast_cancer.dat

clean:
	rm -rf build
