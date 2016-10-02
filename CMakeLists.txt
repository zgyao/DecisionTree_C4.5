cmake_minimum_required(VERSION 3.4)

set (CMAKE_CXX_STANDARD 11)

add_executable (DecisionTree_C4.5
	./DecisionTree_C4.5.cpp
)

add_custom_command(
	TARGET DecisionTree_C4.5 POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy
	${CMAKE_SOURCE_DIR}/Golf.dat
	${CMAKE_CURRENT_BINARY_DIR}/Golf.dat
)