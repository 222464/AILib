cmake_minimum_required(VERSION 3.0)

project(AILib)

# Set options
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
set(SFML_STATIC_LIBS FALSE CACHE BOOL "Choose whether SFML is linked statically or shared.")
set(AILIB_STATIC_STD_LIBS FALSE CACHE BOOL "Use statically linked standard/runtime libraries? This option must match the one used for SFML.")

# Make sure that the runtime library gets link statically
if(AILIB_STATIC_STD_LIBS)
	if(NOT SFML_STATIC_LIBS)
		message("\n-> If you check AILIB_STATIC_STD_LIBS, you also need to check SFML_STATIC_LIBRARIES.")
		message("-> It would lead to multiple runtime environments which result in undefined behavior.\n")
	elseif(WIN32 AND MSVC)
		# Change all MSVC compiler flags to /MT
		foreach(flag CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE)
			if(${flag} MATCHES "/MD")
			string(REGEX REPLACE "/MD" "/MT" ${flag} "${${flag}}")
			endif()
		endforeach()
	elseif(CMAKE_COMPILER_IS_GNUCXX)
		# Note: Doesn't work for TDM compiler, since it's compiling the runtime libs statically by default
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static")
	endif()
endif()

# AILib uses C++11 features
if(CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

# Add directory containing FindSFML.cmake to module path
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/Extlibs/SFML/cmake/Modules/;${CMAKE_MODULE_PATH}")

# Make sure that FindSFML.cmake searches for the static libraries
if(SFML_STATIC_LIBS)
	set(SFML_STATIC_LIBRARIES TRUE)
endif()

# Add eigen include directory
set(INCLUDE_DIR "Extlibs/eigen/")

# Find SFML
find_package(SFML 2 COMPONENTS graphics window audio system)

# Output an error if SFML wasn't found
if(SFML_FOUND)
	set(INCLUDE_DIR ${INCLUDE_DIR} ${SFML_INCLUDE_DIR})
else()
	set(SFML_ROOT "" CACHE PATH "SFML top-level directory")
	message("\n-> SFML directory not found. Set SFML_ROOT to SFML's top-level path (containing \"include\" and \"lib\" directories).")
	message("-> Make sure the SFML libraries with the same configuration (Release/Debug, Static/Dynamic) exist.\n")
endif()

# Add the source directory to the include directories
set(INCLUDE_DIR ${INCLUDE_DIR} Source)

include_directories(${INCLUDE_DIR})

set(SRC_DIR Source)

# Add the source files
set(AILIB_SRC
	${SRC_DIR}/apssc/APSSC.cpp
	${SRC_DIR}/ctrnn/CTRNN.cpp
	${SRC_DIR}/ctrnn/GeneticAlgorithm.cpp
	${SRC_DIR}/deep/ConvNet2D.cpp
	${SRC_DIR}/deep/DAutoEncoder.cpp
	${SRC_DIR}/deep/AutoLSTM.cpp
	${SRC_DIR}/deep/DBN.cpp
	${SRC_DIR}/deep/EQNAC.cpp
	${SRC_DIR}/deep/FA.cpp
	${SRC_DIR}/deep/FERL.cpp
	${SRC_DIR}/deep/RBM.cpp
	${SRC_DIR}/deep/DSOM.cpp
	${SRC_DIR}/dnf/Field.cpp
	${SRC_DIR}/elman/ElmanNetwork.cpp
	${SRC_DIR}/experiments/ExperimentAND.cpp
	${SRC_DIR}/experiments/ExperimentOR.cpp
	${SRC_DIR}/experiments/ExperimentPoleBalancing.cpp
	${SRC_DIR}/experiments/ExperimentXOR.cpp
	${SRC_DIR}/falcon/Falcon.cpp
	${SRC_DIR}/htmrl/HTMRL.cpp
	${SRC_DIR}/htmrl/HTMRLDiscreteAction.cpp
	${SRC_DIR}/htm/Cell.cpp
	${SRC_DIR}/htm/Column.cpp
	${SRC_DIR}/htm/Connection.cpp
	${SRC_DIR}/htm/Region.cpp
	${SRC_DIR}/htm/Segment.cpp
	${SRC_DIR}/hypernet/BayesianOptimizer.cpp
	${SRC_DIR}/hypernet/BayesianOptimizerTrainer.cpp
	${SRC_DIR}/hypernet/Boid.cpp
	${SRC_DIR}/hypernet/Config.cpp
	${SRC_DIR}/hypernet/Decoder.cpp
	${SRC_DIR}/hypernet/Encoder.cpp
	${SRC_DIR}/hypernet/EvolutionaryAlgorithm.cpp
	${SRC_DIR}/hypernet/Experiment.cpp
	${SRC_DIR}/hypernet/FunctionApproximator.cpp
	${SRC_DIR}/hypernet/HyperNet.cpp
	${SRC_DIR}/hypernet/Link.cpp
	${SRC_DIR}/hypernet/Orchestrator.cpp
	${SRC_DIR}/hypernet/SampleField.cpp
	${SRC_DIR}/hypernet/EvolutionaryTrainer.cpp
	${SRC_DIR}/lstmrl/LSTMRL.cpp
	${SRC_DIR}/lstm/LSTMActorCritic.cpp
	${SRC_DIR}/lstm/LSTMG.cpp
	${SRC_DIR}/lstm/LSTMNet.cpp
	${SRC_DIR}/Maze.cpp
	${SRC_DIR}/nn/MultiQ.cpp
	${SRC_DIR}/nn/NCPSOAgent.cpp
	${SRC_DIR}/nn/PSOAgent.cpp
	${SRC_DIR}/nn/SOM.cpp
	${SRC_DIR}/Main.cpp
	${SRC_DIR}/nn/ActorCriticAgent.cpp
	${SRC_DIR}/nn/BrownianPerturbation.cpp
	${SRC_DIR}/nn/Cacla.cpp
	${SRC_DIR}/nn/FeedForwardNeuralNetwork.cpp
	${SRC_DIR}/nn/GeneticAlgorithm.cpp
	${SRC_DIR}/nn/MemoryActor.cpp
	${SRC_DIR}/nn/MemoryCell.cpp
	${SRC_DIR}/nn/Neuron.cpp
	${SRC_DIR}/nn/QAgent.cpp
	${SRC_DIR}/nn/RLLSTMAgent.cpp
	${SRC_DIR}/nn/SOMQAgent.cpp
	${SRC_DIR}/nn/TabularQ.cpp
	${SRC_DIR}/raahn/AutoEncoder.cpp
	${SRC_DIR}/raahn/HebbianLearner.cpp
	${SRC_DIR}/raahn/RAAHN.cpp
	${SRC_DIR}/apssc/APSSC.h
	${SRC_DIR}/ctrnn/CTRNN.h
	${SRC_DIR}/ctrnn/GeneticAlgorithm.h
	${SRC_DIR}/deep/ConvNet2D.h
	${SRC_DIR}/deep/DAutoEncoder.h
	${SRC_DIR}/deep/AutoLSTM.h
	${SRC_DIR}/deep/DBN.h
	${SRC_DIR}/deep/EQNAC.h
	${SRC_DIR}/deep/FA.h
	${SRC_DIR}/deep/FERL.h
	${SRC_DIR}/deep/RBM.h
	${SRC_DIR}/deep/DSOM.h
	${SRC_DIR}/htmrl/HTMRL.h
	${SRC_DIR}/htmrl/HTMRLDiscreteAction.h
	${SRC_DIR}/htm/Segment.h
	${SRC_DIR}/dnf/Field.h
	${SRC_DIR}/elman/ElmanNetwork.h
	${SRC_DIR}/experiments/ExperimentAND.h
	${SRC_DIR}/experiments/ExperimentOR.h
	${SRC_DIR}/experiments/ExperimentPoleBalancing.h
	${SRC_DIR}/experiments/ExperimentXOR.h
	${SRC_DIR}/falcon/Falcon.h
	${SRC_DIR}/htm/Cell.h
	${SRC_DIR}/htm/Column.h
	${SRC_DIR}/htm/Connection.h
	${SRC_DIR}/htm/Region.h
	${SRC_DIR}/hypernet/BayesianOptimizer.h
	${SRC_DIR}/hypernet/BayesianOptimizerTrainer.h
	${SRC_DIR}/hypernet/Boid.h
	${SRC_DIR}/hypernet/Config.h
	${SRC_DIR}/hypernet/Decoder.h
	${SRC_DIR}/hypernet/Encoder.h
	${SRC_DIR}/hypernet/EvolutionaryAlgorithm.h
	${SRC_DIR}/hypernet/Experiment.h
	${SRC_DIR}/hypernet/FunctionApproximator.h
	${SRC_DIR}/hypernet/HyperNet.h
	${SRC_DIR}/hypernet/Link.h
	${SRC_DIR}/hypernet/Orchestrator.h
	${SRC_DIR}/hypernet/SampleField.h
	${SRC_DIR}/hypernet/EvolutionaryTrainer.h
	${SRC_DIR}/lstmrl/LSTMRL.h
	${SRC_DIR}/lstm/LSTMActorCritic.h
	${SRC_DIR}/lstm/LSTMG.h
	${SRC_DIR}/lstm/LSTMNet.h
	${SRC_DIR}/lstm/TupleHash.h
	${SRC_DIR}/nn/MultiQ.h
	${SRC_DIR}/nn/NCPSOAgent.h
	${SRC_DIR}/nn/PSOAgent.h
	${SRC_DIR}/nn/SOM.h
	${SRC_DIR}/nn/ActorCriticAgent.h
	${SRC_DIR}/nn/BrownianPerturbation.h
	${SRC_DIR}/nn/Cacla.h
	${SRC_DIR}/nn/FeedForwardNeuralNetwork.h
	${SRC_DIR}/nn/GeneticAlgorithm.h
	${SRC_DIR}/nn/MemoryActor.h
	${SRC_DIR}/nn/MemoryCell.h
	${SRC_DIR}/nn/Neuron.h
	${SRC_DIR}/nn/QAgent.h
	${SRC_DIR}/nn/RLLSTMAgent.h
	${SRC_DIR}/nn/Sensor.h
	${SRC_DIR}/nn/SOMQAgent.h
	${SRC_DIR}/nn/TabularQ.h
	${SRC_DIR}/raahn/AutoEncoder.h
	${SRC_DIR}/raahn/HebbianLearner.h
	${SRC_DIR}/raahn/RAAHN.h
)

# Tell CMake to build a executable
add_executable(AILib ${AILIB_SRC})

# Link SFML
target_link_libraries(AILib ${SFML_LIBRARIES} ${SFML_DEPENDENCIES})

# Install executable
install(TARGETS AILib
		RUNTIME DESTINATION .)

# Install resources
install(DIRECTORY Resources/
		DESTINATION Resources/)
		
# Install "config" files
install(FILES license.txt README.md
		DESTINATION .)