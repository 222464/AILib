/*
AI Lib
Copyright (C) 2014 Eric Laukien

This software is provided 'as-is', without any express or implied
warranty.  In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
claim that you wrote the original software. If you use this software
in a product, an acknowledgment in the product documentation would be
appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#pragma once

#include <vector>
#include <random>

namespace chtm {
	class CHTMRegion {
	public:
		struct LateralConnection {
			float _weight;
		};

		struct InputConnection {
			float _weight;
			float _width;
		};

		struct ReconConnection {
			float _weight;
		};

		struct OutputConnection {
			float _weight;

			float _eligibility;

			OutputConnection()
				: _eligibility(0.0f)
			{}
		};

		struct ColumnCell {
			float _activation;
			float _state;
			float _statePrev;
			float _predictionState;
			float _predictionStatePrev;
			float _prePrediction;
			float _prediction;
			float _predictionPrev;
			float _perturbedPrediction;
			float _perturbedPredictionPrev;

			float _intent;

			std::vector<LateralConnection> _connections;

			LateralConnection _bias;

			ColumnCell()
				: _activation(0.0f), _state(0.0f), _statePrev(0.0f), _predictionState(0.0f), _predictionStatePrev(0.0f),
				_prePrediction(0.0f), _prediction(0.0f), _predictionPrev(0.0f),
				_perturbedPrediction(0.0f), _perturbedPredictionPrev(0.0f),
				_intent(0.0f)
			{}
		};

		struct Column {
			std::vector<InputConnection> _center;

			float _activation;
			float _state;
			float _prediction;
			float _predictionPrev;
			float _perturbedPrediction;
			float _perturbedPredictionPrev;
			float _intent;
			float _action;
			float _output;

			std::vector<ColumnCell> _cells;

			Column()
				: _activation(0.0f), _state(0.0f), _prediction(0.0f), _predictionPrev(0.0f),
				_perturbedPrediction(0.0f), _perturbedPredictionPrev(0.0f), _output(0.0f), _intent(0.0f), _action(0.0f)
			{}
		};

		struct ReconNode {
			std::vector<ReconConnection> _connections;

			ReconConnection _bias;
		};

		struct OutputNode {
			std::vector<OutputConnection> _connections;

			OutputConnection _bias;
		};

		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

	private:
		std::vector<ReconNode> _reconNodes;
		std::vector<Column> _columns;
		std::vector<OutputNode> _outputNodes;

		int _inputWidth, _inputHeight;
		int _columnsWidth, _columnsHeight;
		int _cellsPerColumn;
		int _receptiveRadius;
		int _cellRadius;

	public:
		void createRandom(int inputWidth, int inputHeight, int columnsWidth, int columnsHeight, int cellsPerColumn, int receptiveRadius, int cellRadius, int numOutputs,
			float minCenter, float maxCenter, float minWidth, float maxWidth, float minInputWeight, float maxInputWeight, float minReconWeight, float maxReconWeight,
			float minCellWeight, float maxCellWeight, float minOutputWeight, float maxOutputWeight, std::mt19937 &generator);

		void stepBegin();

		void getOutput(const std::vector<float> &input, std::vector<float> &output, int inhibitionRadius, float sparsity, float cellIntensity, float predictionIntensity, std::mt19937 &generator);
	
		void getOutputAction(const std::vector<float> &input, std::vector<float> &output, std::vector<float> &action, float indecisivnessIntensity, float perturbationIntensity, float predictionSparsity, float intentIntensity, int inhibitionRadius, float sparsity, float cellIntensity, float predictionIntensity, int optimizationSteps, float optimizationAlpha, float annealingPerturbationStdDev, float annealingPerturbationDecay, float reconAlpha, std::mt19937 &generator);

		void learn(const std::vector<float> &input, const std::vector<float> &output, const std::vector<float> &target, float weightAlpha, float reconAlpha, float centerAlpha, float widthAlpha, float widthScalar, float minDistance, float minLearningThreshold, float cellAlpha);

		void learnTraces(const std::vector<float> &input, const std::vector<float> &output, const std::vector<float> &error, const std::vector<float> &outputWeightAlphas, float centerAlpha, float widthAlpha, float widthScalar, float minDistance, float minLearningThreshold, float cellAlpha, float perturbationIntensity, const std::vector<float> &outputLambdas);

		void findInput(const std::vector<float> &input, const std::vector<float> &output, std::vector<float> &newInput, float cellIntensity, int optimizationSteps, float optimizationAlpha, float annealingPerturbationStdDev, float annealingPerturbationDecay, std::mt19937 &generator);

		int getNumOutputs() const {
			return _outputNodes.size();
		}

		int getNumColumns() const {
			return _columns.size();
		}

		const Column &getColumn(int x, int y) const {
			return _columns[x + y * _columnsWidth];
		}
	};
}