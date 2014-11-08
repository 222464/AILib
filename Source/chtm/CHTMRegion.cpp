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

#include <chtm/CHTMRegion.h>

#include <algorithm>

#include <iostream>

#include <assert.h>

using namespace chtm;

void CHTMRegion::createRandom(int inputWidth, int inputHeight, int columnsWidth, int columnsHeight, int cellsPerColumn, int receptiveRadius, int cellRadius, int numOutputs,
	float minCenter, float maxCenter, float minWidth, float maxWidth, float minInputWeight, float maxInputWeight, float minReconWeight, float maxReconWeight,
	float minCellWeight, float maxCellWeight, float minOutputWeight, float maxOutputWeight, std::mt19937 &generator)
{
	std::uniform_real_distribution<float> centerDist(minCenter, maxCenter);
	std::uniform_real_distribution<float> widthDist(minWidth, maxWidth);
	std::uniform_real_distribution<float> inputWeightDist(minInputWeight, maxInputWeight);
	std::uniform_real_distribution<float> reconWeightDist(minReconWeight, maxReconWeight);
	std::uniform_real_distribution<float> cellWeightDist(minCellWeight, maxCellWeight);
	std::uniform_real_distribution<float> outputWeightDist(minOutputWeight, maxOutputWeight);

	_inputWidth = inputWidth;
	_inputHeight = inputHeight;

	_columnsWidth = columnsWidth;
	_columnsHeight = columnsHeight;

	_cellsPerColumn = cellsPerColumn;

	_receptiveRadius = receptiveRadius;

	_cellRadius = cellRadius;

	int numInputs = _inputWidth * _inputHeight;
	int numColumns = _columnsWidth * _columnsHeight;
	int numColumnInputWeights = std::pow(receptiveRadius * 2 + 1, 2);
	int numCellConnections = std::pow(cellRadius * 2 + 1, 2) * cellsPerColumn;
	
	_columns.resize(numColumns);

	for (int i = 0; i < _columns.size(); i++) {
		_columns[i]._center.resize(numColumnInputWeights);

		for (int j = 0; j < _columns[i]._center.size(); j++) {
			_columns[i]._center[j]._weight = centerDist(generator);
			_columns[i]._center[j]._width = widthDist(generator);
		}

		_columns[i]._cells.resize(cellsPerColumn);

		for (int j = 0; j < _columns[i]._cells.size(); j++) {
			_columns[i]._cells[j]._connections.resize(numCellConnections);

			_columns[i]._cells[j]._bias._weight = cellWeightDist(generator);

			for (int k = 0; k < _columns[i]._cells[j]._connections.size(); k++)
				_columns[i]._cells[j]._connections[k]._weight = cellWeightDist(generator);
		}
	}

	_outputNodes.resize(numOutputs);

	int connectionsPerOutput = _columns.size() * _cellsPerColumn;

	for (int i = 0; i < _outputNodes.size(); i++) {
		_outputNodes[i]._connections.resize(connectionsPerOutput);

		for (int j = 0; j < _outputNodes[i]._connections.size(); j++)
			_outputNodes[i]._connections[j]._weight = outputWeightDist(generator);

		_outputNodes[i]._bias._weight = outputWeightDist(generator);
	}

	_reconNodes.resize(numInputs);

	for (int i = 0; i < _reconNodes.size(); i++)
		_reconNodes[i]._bias._weight = reconWeightDist(generator);

	float inputWidthInv = 1.0f / _inputWidth;
	float inputHeightInv = 1.0f / _inputHeight;

	float rbfWidthInv = 1.0f / _columnsWidth;
	float rbfHeightInv = 1.0f / _columnsHeight;

	for (int rx = 0; rx < _columnsWidth; rx++)
	for (int ry = 0; ry < _columnsHeight; ry++) {
		int i = rx + ry * _columnsWidth;

		float rxn = rx * rbfWidthInv;
		float ryn = ry * rbfHeightInv;

		for (int dx = -_receptiveRadius; dx <= _receptiveRadius; dx++)
		for (int dy = -_receptiveRadius; dy <= _receptiveRadius; dy++) {
			float xn = rxn + dx * inputWidthInv;
			float yn = ryn + dy * inputHeightInv;

			if (xn >= 0.0f && xn < 1.0f && yn >= 0.0f && yn < 1.0f) {
				int x = xn * _inputWidth;
				int y = yn * _inputHeight;

				int j = x + y * _inputWidth;

				ReconConnection c;

				c._weight = reconWeightDist(generator);

				_reconNodes[j]._connections.push_back(c);
			}
		}
	}
}

void CHTMRegion::stepBegin() {
	for (int i = 0; i < _columns.size(); i++) {
		_columns[i]._predictionPrev = _columns[i]._prediction;
		_columns[i]._perturbedPredictionPrev = _columns[i]._perturbedPrediction;

		for (int j = 0; j < _cellsPerColumn; j++) {
			_columns[i]._cells[j]._statePrev = _columns[i]._cells[j]._state;
			_columns[i]._cells[j]._predictionPrev = _columns[i]._cells[j]._prediction;
			_columns[i]._cells[j]._predictionStatePrev = _columns[i]._cells[j]._predictionState;
			_columns[i]._cells[j]._perturbedPredictionPrev = _columns[i]._cells[j]._perturbedPrediction;
		}
	}
}

void CHTMRegion::getOutput(const std::vector<float> &input, std::vector<float> &output, int inhibitionRadius, float sparsity, float cellIntensity, float predictionIntensity, std::mt19937 &generator) {
	float inputWidthInv = 1.0f / _inputWidth;
	float inputHeightInv = 1.0f / _inputHeight;

	float rbfWidthInv = 1.0f / _columnsWidth;
	float rbfHeightInv = 1.0f / _columnsHeight;

	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

	for (int rx = 0; rx < _columnsWidth; rx++)
	for (int ry = 0; ry < _columnsHeight; ry++) {
		int i = rx + ry * _columnsWidth;

		float rxn = rx * rbfWidthInv;
		float ryn = ry * rbfHeightInv;

		float dist2 = 0.0f;

		int wi = 0;

		for (int dx = -_receptiveRadius; dx <= _receptiveRadius; dx++)
		for (int dy = -_receptiveRadius; dy <= _receptiveRadius; dy++) {
			float xn = rxn + dx * inputWidthInv;
			float yn = ryn + dy * inputHeightInv;

			if (xn >= 0.0f && xn < 1.0f && yn >= 0.0f && yn < 1.0f) {
				int x = xn * _inputWidth;
				int y = yn * _inputHeight;

				int j = x + y * _inputWidth;

				float delta = (input[j] - _columns[i]._center[wi]._weight) * _columns[i]._center[wi]._width;

				dist2 += delta * delta;
			}

			wi++;
		}

		_columns[i]._activation = std::exp(-dist2);
	}

	// Sparsify
	for (int rx = 0; rx < _columnsWidth; rx++)
	for (int ry = 0; ry < _columnsHeight; ry++) {
		int i = rx + ry * _columnsWidth;

		float maximum = 0.0f;
		float average = 0.0f;

		int count = 0;

		for (int dx = -inhibitionRadius; dx <= inhibitionRadius; dx++)
		for (int dy = -inhibitionRadius; dy <= inhibitionRadius; dy++) {
			int x = rx + dx;
			int y = ry + dy;

			if (x >= 0 && x < _columnsWidth && y >= 0 && y < _columnsHeight) {
				int j = x + y * _columnsWidth;

				maximum = std::max(maximum, _columns[j]._activation);

				average += _columns[j]._activation;
				count++;
			}
		}

		average /= count;

		float divisor = maximum - average;

		if (divisor == 0.0f)
			_columns[i]._state = uniformDist(generator) < (1.0f / count) ? 1.0f : 0.0f;
		else
			_columns[i]._state = std::exp((_columns[i]._activation - maximum) / divisor * sparsity);
	}

	for (int rx = 0; rx < _columnsWidth; rx++)
	for (int ry = 0; ry < _columnsHeight; ry++) {
		int i = rx + ry * _columnsWidth;

		float columnState = _columns[i]._state;

		float minPredictionError = 1.0f;

		for (int ci = 0; ci < _cellsPerColumn; ci++) {
			float prediction = _columns[i]._cells[ci]._predictionPrev;

			float predictionError = std::fabs(columnState - prediction);

			minPredictionError = std::min(minPredictionError, predictionError);
		}

		for (int ci = 0; ci < _cellsPerColumn; ci++) {
			float prediction = _columns[i]._cells[ci]._predictionPrev;

			float predictionError = std::fabs(columnState - prediction);

			_columns[i]._cells[ci]._state = std::exp((minPredictionError - predictionError) * cellIntensity) * columnState;
		}
	}

	// Form predictions
	for (int rx = 0; rx < _columnsWidth; rx++)
	for (int ry = 0; ry < _columnsHeight; ry++) {
		int i = rx + ry * _columnsWidth;

		float maxPrediction = 0.0f;

		for (int ci = 0; ci < _cellsPerColumn; ci++) {
			float sum = _columns[i]._cells[ci]._bias._weight;

			// Go through all connections 
			int wi = 0;

			for (int dx = -_cellRadius; dx <= _cellRadius; dx++)
			for (int dy = -_cellRadius; dy <= _cellRadius; dy++)
			for (int cio = 0; cio < _cellsPerColumn; cio++) {
				int cx = rx + dx;
				int cy = ry + dy;

				if (cx >= 0 && cx < _columnsWidth && cy >= 0 && cy < _columnsHeight) {
					float cellWeight = _columns[i]._cells[ci]._connections[wi]._weight;

					float connectionState = _columns[cx + cy * _columnsWidth]._cells[cio]._state;

					sum += cellWeight * connectionState;
				}

				wi++;
			}

			_columns[i]._cells[ci]._prediction = sigmoid(sum * predictionIntensity);

			maxPrediction = std::max(maxPrediction, _columns[i]._cells[ci]._prediction);
		}

		_columns[i]._prediction = maxPrediction;

		_columns[i]._output = std::max(_columns[i]._state, _columns[i]._prediction);
	}

	if (output.size() != _outputNodes.size())
		output.resize(_outputNodes.size());

	for (int i = 0; i < _outputNodes.size(); i++) {
		float sum = _outputNodes[i]._bias._weight;

		for (int j = 0; j < _columns.size(); j++)
		for (int k = 0; k < _cellsPerColumn; k++)
			sum += _columns[j]._cells[k]._state * _outputNodes[i]._connections[k + j * _cellsPerColumn]._weight;

		output[i] = sum;
	}
}

void CHTMRegion::getOutputAction(const std::vector<float> &input, std::vector<float> &output, std::vector<float> &action, float indecisivnessIntensity, float perturbationIntensity, float predictionSparsity, float intentIntensity, int inhibitionRadius, float sparsity, float cellIntensity, float predictionIntensity, int optimizationSteps, float optimizationAlpha, float annealingPerturbationStdDev, float annealingPerturbationDecay, float reconAlpha, std::mt19937 &generator) {
	float inputWidthInv = 1.0f / _inputWidth;
	float inputHeightInv = 1.0f / _inputHeight;

	float rbfWidthInv = 1.0f / _columnsWidth;
	float rbfHeightInv = 1.0f / _columnsHeight;

	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

	std::normal_distribution<float> annealingPerurbationDist(0.0f, annealingPerturbationStdDev);

	for (int rx = 0; rx < _columnsWidth; rx++)
	for (int ry = 0; ry < _columnsHeight; ry++) {
		int i = rx + ry * _columnsWidth;

		float rxn = rx * rbfWidthInv;
		float ryn = ry * rbfHeightInv;

		float dist2 = 0.0f;

		int wi = 0;

		for (int dx = -_receptiveRadius; dx <= _receptiveRadius; dx++)
		for (int dy = -_receptiveRadius; dy <= _receptiveRadius; dy++) {
			float xn = rxn + dx * inputWidthInv;
			float yn = ryn + dy * inputHeightInv;

			if (xn >= 0.0f && xn < 1.0f && yn >= 0.0f && yn < 1.0f) {
				int x = xn * _inputWidth;
				int y = yn * _inputHeight;

				int j = x + y * _inputWidth;

				float delta = (input[j] - _columns[i]._center[wi]._weight) * _columns[i]._center[wi]._width;

				dist2 += delta * delta;
			}

			wi++;
		}

		_columns[i]._activation = std::exp(-dist2);
	}

	// Sparsify
	for (int rx = 0; rx < _columnsWidth; rx++)
	for (int ry = 0; ry < _columnsHeight; ry++) {
		int i = rx + ry * _columnsWidth;

		float maximum = 0.0f;
		float average = 0.0f;

		int count = 0;

		for (int dx = -inhibitionRadius; dx <= inhibitionRadius; dx++)
		for (int dy = -inhibitionRadius; dy <= inhibitionRadius; dy++) {
			int x = rx + dx;
			int y = ry + dy;

			if (x >= 0 && x < _columnsWidth && y >= 0 && y < _columnsHeight) {
				int j = x + y * _columnsWidth;

				maximum = std::max(maximum, _columns[j]._activation);

				average += _columns[j]._activation;
				count++;
			}
		}

		average /= count;

		float divisor = maximum - average;

		if (divisor == 0.0f)
			_columns[i]._state = uniformDist(generator) < (1.0f / count) ? 1.0f : 0.0f;
		else
			_columns[i]._state = std::exp((_columns[i]._activation - maximum) / divisor * sparsity);
	}

	// Determine and correct reconstruction
	std::vector<float> reconNodeSums(_reconNodes.size());
	std::vector<int> reconWeightCounters(_reconNodes.size(), 0);

	for (int i = 0; i < _reconNodes.size(); i++)
		reconNodeSums[i] = _reconNodes[i]._bias._weight;

	for (int rx = 0; rx < _columnsWidth; rx++)
	for (int ry = 0; ry < _columnsHeight; ry++) {
		int i = rx + ry * _columnsWidth;

		float rxn = rx * rbfWidthInv;
		float ryn = ry * rbfHeightInv;

		for (int dx = -_receptiveRadius; dx <= _receptiveRadius; dx++)
		for (int dy = -_receptiveRadius; dy <= _receptiveRadius; dy++) {
			float xn = rxn + dx * inputWidthInv;
			float yn = ryn + dy * inputHeightInv;

			if (xn >= 0.0f && xn < 1.0f && yn >= 0.0f && yn < 1.0f) {
				int x = xn * _inputWidth;
				int y = yn * _inputHeight;

				int j = x + y * _inputWidth;

				reconNodeSums[j] += _reconNodes[j]._connections[reconWeightCounters[j]]._weight * _columns[i]._state;

				reconWeightCounters[j]++;
			}
		}
	}

	// Learn reconstruction
	std::vector<float> reconErrors(_reconNodes.size());

	for (int i = 0; i < _reconNodes.size(); i++) {
		reconErrors[i] = input[i] - reconNodeSums[i];

		_reconNodes[i]._bias._weight += reconAlpha * reconErrors[i];
	}

	reconWeightCounters.clear();
	reconWeightCounters.assign(_reconNodes.size(), 0);

	for (int rx = 0; rx < _columnsWidth; rx++)
	for (int ry = 0; ry < _columnsHeight; ry++) {
		int i = rx + ry * _columnsWidth;

		float rxn = rx * rbfWidthInv;
		float ryn = ry * rbfHeightInv;

		int wi = 0;

		for (int dx = -_receptiveRadius; dx <= _receptiveRadius; dx++)
		for (int dy = -_receptiveRadius; dy <= _receptiveRadius; dy++) {
			float xn = rxn + dx * inputWidthInv;
			float yn = ryn + dy * inputHeightInv;

			if (xn >= 0.0f && xn < 1.0f && yn >= 0.0f && yn < 1.0f) {
				int x = xn * _inputWidth;
				int y = yn * _inputHeight;

				int j = x + y * _inputWidth;

				_reconNodes[j]._connections[reconWeightCounters[j]]._weight += reconAlpha * reconErrors[j] * _columns[i]._state;

				reconWeightCounters[j]++;
			}

			wi++;
		}
	}

	for (int rx = 0; rx < _columnsWidth; rx++)
	for (int ry = 0; ry < _columnsHeight; ry++) {
		int i = rx + ry * _columnsWidth;

		float columnState = _columns[i]._state;

		float minPredictionError = 1.0f;

		for (int ci = 0; ci < _cellsPerColumn; ci++) {
			float prediction = _columns[i]._cells[ci]._predictionPrev;

			float predictionError = std::fabs(columnState - prediction);

			minPredictionError = std::min(minPredictionError, predictionError);
		}

		for (int ci = 0; ci < _cellsPerColumn; ci++) {
			float prediction = _columns[i]._cells[ci]._predictionPrev;

			float predictionError = std::fabs(columnState - prediction);

			_columns[i]._cells[ci]._activation = std::exp((minPredictionError - predictionError) * cellIntensity);
			_columns[i]._cells[ci]._state = _columns[i]._cells[ci]._activation * columnState;
		}
	}

	// Clear intents
	for (int i = 0; i < _columns.size(); i++)
	for (int ci = 0; ci < _cellsPerColumn; ci++) {
		_columns[i]._cells[ci]._intent = 0.0f;
	}

	// Compute average variance of Q weights to scale for perturbation
	float averageWeightVariance = 0.0f;
	int weightCount = 0;

	for (int i = 0; i < _outputNodes.size(); i++) {
		weightCount += _outputNodes[i]._connections.size();

		for (int j = 0; j < _columns.size(); j++)
		for (int k = 0; k < _cellsPerColumn; k++)
			averageWeightVariance += std::fabs(_outputNodes[i]._connections[k + j * _cellsPerColumn]._weight);
	}

	averageWeightVariance /= weightCount;

	float averageWeightVarianceInv = 1.0f / averageWeightVariance;

	std::vector<float> bestCellStates(_cellsPerColumn);

	// Form predictions
	for (int rx = 0; rx < _columnsWidth; rx++)
	for (int ry = 0; ry < _columnsHeight; ry++) {
		int i = rx + ry * _columnsWidth;

		float maxPrediction = 0.0f;
		//float maxPerturbedPrediction = 0.0f;
		
		for (int ci = 0; ci < _cellsPerColumn; ci++) {
			float sum = _columns[i]._cells[ci]._bias._weight;
	
			// Go through all connections 
			int wi = 0;

			for (int dx = -_cellRadius; dx <= _cellRadius; dx++)
			for (int dy = -_cellRadius; dy <= _cellRadius; dy++)
			for (int cio = 0; cio < _cellsPerColumn; cio++) {
				int cx = rx + dx;
				int cy = ry + dy;

				if (cx >= 0 && cx < _columnsWidth && cy >= 0 && cy < _columnsHeight) {
					float cellWeight = _columns[i]._cells[ci]._connections[wi]._weight;

					int connectionIndex = cx + cy * _columnsWidth;

					sum += cellWeight * _columns[connectionIndex]._cells[cio]._state;
				}

				wi++;
			}

			_columns[i]._cells[ci]._prediction = sigmoid(sum * predictionIntensity);

			maxPrediction = std::max(maxPrediction, _columns[i]._cells[ci]._prediction);
		}

		/*float columnUncertainty = 0.0f;

		for (int ci = 0; ci < _cellsPerColumn; ci++) {
			columnUncertainty += maxPrediction - _columns[i]._cells[ci]._prediction;
		}

		columnUncertainty /= _cellsPerColumn - 1;*/

		_columns[i]._prediction = maxPrediction;

		_columns[i]._output = std::max(_columns[i]._state, _columns[i]._prediction);
	}

	// Derive action
	for (int rx = 0; rx < _columnsWidth; rx++)
	for (int ry = 0; ry < _columnsHeight; ry++) {
		int i = rx + ry * _columnsWidth;

		// Simulate competition to find desirable (high output) states of cells
		float chooseState = _columns[i]._prediction;

		float maxQ;

		{
			float minPredictionError = 1.0f;

			for (int ci = 0; ci < _cellsPerColumn; ci++) {
				float prediction = _columns[i]._cells[ci]._prediction;

				float predictionError = std::fabs(chooseState - prediction);

				minPredictionError = std::min(minPredictionError, predictionError);
			}

			for (int ci = 0; ci < _cellsPerColumn; ci++) {
				float prediction = _columns[i]._cells[ci]._prediction;

				float predictionError = std::fabs(chooseState - prediction);

				_columns[i]._cells[ci]._perturbedPrediction = std::exp((minPredictionError - predictionError) * cellIntensity) * chooseState;
			}

			// Compute direction of ascent
			float result = 0.0f;

			for (int ci = 0; ci < _cellsPerColumn; ci++) {
				result += _columns[i]._cells[ci]._perturbedPrediction * _outputNodes.front()._connections[ci + i * _cellsPerColumn]._weight * averageWeightVarianceInv;
			}

			maxQ = result;

			for (int ci = 0; ci < _cellsPerColumn; ci++)
				bestCellStates[ci] = _columns[i]._cells[ci]._perturbedPrediction;
		}

		float perturbationMultiplier = 1.0f;

		for (int t = 0; t < optimizationSteps; t++) {
			float testState = std::min(1.0f, std::max(0.0f, chooseState + perturbationMultiplier * annealingPerurbationDist(generator)));

			perturbationMultiplier *= annealingPerturbationDecay;

			float minPredictionError = 1.0f;

			for (int ci = 0; ci < _cellsPerColumn; ci++) {
				float prediction = _columns[i]._cells[ci]._prediction;

				float predictionError = std::fabs(testState - prediction);

				minPredictionError = std::min(minPredictionError, predictionError);
			}

			for (int ci = 0; ci < _cellsPerColumn; ci++) {
				float prediction = _columns[i]._cells[ci]._prediction;

				float predictionError = std::fabs(testState - prediction);

				_columns[i]._cells[ci]._perturbedPrediction = std::exp((minPredictionError - predictionError) * cellIntensity) * testState;
			}

			// Compute direction of ascent
			float result = 0.0f;

			for (int ci = 0; ci < _cellsPerColumn; ci++) {
				result += _columns[i]._cells[ci]._perturbedPrediction * _outputNodes.front()._connections[ci + i * _cellsPerColumn]._weight * averageWeightVarianceInv;
			}

			if (result > maxQ) {
				maxQ = result;
				chooseState = testState;

				for (int ci = 0; ci < _cellsPerColumn; ci++)
					bestCellStates[ci] = _columns[i]._cells[ci]._perturbedPrediction;
			}
		}

		for (int ci = 0; ci < _cellsPerColumn; ci++)
			_columns[i]._cells[ci]._perturbedPrediction = bestCellStates[ci];

		_columns[i]._intent = _columns[i]._prediction + perturbationIntensity * (chooseState - _columns[i]._prediction);
	}

	if (output.size() != _outputNodes.size())
		output.resize(_outputNodes.size());

	for (int i = 0; i < _outputNodes.size(); i++) {
		float sum = _outputNodes[i]._bias._weight;

		for (int j = 0; j < _columns.size(); j++)
		for (int k = 0; k < _cellsPerColumn; k++)
			sum += _columns[j]._cells[k]._state * _outputNodes[i]._connections[k + j * _cellsPerColumn]._weight;

		output[i] = sum;
	}

	// Reconstruct input using perturbed columns outputs
	if (action.size() != input.size())
		action.resize(input.size());

	for (int i = 0; i < action.size(); i++)
		action[i] = _reconNodes[i]._bias._weight;

	reconWeightCounters.clear();
	reconWeightCounters.assign(_reconNodes.size(), 0);

	for (int rx = 0; rx < _columnsWidth; rx++)
	for (int ry = 0; ry < _columnsHeight; ry++) {
		int i = rx + ry * _columnsWidth;

		float rxn = rx * rbfWidthInv;
		float ryn = ry * rbfHeightInv;

		for (int dx = -_receptiveRadius; dx <= _receptiveRadius; dx++)
		for (int dy = -_receptiveRadius; dy <= _receptiveRadius; dy++) {
			float xn = rxn + dx * inputWidthInv;
			float yn = ryn + dy * inputHeightInv;

			if (xn >= 0.0f && xn < 1.0f && yn >= 0.0f && yn < 1.0f) {
				int x = xn * _inputWidth;
				int y = yn * _inputHeight;

				int j = x + y * _inputWidth;

				action[j] += _reconNodes[j]._connections[reconWeightCounters[j]]._weight * _columns[i]._intent;

				reconWeightCounters[j]++;
			}
		}
	}
}

void CHTMRegion::learn(const std::vector<float> &input, const std::vector<float> &output, const std::vector<float> &target, float weightAlpha, float reconAlpha, float centerAlpha, float widthAlpha, float widthScalar, float minDistance, float minLearningThreshold, float cellAlpha) {
	// Update output node weights
	for (int i = 0; i < _outputNodes.size(); i++) {
		float alphaError = weightAlpha * (target[i] - output[i]);

		for (int j = 0; j < _columns.size(); j++)
		for (int k = 0; k < _cellsPerColumn; k++)
			_outputNodes[i]._connections[k + j * _cellsPerColumn]._weight += alphaError * _columns[j]._cells[k]._state;
		
		_outputNodes[i]._bias._weight += alphaError;
	}

	std::vector<float> reconNodeSums(_reconNodes.size());
	std::vector<int> reconWeightCounters(_reconNodes.size(), 0);

	for (int i = 0; i < _reconNodes.size(); i++)
		reconNodeSums[i] = _reconNodes[i]._bias._weight;

	float inputWidthInv = 1.0f / _inputWidth;
	float inputHeightInv = 1.0f / _inputHeight;

	float rbfWidthInv = 1.0f / _columnsWidth;
	float rbfHeightInv = 1.0f / _columnsHeight;

	for (int rx = 0; rx < _columnsWidth; rx++)
	for (int ry = 0; ry < _columnsHeight; ry++) {
		int i = rx + ry * _columnsWidth;

		float rxn = rx * rbfWidthInv;
		float ryn = ry * rbfHeightInv;

		int wi = 0;

		float learnScalar = std::max(0.0f, _columns[i]._state - minLearningThreshold);

		for (int dx = -_receptiveRadius; dx <= _receptiveRadius; dx++)
		for (int dy = -_receptiveRadius; dy <= _receptiveRadius; dy++) {
			float xn = rxn + dx * inputWidthInv;
			float yn = ryn + dy * inputHeightInv;

			if (xn >= 0.0f && xn < 1.0f && yn >= 0.0f && yn < 1.0f) {
				int x = xn * _inputWidth;
				int y = yn * _inputHeight;

				int j = x + y * _inputWidth;

				_columns[i]._center[wi]._weight += centerAlpha * learnScalar * (input[j] - _columns[i]._center[wi]._weight);

				float delta = input[j] - _columns[i]._center[wi]._weight;

				float dist = std::fabs(delta);

				_columns[i]._center[wi]._width = std::max(0.0f, _columns[i]._center[wi]._width + widthAlpha * learnScalar * (widthScalar / std::max(minDistance, dist) - _columns[i]._center[wi]._width));
			
				reconNodeSums[j] += _reconNodes[j]._connections[reconWeightCounters[j]]._weight * _columns[i]._state;

				reconWeightCounters[j]++;
			}

			wi++;
		}

		float columnPredictionError = _columns[i]._state - _columns[i]._predictionPrev;

		for (int ci = 0; ci < _cellsPerColumn; ci++) {
			float cellState = _columns[i]._cells[ci]._state;

			_columns[i]._cells[ci]._bias._weight += cellAlpha * columnPredictionError;

			// Go through all connections and update them
			int wi = 0;

			for (int dx = -_cellRadius; dx <= _cellRadius; dx++)
			for (int dy = -_cellRadius; dy <= _cellRadius; dy++)
			for (int cio = 0; cio < _cellsPerColumn; cio++) {
				int cx = rx + dx;
				int cy = ry + dy;

				if (cx >= 0 && cx < _columnsWidth && cy >= 0 && cy < _columnsHeight) {
					float connectionState = _columns[cx + cy * _columnsWidth]._cells[cio]._statePrev;

					_columns[i]._cells[ci]._connections[wi]._weight += cellAlpha * columnPredictionError * connectionState;
				}

				wi++;
			}
		}
	}

	// Learn reconstruction
	std::vector<float> reconErrors(_reconNodes.size());

	for (int i = 0; i < _reconNodes.size(); i++) {
		reconErrors[i] = input[i] - reconNodeSums[i];

		_reconNodes[i]._bias._weight += reconAlpha * reconErrors[i];
	}

	reconWeightCounters.clear();
	reconWeightCounters.assign(_reconNodes.size(), 0);

	for (int rx = 0; rx < _columnsWidth; rx++)
	for (int ry = 0; ry < _columnsHeight; ry++) {
		int i = rx + ry * _columnsWidth;

		float rxn = rx * rbfWidthInv;
		float ryn = ry * rbfHeightInv;

		int wi = 0;

		float learnScalar = std::max(0.0f, _columns[i]._state - minLearningThreshold);

		for (int dx = -_receptiveRadius; dx <= _receptiveRadius; dx++)
		for (int dy = -_receptiveRadius; dy <= _receptiveRadius; dy++) {
			float xn = rxn + dx * inputWidthInv;
			float yn = ryn + dy * inputHeightInv;

			if (xn >= 0.0f && xn < 1.0f && yn >= 0.0f && yn < 1.0f) {
				int x = xn * _inputWidth;
				int y = yn * _inputHeight;

				int j = x + y * _inputWidth;

				_reconNodes[j]._connections[reconWeightCounters[j]++]._weight += reconAlpha * reconErrors[j] * _columns[i]._state;
			}

			wi++;
		}
	}
}

void CHTMRegion::learnTraces(const std::vector<float> &input, const std::vector<float> &output, const std::vector<float> &error, const std::vector<float> &outputWeightAlphas, float centerAlpha, float widthAlpha, float widthScalar, float minDistance, float minLearningThreshold, float cellAlpha, float perturbationIntensity, const std::vector<float> &outputLambdas) {
	// Update output node weights
	for (int i = 0; i < _outputNodes.size(); i++) {
		float alphaError = outputWeightAlphas[i] * error[i];

		for (int j = 0; j < _columns.size(); j++)
		for (int k = 0; k < _cellsPerColumn; k++) {
			int connectionIndex = k + j * _cellsPerColumn;

			_outputNodes[i]._connections[connectionIndex]._weight += alphaError * _outputNodes[i]._connections[connectionIndex]._eligibility;
			_outputNodes[i]._connections[connectionIndex]._eligibility *= outputLambdas[i];
			_outputNodes[i]._connections[connectionIndex]._eligibility += _columns[j]._cells[k]._state;
		}

		_outputNodes[i]._bias._weight += alphaError * _outputNodes[i]._bias._eligibility;
		_outputNodes[i]._bias._eligibility *= outputLambdas[i];
		_outputNodes[i]._bias._eligibility += 1.0f;
	}

	float inputWidthInv = 1.0f / _inputWidth;
	float inputHeightInv = 1.0f / _inputHeight;

	float rbfWidthInv = 1.0f / _columnsWidth;
	float rbfHeightInv = 1.0f / _columnsHeight;

	for (int rx = 0; rx < _columnsWidth; rx++)
	for (int ry = 0; ry < _columnsHeight; ry++) {
		int i = rx + ry * _columnsWidth;

		float rxn = rx * rbfWidthInv;
		float ryn = ry * rbfHeightInv;

		int wi = 0;

		float learnScalar = std::max(0.0f, _columns[i]._state - minLearningThreshold);

		for (int dx = -_receptiveRadius; dx <= _receptiveRadius; dx++)
		for (int dy = -_receptiveRadius; dy <= _receptiveRadius; dy++) {
			float xn = rxn + dx * inputWidthInv;
			float yn = ryn + dy * inputHeightInv;

			if (xn >= 0.0f && xn < 1.0f && yn >= 0.0f && yn < 1.0f) {
				int x = xn * _inputWidth;
				int y = yn * _inputHeight;

				int j = x + y * _inputWidth;

				_columns[i]._center[wi]._weight += centerAlpha * learnScalar * (input[j] - _columns[i]._center[wi]._weight);

				float delta = input[j] - _columns[i]._center[wi]._weight;

				float dist = std::fabs(delta);

				_columns[i]._center[wi]._width = std::max(0.0f, _columns[i]._center[wi]._width + widthAlpha * learnScalar * (widthScalar / std::max(minDistance, dist) - _columns[i]._center[wi]._width));
			}

			wi++;
		}

		float columnPredictionError = _columns[i]._state - _columns[i]._predictionPrev;

		for (int ci = 0; ci < _cellsPerColumn; ci++) {
			float cellState = _columns[i]._cells[ci]._state;

			_columns[i]._cells[ci]._bias._weight += cellAlpha * columnPredictionError;

			// Go through all connections and update them
			int wi = 0;

			for (int dx = -_cellRadius; dx <= _cellRadius; dx++)
			for (int dy = -_cellRadius; dy <= _cellRadius; dy++)
			for (int cio = 0; cio < _cellsPerColumn; cio++) {
				int cx = rx + dx;
				int cy = ry + dy;

				if (cx >= 0 && cx < _columnsWidth && cy >= 0 && cy < _columnsHeight) {
					int connectionIndex = cx + cy * _columnsWidth;

					_columns[i]._cells[ci]._connections[wi]._weight += cellAlpha * columnPredictionError * _columns[connectionIndex]._cells[cio]._statePrev;
				}

				wi++;
			}
		}
	}
}

void CHTMRegion::findInput(const std::vector<float> &input, const std::vector<float> &output, std::vector<float> &newInput, float cellIntensity, int optimizationSteps, float optimizationAlpha, float annealingPerturbationStdDev, float annealingPerturbationDecay, std::mt19937 &generator) {
	float averageWeightVariance = 0.0f;
	int weightCount = 0;

	for (int i = 0; i < _outputNodes.size(); i++) {
		weightCount += _outputNodes[i]._connections.size();

		for (int j = 0; j < _columns.size(); j++)
		for (int k = 0; k < _cellsPerColumn; k++)
			averageWeightVariance += std::fabs(_outputNodes[i]._connections[k + j * _cellsPerColumn]._weight);
	}

	averageWeightVariance /= weightCount;

	float averageWeightVarianceInv = 1.0f / averageWeightVariance;
	
	std::vector<float> cellErrorsFirstPass(_columns.size() * _cellsPerColumn);

	std::vector<float> optimizationActivations(_cellsPerColumn, 0.0f);
	std::vector<float> optimizationStates(_cellsPerColumn, 0.0f);

	// Lateral error
	/*std::vector<float> cellErrorsSecondPass(_columns.size() * _cellsPerColumn);

	for (int rx = 0; rx < _columnsWidth; rx++)
	for (int ry = 0; ry < _columnsHeight; ry++) {
		int i = rx + ry * _columnsWidth;

		for (int ci = 0; ci < _cellsPerColumn; ci++) {
			float lateralError = 0.0f;

			int wi = 0;

			for (int dx = -_cellRadius; dx <= _cellRadius; dx++)
			for (int dy = -_cellRadius; dy <= _cellRadius; dy++)
			for (int cio = 0; cio < _cellsPerColumn; cio++) {
				int cx = rx + dx;
				int cy = ry + dy;

				if (cx >= 0 && cx < _columnsWidth && cy >= 0 && cy < _columnsHeight) {
					int cellColumnIndex = cx + cy * _columnsWidth;

					float cellError = cellErrorsFirstPass[cio + cellColumnIndex * _cellsPerColumn];

					float cellWeight = _columns[cellColumnIndex]._cells[cio]._connections[ci + (-dy + _cellRadius) * _cellsPerColumn + (-dx + _cellRadius) * _cellsPerColumn * _cellRadius]._weight;

					lateralError += cellError * cellWeight;
				}
			}

			// Add on additional error
			cellErrorsSecondPass[ci + i * _cellsPerColumn] = lateralError;
		}
	}*/

	newInput.clear();
	newInput.assign(_inputWidth * _inputHeight, 0.0f);

	float inputWidthInv = 1.0f / _inputWidth;
	float inputHeightInv = 1.0f / _inputHeight;

	float rbfWidthInv = 1.0f / _columnsWidth;
	float rbfHeightInv = 1.0f / _columnsHeight;

	for (int i = 0; i < newInput.size(); i++)
		newInput[i] = _reconNodes[i]._bias._weight;

	std::vector<int> reconWeightCounters(_reconNodes.size(), 0);

	std::normal_distribution<float> annealingPerurbationDist(0.0f, annealingPerturbationStdDev);

	for (int rx = 0; rx < _columnsWidth; rx++)
	for (int ry = 0; ry < _columnsHeight; ry++) {
		int i = rx + ry * _columnsWidth;

		float rxn = rx * rbfWidthInv;
		float ryn = ry * rbfHeightInv;

		// Simulate competition to find desirable (high output) states of cells
		float chooseState = _columns[i]._state;

		float maxQ;

		{
			float minPredictionError = 1.0f;

			for (int ci = 0; ci < _cellsPerColumn; ci++) {
				float prediction = _columns[i]._cells[ci]._predictionPrev;

				float predictionError = std::fabs(chooseState - prediction);

				minPredictionError = std::min(minPredictionError, predictionError);
			}

			for (int ci = 0; ci < _cellsPerColumn; ci++) {
				float prediction = _columns[i]._cells[ci]._predictionPrev;

				float predictionError = std::fabs(chooseState - prediction);

				optimizationStates[ci] = std::exp((minPredictionError - predictionError) * cellIntensity) * chooseState;
			}

			// Compute direction of ascent
			float result = 0.0f;

			for (int ci = 0; ci < _cellsPerColumn; ci++) {
				result += optimizationStates[ci] * _outputNodes.front()._connections[ci + i * _cellsPerColumn]._weight * averageWeightVarianceInv;
			}

			maxQ = result;
		}

		float perturbationMultiplier = 1.0f;

		for (int t = 0; t < optimizationSteps; t++) {
			float testState = std::min(1.0f, std::max(0.0f, chooseState + perturbationMultiplier * annealingPerurbationDist(generator)));

			perturbationMultiplier *= annealingPerturbationDecay;

			float minPredictionError = 1.0f;

			for (int ci = 0; ci < _cellsPerColumn; ci++) {
				float prediction = _columns[i]._cells[ci]._predictionPrev;

				float predictionError = std::fabs(testState - prediction);

				minPredictionError = std::min(minPredictionError, predictionError);
			}

			for (int ci = 0; ci < _cellsPerColumn; ci++) {
				float prediction = _columns[i]._cells[ci]._predictionPrev;

				float predictionError = std::fabs(testState - prediction);

				optimizationStates[ci] = std::exp((minPredictionError - predictionError) * cellIntensity) * testState;
			}

			// Compute direction of ascent
			float result = 0.0f;

			for (int ci = 0; ci < _cellsPerColumn; ci++) {
				result += optimizationStates[ci] * _outputNodes.front()._connections[ci + i * _cellsPerColumn]._weight * averageWeightVarianceInv;
			}

			if (result > maxQ) {
				maxQ = result;
				chooseState = testState;
			}
		}

		float dist2 = 0.0f;

		int wi = 0;

		for (int dx = -_receptiveRadius; dx <= _receptiveRadius; dx++)
		for (int dy = -_receptiveRadius; dy <= _receptiveRadius; dy++) {
			float xn = rxn + dx * inputWidthInv;
			float yn = ryn + dy * inputHeightInv;

			if (xn >= 0.0f && xn < 1.0f && yn >= 0.0f && yn < 1.0f) {
				int x = xn * _inputWidth;
				int y = yn * _inputHeight;

				int j = x + y * _inputWidth;

				newInput[j] += chooseState * _reconNodes[j]._connections[reconWeightCounters[j]++]._weight;
			}

			wi++;
		}
	}
}