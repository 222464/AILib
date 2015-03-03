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
/*
#include <rbf/CHTMRL.h>

#include <algorithm>

#include <iostream>

#include <assert.h>

using namespace rbf;

void CHTMRL::createRandom(int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, int numOutputs, float minCenter, float maxCenter, float minContextWeight, float maxContextWeight, float minFFNNWeight, float maxFFNNWeight, std::mt19937 &generator) {
	std::uniform_real_distribution<float> centerDist(minCenter, maxCenter);
	std::uniform_real_distribution<float> weightContextDist(minContextWeight, maxContextWeight);
	std::uniform_real_distribution<float> weightFFNNDist(minFFNNWeight, maxFFNNWeight);

	_inputWidth = inputWidth;
	_inputHeight = inputHeight;

	int numInputs = _inputWidth * _inputHeight;

	_layerDescs = layerDescs;

	_layers.resize(_layerDescs.size());

	// For keeping track of previous layer dimensions
	int prevLayerWidth = _inputWidth;
	int prevLayerHeight = _inputHeight;
	int prevLayerContextSize = 1;

	// Create layers
	for (int l = 0; l < _layers.size(); l++) {
		int numRBF = _layerDescs[l]._rbfWidth * _layerDescs[l]._rbfHeight;

		// Pre-calculate weight array sizes
		int rbfCenterSize = std::pow(_layerDescs[l]._rbfReceptiveRadius * 2 + 1, 2);
		int ffnnWeightsSize = std::pow(_layerDescs[l]._ffnnReceptiveRadius * 2 + 1, 2) * prevLayerContextSize;
		int contextWeightsSize = std::pow(_layerDescs[l]._contextReceptiveRadius * 2 + 1, 2) * _layerDescs[l]._numContextNodes;

		_layers[l]._rbfNodes.resize(numRBF);

		// Invert values for normalizing coordinates
		float rbfWidthInv = 1.0f / _layerDescs[l]._rbfWidth;
		float rbfHeightInv = 1.0f / _layerDescs[l]._rbfHeight;

		// Create RBF nodes
		for (int rx = 0; rx < _layerDescs[l]._rbfWidth; rx++)
		for (int ry = 0; ry < _layerDescs[l]._rbfHeight; ry++) {
			int i = rx + ry * _layerDescs[l]._rbfWidth;

			_layers[l]._rbfNodes[i]._center.resize(rbfCenterSize);

			for (int j = 0; j < rbfCenterSize; j++)
				_layers[l]._rbfNodes[i]._center[j] = centerDist(generator);

			_layers[l]._rbfNodes[i]._contextNodes.resize(_layerDescs[l]._numContextNodes);

			// Create context nodes
			for (int j = 0; j < _layerDescs[l]._numContextNodes; j++) {
				ContextNode &cn = _layers[l]._rbfNodes[i]._contextNodes[j];

				cn._contextBias = weightContextDist(generator);

				cn._contextWeights.resize(contextWeightsSize);

				for (int k = 0; k < contextWeightsSize; k++)
					cn._contextWeights[k] = weightContextDist(generator);

				cn._ffnnBias._weight = weightContextDist(generator);

				cn._ffnnConnections.resize(ffnnWeightsSize);

				for (int k = 0; k < contextWeightsSize; k++)
					cn._contextWeights[k] = weightContextDist(generator);

				// If not first layer, add back connections to previous layer
				if (l > 0) {
					float rxn = rx * rbfWidthInv;
					float ryn = ry * rbfHeightInv;

					int x = std::round(rxn * prevLayerWidth);
					int y = std::round(ryn * prevLayerHeight);

					int weightIndex = 0;

					for (int dx = -_layerDescs[l]._contextReceptiveRadius; dx <= _layerDescs[l]._contextReceptiveRadius; dx++)
					for (int dy = -_layerDescs[l]._contextReceptiveRadius; dy <= _layerDescs[l]._contextReceptiveRadius; dy++) {
						int xn = x + dx;
						int yn = y + dy;

						if (xn >= 0 && xn < prevLayerWidth && yn >= 0 && yn < prevLayerHeight) {
							int iPrev = xn + yn * prevLayerWidth;

							for (int k = 0; k < prevLayerContextSize; k++) {
								BackConnection bc;
								bc._nodeIndex = i;
								bc._contextIndex = j;
								bc._weightIndex = weightIndex;

								_layers[l - 1]._rbfNodes[iPrev]._contextNodes[k]._backConnections.push_back(bc);
							}
						}

						weightIndex++;
					}
				}
			}
		}

		prevLayerWidth = _layerDescs[l]._rbfWidth;
		prevLayerHeight = _layerDescs[l]._rbfHeight;
		prevLayerContextSize = _layerDescs[l]._numContextNodes;
	}

	// Set up Q output node
	_qOutput._connections.resize(_layerDescs.back()._rbfWidth * _layerDescs.back()._rbfHeight);

	for (int j = 0; j < _qOutput._connections.size(); j++)
		_qOutput._connections[j]._weight = weightFFNNDist(generator);

	_qOutput._bias._weight = weightFFNNDist(generator);
}

void CHTMRL::getOutput(const std::vector<float> &input, std::vector<float> &output, float activationIntensity, float dutyCycleDecay, float randomFireChance, float randomFireStrength, float minDistance, float minDutyCycle, std::mt19937 &generator) {
	int prevLayerWidth = _inputWidth;
	int prevLayerHeight = _inputHeight;

	std::vector<float> prevLayerOutput = input;

	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

	for (int l = 0; l < _layers.size(); l++) {
		float rbfWidthInv = 1.0f / _layerDescs[l]._rbfWidth;
		float rbfHeightInv = 1.0f / _layerDescs[l]._rbfHeight;

		for (int rx = 0; rx < _layerDescs[l]._rbfWidth; rx++)
		for (int ry = 0; ry < _layerDescs[l]._rbfHeight; ry++) {
			int i = rx + ry * _layerDescs[l]._rbfWidth;

			float rxn = rx * rbfWidthInv;
			float ryn = ry * rbfHeightInv;

			int x = std::round(rxn * prevLayerWidth);
			int y = std::round(ryn * prevLayerHeight);

			float dist2 = 0.0f;

			int weightIndex = 0;
			int usedWeightCount = 0;

			for (int dx = -_layerDescs[l]._receptiveRadius; dx <= _layerDescs[l]._receptiveRadius; dx++)
			for (int dy = -_layerDescs[l]._receptiveRadius; dy <= _layerDescs[l]._receptiveRadius; dy++) {
				int xn = x + dx;
				int yn = y + dy;

				if (xn >= 0 && xn < prevLayerWidth && yn >= 0 && yn < prevLayerHeight) {
					int j = xn + yn * prevLayerWidth;

					float delta = prevLayerOutput[j] - _layers[l]._rbfNodes[i]._center[weightIndex];

					float modulation = (std::abs(dx) + std::abs(dy)) / static_cast<float>(_layerDescs[l]._receptiveRadius * 2.0f);

					dist2 += delta * delta * modulation;

					usedWeightCount++;
				}

				weightIndex++;
			}

			_layers[l]._rbfNodes[i]._rbfActivation = -dist2 / (static_cast<float>(usedWeightCount) / weightIndex);
		}

		// Sparsify
		for (int rx = 0; rx < _layerDescs[l]._rbfWidth; rx++)
		for (int ry = 0; ry < _layerDescs[l]._rbfHeight; ry++) {
			int i = rx + ry * _layerDescs[l]._rbfWidth;

			float numHigher = 0.0f;

			//float maxNeighborhoodDutyCycle = 0.0f;

			for (int dx = -_layerDescs[l]._inhibitionRadius; dx <= _layerDescs[l]._inhibitionRadius; dx++)
			for (int dy = -_layerDescs[l]._inhibitionRadius; dy <= _layerDescs[l]._inhibitionRadius; dy++) {
				int x = rx + dx;
				int y = ry + dy;

				if (x >= 0 && x < _layerDescs[l]._rbfWidth && y >= 0 && y < _layerDescs[l]._rbfHeight) {
					int j = x + y * _layerDescs[l]._rbfWidth;

					if (_layers[l]._rbfNodes[j]._rbfActivation >= _layers[l]._rbfNodes[i]._rbfActivation)
						numHigher += 1.0f;
				}
			}

			float out = _layers[l]._rbfNodes[i]._rbfOutput = std::exp(-numHigher * _layerDescs[l]._outputIntensity);// sigmoid((_layerDescs[l]._localActivity - numHigher) * _layerDescs[l]._outputIntensity);

			_layers[l]._rbfNodes[i]._dutyCycle = (1.0f - dutyCycleDecay) * (1.0f - out) * _layers[l]._rbfNodes[i]._dutyCycle + out;
		}

		prevLayerOutput.resize(_layers[l]._rbfNodes.size());

		for (int i = 0; i < _layers[l]._rbfNodes.size(); i++)
			prevLayerOutput[i] = _layers[l]._rbfNodes[i]._rbfOutput;

		prevLayerWidth = _layerDescs[l]._rbfWidth;
		prevLayerHeight = _layerDescs[l]._rbfHeight;
	}

	// Activate through weights
	prevLayerWidth = _inputWidth;
	prevLayerHeight = _inputHeight;

	prevLayerOutput = input;

	for (int l = 0; l < _layers.size(); l++) {
		float rbfWidthInv = 1.0f / _layerDescs[l]._rbfWidth;
		float rbfHeightInv = 1.0f / _layerDescs[l]._rbfHeight;

		for (int rx = 0; rx < _layerDescs[l]._rbfWidth; rx++)
		for (int ry = 0; ry < _layerDescs[l]._rbfHeight; ry++) {
			int i = rx + ry * _layerDescs[l]._rbfWidth;

			float rxn = rx * rbfWidthInv;
			float ryn = ry * rbfHeightInv;

			int x = std::round(rxn * prevLayerWidth);
			int y = std::round(ryn * prevLayerHeight);

			float sum = _layers[l]._rbfNodes[i]._bias;

			int weightIndex = 0;
			int usedWeightCount = 0;

			for (int dx = -_layerDescs[l]._receptiveRadius; dx <= _layerDescs[l]._receptiveRadius; dx++)
			for (int dy = -_layerDescs[l]._receptiveRadius; dy <= _layerDescs[l]._receptiveRadius; dy++) {
				int xn = x + dx;
				int yn = y + dy;

				if (xn >= 0 && xn < prevLayerWidth && yn >= 0 && yn < prevLayerHeight) {
					int j = xn + yn * prevLayerWidth;

					sum += _layers[l]._rbfNodes[i]._weights[weightIndex] * prevLayerOutput[j];
				}

				weightIndex++;
			}

			_layers[l]._rbfNodes[i]._sig = sigmoid(sum);
			_layers[l]._rbfNodes[i]._output = _layers[l]._rbfNodes[i]._sig * _layers[l]._rbfNodes[i]._rbfOutput;
		}

		prevLayerOutput.resize(_layers[l]._rbfNodes.size());

		for (int i = 0; i < _layers[l]._rbfNodes.size(); i++)
			prevLayerOutput[i] = _layers[l]._rbfNodes[i]._output;

		prevLayerWidth = _layerDescs[l]._rbfWidth;
		prevLayerHeight = _layerDescs[l]._rbfHeight;
	}

	if (output.size() != _outputNodes.size())
		output.resize(_outputNodes.size());

	for (int i = 0; i < _outputNodes.size(); i++) {
		float sum = _outputNodes[i]._bias._weight;

		for (int rx = 0; rx < _layerDescs.back()._rbfWidth; rx++)
		for (int ry = 0; ry < _layerDescs.back()._rbfHeight; ry++) {
			int j = rx + ry * _layerDescs.back()._rbfWidth;

			sum += _layers.back()._rbfNodes[j]._output * _outputNodes[i]._connections[j]._weight;
		}

		output[i] = sigmoid(sum);
	}
}

void CHTMRL::updateUnsupervised(const std::vector<float> &input, float weightAlpha, float centerAlpha, float widthAlpha, float widthScalar, float minDistance, float minDutyCycle) {
	int prevLayerWidth = _inputWidth;
	int prevLayerHeight = _inputHeight;

	std::vector<float> prevLayerOutput = input;

	for (int l = 0; l < _layers.size(); l++) {
		float rbfWidthInv = 1.0f / _layerDescs[l]._rbfWidth;
		float rbfHeightInv = 1.0f / _layerDescs[l]._rbfHeight;

		for (int rx = 0; rx < _layerDescs[l]._rbfWidth; rx++)
		for (int ry = 0; ry < _layerDescs[l]._rbfHeight; ry++) {
			int i = rx + ry * _layerDescs[l]._rbfWidth;

			float rxn = rx * rbfWidthInv;
			float ryn = ry * rbfHeightInv;

			int x = std::round(rxn * prevLayerWidth);
			int y = std::round(ryn * prevLayerHeight);

			float boost = boostFunction(_layers[l]._rbfNodes[i]._dutyCycle, minDutyCycle);

			float learnScalar = _layers[l]._rbfNodes[i]._rbfOutput * (1.0f - boost) + boost;

			int weightIndex = 0;

			for (int dx = -_layerDescs[l]._receptiveRadius; dx <= _layerDescs[l]._receptiveRadius; dx++)
			for (int dy = -_layerDescs[l]._receptiveRadius; dy <= _layerDescs[l]._receptiveRadius; dy++) {
				int xn = x + dx;
				int yn = y + dy;

				if (xn >= 0 && xn < prevLayerWidth && yn >= 0 && yn < prevLayerHeight) {
					int j = xn + yn * prevLayerWidth;

					_layers[l]._rbfNodes[i]._center[weightIndex] += centerAlpha * learnScalar * (prevLayerOutput[j] - _layers[l]._rbfNodes[i]._center[weightIndex]);
				}

				weightIndex++;
			}

			//_layers[l]._rbfNodes[i]._width = std::max(0.0f, _layers[l]._rbfNodes[i]._width + widthAlpha * learnScalar * (widthScalar / std::max(minDistance, dist2) - _layers[l]._rbfNodes[i]._width));
		}

		prevLayerOutput.resize(_layers[l]._rbfNodes.size());

		for (int i = 0; i < _layers[l]._rbfNodes.size(); i++)
			prevLayerOutput[i] = _layers[l]._rbfNodes[i]._rbfOutput;

		prevLayerWidth = _layerDescs[l]._rbfWidth;
		prevLayerHeight = _layerDescs[l]._rbfHeight;
	}
}

void CHTMRL::updateSupervised(const std::vector<float> &input, const std::vector<float> &output, const std::vector<float> &target, float weightAlpha, float centerAlpha, float widthAlpha, float widthScalar, float minDistance, float minDutyCycle) {
	for (int i = 0; i < _outputNodes.size(); i++)
		_outputNodes[i]._error = target[i] - output[i];

	// Back propagate - first layer
	for (int rx = 0; rx < _layerDescs.back()._rbfWidth; rx++)
	for (int ry = 0; ry < _layerDescs.back()._rbfHeight; ry++) {
		int i = rx + ry * _layerDescs.back()._rbfWidth;

		float sum = 0.0f;

		for (int j = 0; j < _outputNodes.size(); j++)
			sum += _outputNodes[j]._connections[i]._weight * _outputNodes[j]._error;

		_layers.back()._rbfNodes[i]._error = sum * _layers.back()._rbfNodes[i]._sig * (1.0f - _layers.back()._rbfNodes[i]._sig) * _layers.back()._rbfNodes[i]._rbfOutput;
	}

	// Back propagate - all other layers (exclude first, want to use sparseness as input)
	for (int l = _layerDescs.size() - 2; l >= 0; l--) {
		int nl = l + 1;

		for (int rx = 0; rx < _layerDescs[l]._rbfWidth; rx++)
		for (int ry = 0; ry < _layerDescs[l]._rbfHeight; ry++) {
			int i = rx + ry * _layerDescs[l]._rbfWidth;

			float sum = 0.0f;

			for (int j = 0; j < _layers[l]._rbfNodes[i]._backConnections.size(); j++) {
				int ni = _layers[l]._rbfNodes[i]._backConnections[j]._nodeIndex;
				int wi = _layers[l]._rbfNodes[i]._backConnections[j]._weightIndex;

				sum += _layers[nl]._rbfNodes[ni]._weights[wi] * _layers[nl]._rbfNodes[ni]._error;
			}

			_layers[l]._rbfNodes[i]._error = sum * _layers[l]._rbfNodes[i]._sig * (1.0f - _layers[l]._rbfNodes[i]._sig) * _layers[l]._rbfNodes[i]._rbfOutput;
		}
	}

	// Update weights
	for (int i = 0; i < _outputNodes.size(); i++) {
		float alphaError = weightAlpha * _outputNodes[i]._error;

		for (int rx = 0; rx < _layerDescs.back()._rbfWidth; rx++)
		for (int ry = 0; ry < _layerDescs.back()._rbfHeight; ry++) {
			int j = rx + ry * _layerDescs.back()._rbfWidth;

			_outputNodes[i]._connections[j]._weight += alphaError * _layers.back()._rbfNodes[j]._output;
		}

		_outputNodes[i]._bias._weight += alphaError;
	}

	int prevLayerWidth = _inputWidth;
	int prevLayerHeight = _inputHeight;

	std::vector<float> prevLayerOutput = input;

	for (int l = 0; l < _layers.size(); l++) {
		float rbfWidthInv = 1.0f / _layerDescs[l]._rbfWidth;
		float rbfHeightInv = 1.0f / _layerDescs[l]._rbfHeight;

		for (int rx = 0; rx < _layerDescs[l]._rbfWidth; rx++)
		for (int ry = 0; ry < _layerDescs[l]._rbfHeight; ry++) {
			int i = rx + ry * _layerDescs[l]._rbfWidth;

			float rxn = rx * rbfWidthInv;
			float ryn = ry * rbfHeightInv;

			int x = std::round(rxn * prevLayerWidth);
			int y = std::round(ryn * prevLayerHeight);

			int weightIndex = 0;

			for (int dx = -_layerDescs[l]._receptiveRadius; dx <= _layerDescs[l]._receptiveRadius; dx++)
			for (int dy = -_layerDescs[l]._receptiveRadius; dy <= _layerDescs[l]._receptiveRadius; dy++) {
				int xn = x + dx;
				int yn = y + dy;

				if (xn >= 0 && xn < prevLayerWidth && yn >= 0 && yn < prevLayerHeight) {
					int j = xn + yn * prevLayerWidth;

					_layers[l]._rbfNodes[i]._weights[weightIndex] += _layerDescs[l]._weightAlpha * _layers[l]._rbfNodes[i]._error * prevLayerOutput[j];
				}

				weightIndex++;
			}

			_layers[l]._rbfNodes[i]._bias += weightAlpha * _layers[l]._rbfNodes[i]._error;

			//_layers[l]._rbfNodes[i]._width = std::max(0.0f, _layers[l]._rbfNodes[i]._width + widthAlpha * learnScalar * (widthScalar / std::max(minDistance, dist2) - _layers[l]._rbfNodes[i]._width));
		}

		prevLayerOutput.resize(_layers[l]._rbfNodes.size());

		for (int i = 0; i < _layers[l]._rbfNodes.size(); i++)
			prevLayerOutput[i] = _layers[l]._rbfNodes[i]._output;

		prevLayerWidth = _layerDescs[l]._rbfWidth;
		prevLayerHeight = _layerDescs[l]._rbfHeight;
	}
}

void CHTMRL::getImages(std::vector<sf::Image> &images) {
	images.clear();
	images.reserve(_layers.size());

	for (int l = 0; l < _layers.size(); l++) {
		sf::Image img;

		img.create(_layerDescs[l]._rbfWidth, _layerDescs[l]._rbfHeight);

		for (int x = 0; x < img.getSize().x; x++)
		for (int y = 0; y < img.getSize().y; y++) {
			sf::Color color = sf::Color::White;

			color.r = color.b = color.g = _layers[l]._rbfNodes[x + y * _layerDescs[l]._rbfWidth]._output * 255.0f;

			img.setPixel(x, y, color);
		}

		images.push_back(img);
	}
}*/