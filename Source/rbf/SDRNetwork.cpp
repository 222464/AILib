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

#include <rbf/SDRNetwork.h>

#include <algorithm>

#include <iostream>

#include <assert.h>

using namespace sdr;

void SDRNetwork::createRandom(int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, int numOutputs, float minSDRWeight, float maxSDRWeight, float minInhibitionWeight, float maxInhibitionWeight, float minBackWeight, float maxBackWeight, std::mt19937 &generator) {
	std::uniform_real_distribution<float> sdrWeightDist(minSDRWeight, maxSDRWeight);
	std::uniform_real_distribution<float> inhibitionDist(minInhibitionWeight, maxInhibitionWeight);
	std::uniform_real_distribution<float> backWeightDist(minBackWeight, maxBackWeight);

	_inputWidth = inputWidth;
	_inputHeight = inputHeight;

	int numInputs = _inputWidth * _inputHeight;

	_layerDescs = layerDescs;

	_layers.resize(_layerDescs.size());

	int prevLayerWidth = _inputWidth;
	int prevLayerHeight = _inputHeight;

	for (int l = 0; l < _layers.size(); l++) {
		int numNodes = _layerDescs[l]._width * _layerDescs[l]._height;
		int numWeights = std::pow(_layerDescs[l]._receptiveRadius * 2 + 1, 2);
		int numInhibition = std::pow(_layerDescs[l]._inhibitionRadius * 2 + 1, 2);

		_layers[l]._nodes.resize(numNodes);

		float rbfWidthInv = 1.0f / _layerDescs[l]._width;
		float rbfHeightInv = 1.0f / _layerDescs[l]._height;

		for (int rx = 0; rx < _layerDescs[l]._width; rx++)
		for (int ry = 0; ry < _layerDescs[l]._height; ry++) {
			int i = rx + ry * _layerDescs[l]._width;

			_layers[l]._nodes[i]._sdrWeights.resize(numWeights);
			_layers[l]._nodes[i]._sdrInhibition.resize(numInhibition);
			_layers[l]._nodes[i]._backWeights.resize(numWeights);

			_layers[l]._nodes[i]._backBias._weight = backWeightDist(generator);
			_layers[l]._nodes[i]._sdrBias = sdrWeightDist(generator);

			for (int j = 0; j < numWeights; j++) {
				_layers[l]._nodes[i]._sdrWeights[j] = sdrWeightDist(generator);

				_layers[l]._nodes[i]._backWeights[j]._weight = backWeightDist(generator);
			}

			for (int j = 0; j < numInhibition; j++)
				_layers[l]._nodes[i]._sdrInhibition[j] = inhibitionDist(generator);

			// If not first layer, add back connections to previous layer
			if (l > 0) {
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

						BackConnection bc;
						bc._nodeIndex = i;
						bc._weightIndex = weightIndex;

						_layers[l - 1]._nodes[j]._backConnections.push_back(bc);
					}

					weightIndex++;
				}
			}
		}

		prevLayerWidth = _layerDescs[l]._width;
		prevLayerHeight = _layerDescs[l]._height;
	}

	_outputNodes.resize(numOutputs);

	for (int i = 0; i < _outputNodes.size(); i++) {
		_outputNodes[i]._connections.resize(_layerDescs.back()._width * _layerDescs.back()._height);

		for (int j = 0; j < _outputNodes[i]._connections.size(); j++)
			_outputNodes[i]._connections[j]._weight = backWeightDist(generator);

		_outputNodes[i]._bias._weight = backWeightDist(generator);
	}
}

void SDRNetwork::getOutput(const std::vector<float> &input, std::vector<float> &output, std::mt19937 &generator) {
	int prevLayerWidth = _inputWidth;
	int prevLayerHeight = _inputHeight;

	std::vector<float> prevLayerOutput = input;

	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

	for (int l = 0; l < _layers.size(); l++) {
		float rbfWidthInv = 1.0f / _layerDescs[l]._width;
		float rbfHeightInv = 1.0f / _layerDescs[l]._height;

		for (int rx = 0; rx < _layerDescs[l]._width; rx++)
		for (int ry = 0; ry < _layerDescs[l]._height; ry++) {
			int i = rx + ry * _layerDescs[l]._width;

			float rxn = rx * rbfWidthInv;
			float ryn = ry * rbfHeightInv;

			int x = std::round(rxn * prevLayerWidth);
			int y = std::round(ryn * prevLayerHeight);

			float sum = 0.0f;

			int weightIndex = 0;
	
			for (int dx = -_layerDescs[l]._receptiveRadius; dx <= _layerDescs[l]._receptiveRadius; dx++)
			for (int dy = -_layerDescs[l]._receptiveRadius; dy <= _layerDescs[l]._receptiveRadius; dy++) {
				int xn = x + dx;
				int yn = y + dy;

				if (xn >= 0 && xn < prevLayerWidth && yn >= 0 && yn < prevLayerHeight) {
					int j = xn + yn * prevLayerWidth;

					sum += prevLayerOutput[j] * _layers[l]._nodes[i]._sdrWeights[weightIndex];
				}

				weightIndex++;
			}

			_layers[l]._nodes[i]._sdrActivation = std::max(_layerDescs[l]._sdrActivationLeak, sum);// std::max(0.0f, sum);
		}

		// Sparsify
		for (int rx = 0; rx < _layerDescs[l]._width; rx++)
		for (int ry = 0; ry < _layerDescs[l]._height; ry++) {
			int i = rx + ry * _layerDescs[l]._width;

			float sum = std::max(_layerDescs[l]._sdrActivationLeak, _layers[l]._nodes[i]._sdrActivation);

			int weightIndex = 0;

			for (int dx = -_layerDescs[l]._inhibitionRadius; dx <= _layerDescs[l]._inhibitionRadius; dx++)
			for (int dy = -_layerDescs[l]._inhibitionRadius; dy <= _layerDescs[l]._inhibitionRadius; dy++) {
				if (dx != 0 && dy != 0) {
					int x = rx + dx;
					int y = ry + dy;

					if (x >= 0 && x < _layerDescs[l]._width && y >= 0 && y < _layerDescs[l]._height) {
						int j = x + y * _layerDescs[l]._width;

						float dist2 = dx * dx + dy * dy;

						float dFactor = std::exp(-_layerDescs[l]._similarityDistanceFactor * dist2);

						sum -= dFactor * _layers[l]._nodes[i]._sdrInhibition[weightIndex] * _layers[l]._nodes[j]._sdrActivation * _layers[l]._nodes[i]._sdrActivation;
					}
				}

				weightIndex++;
			}

			_layers[l]._nodes[i]._sdrOutput = std::max(0.0f, sum);
		}

		prevLayerOutput.resize(_layers[l]._nodes.size());

		for (int i = 0; i < _layers[l]._nodes.size(); i++)
			prevLayerOutput[i] = _layers[l]._nodes[i]._sdrOutput;

		prevLayerWidth = _layerDescs[l]._width;
		prevLayerHeight = _layerDescs[l]._height;
	}

	// Activate through weights
	prevLayerWidth = _inputWidth;
	prevLayerHeight = _inputHeight;

	prevLayerOutput = input;

	for (int l = 0; l < _layers.size(); l++) {
		float rbfWidthInv = 1.0f / _layerDescs[l]._width;
		float rbfHeightInv = 1.0f / _layerDescs[l]._height;

		for (int rx = 0; rx < _layerDescs[l]._width; rx++)
		for (int ry = 0; ry < _layerDescs[l]._height; ry++) {
			int i = rx + ry * _layerDescs[l]._width;

			float rxn = rx * rbfWidthInv;
			float ryn = ry * rbfHeightInv;

			int x = std::round(rxn * prevLayerWidth);
			int y = std::round(ryn * prevLayerHeight);

			float sum = _layers[l]._nodes[i]._backBias._weight;

			int weightIndex = 0;

			for (int dx = -_layerDescs[l]._receptiveRadius; dx <= _layerDescs[l]._receptiveRadius; dx++)
			for (int dy = -_layerDescs[l]._receptiveRadius; dy <= _layerDescs[l]._receptiveRadius; dy++) {
				int xn = x + dx;
				int yn = y + dy;

				if (xn >= 0 && xn < prevLayerWidth && yn >= 0 && yn < prevLayerHeight) {
					int j = xn + yn * prevLayerWidth;

					sum += _layers[l]._nodes[i]._backWeights[weightIndex]._weight * prevLayerOutput[j];
				}

				weightIndex++;
			}

			_layers[l]._nodes[i]._backSig = sigmoid(sum);
			_layers[l]._nodes[i]._backOutput = _layers[l]._nodes[i]._backSig * _layers[l]._nodes[i]._sdrOutput;
		}

		prevLayerOutput.resize(_layers[l]._nodes.size());

		for (int i = 0; i < _layers[l]._nodes.size(); i++)
			prevLayerOutput[i] = _layers[l]._nodes[i]._backOutput;

		prevLayerWidth = _layerDescs[l]._width;
		prevLayerHeight = _layerDescs[l]._height;
	}

	if (output.size() != _outputNodes.size())
		output.resize(_outputNodes.size());

	for (int i = 0; i < _outputNodes.size(); i++) {
		float sum = _outputNodes[i]._bias._weight;

		for (int rx = 0; rx < _layerDescs.back()._width; rx++)
		for (int ry = 0; ry < _layerDescs.back()._height; ry++) {
			int j = rx + ry * _layerDescs.back()._width;

			sum += _layers.back()._nodes[j]._backOutput * _outputNodes[i]._connections[j]._weight;
		}

		output[i] = sum;
	}
}

void SDRNetwork::updateUnsupervised(const std::vector<float> &input, float sdrWeightAlpha, float inhibitionAlpha, float biasAlpha) {
	int prevLayerWidth = _inputWidth;
	int prevLayerHeight = _inputHeight;

	std::vector<float> prevLayerOutput = input;

	for (int l = 0; l < _layers.size(); l++) {
		float rbfWidthInv = 1.0f / _layerDescs[l]._width;
		float rbfHeightInv = 1.0f / _layerDescs[l]._height;

		/*std::vector<float> reconstruction(prevLayerWidth * prevLayerHeight, 0.0f);

		for (int rx = 0; rx < _layerDescs[l]._width; rx++)
			for (int ry = 0; ry < _layerDescs[l]._height; ry++) {
				int i = rx + ry * _layerDescs[l]._width;

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

							reconstruction[j] += _layers[l]._nodes[i]._sdrOutput * _layers[l]._nodes[i]._sdrWeights[weightIndex];
						}

						weightIndex++;
					}
			}

		std::vector<float> error(prevLayerWidth * prevLayerHeight);

		for (int i = 0; i < error.size(); i++)
			error[i] = prevLayerOutput[i] - reconstruction[i];*/

		for (int rx = 0; rx < _layerDescs[l]._width; rx++)
		for (int ry = 0; ry < _layerDescs[l]._height; ry++) {
			int i = rx + ry * _layerDescs[l]._width;

			float rxn = rx * rbfWidthInv;
			float ryn = ry * rbfHeightInv;

			int x = std::round(rxn * prevLayerWidth);
			int y = std::round(ryn * prevLayerHeight);

			float learn = _layers[l]._nodes[i]._sdrOutput;

			int weightIndex = 0;

			for (int dx = -_layerDescs[l]._receptiveRadius; dx <= _layerDescs[l]._receptiveRadius; dx++)
			for (int dy = -_layerDescs[l]._receptiveRadius; dy <= _layerDescs[l]._receptiveRadius; dy++) {
				int xn = x + dx;
				int yn = y + dy;

				if (xn >= 0 && xn < prevLayerWidth && yn >= 0 && yn < prevLayerHeight) {
					int j = xn + yn * prevLayerWidth;

					_layers[l]._nodes[i]._sdrWeights[weightIndex] += sdrWeightAlpha * learn * (prevLayerOutput[j] - _layers[l]._nodes[i]._sdrActivation * _layers[l]._nodes[i]._sdrWeights[weightIndex]);
				}

				weightIndex++;
			}

			weightIndex = 0;

			for (int dx = -_layerDescs[l]._inhibitionRadius; dx <= _layerDescs[l]._inhibitionRadius; dx++)
				for (int dy = -_layerDescs[l]._inhibitionRadius; dy <= _layerDescs[l]._inhibitionRadius; dy++) {
					int x = rx + dx;
					int y = ry + dy;

					if (x >= 0 && x < _layerDescs[l]._width && y >= 0 && y < _layerDescs[l]._height) {
						int j = x + y * _layerDescs[l]._width;

						_layers[l]._nodes[i]._sdrInhibition[weightIndex] = std::max(0.0f, _layers[l]._nodes[i]._sdrInhibition[weightIndex] + inhibitionAlpha * ((_layers[l]._nodes[i]._sdrOutput > 0.0f ? 1.0f : 0.0f) - _layerDescs[l]._sparsity) * std::max(0.0f, _layers[l]._nodes[j]._sdrActivation - _layers[l]._nodes[i]._sdrActivation));
					}

					weightIndex++;
				}

			_layers[l]._nodes[i]._sdrBias += biasAlpha * (_layerDescs[l]._sparsity - _layers[l]._nodes[i]._sdrOutput);
		}

		prevLayerOutput.resize(_layers[l]._nodes.size());

		for (int i = 0; i < _layers[l]._nodes.size(); i++)
			prevLayerOutput[i] = _layers[l]._nodes[i]._sdrOutput;

		prevLayerWidth = _layerDescs[l]._width;
		prevLayerHeight = _layerDescs[l]._height;
	}
}

void SDRNetwork::updateSupervised(const std::vector<float> &input, const std::vector<float> &output, const std::vector<float> &target, float backWeightAlpha, float backWeightOutputLayerAlpha, float momentum) {
	for (int i = 0; i < _outputNodes.size(); i++)
		_outputNodes[i]._error = target[i] - output[i];

	// Back propagate - first layer
	for (int rx = 0; rx < _layerDescs.back()._width; rx++)
	for (int ry = 0; ry < _layerDescs.back()._height; ry++) {
		int i = rx + ry * _layerDescs.back()._width;

		float sum = 0.0f;

		for (int j = 0; j < _outputNodes.size(); j++)
			sum += _outputNodes[j]._connections[i]._weight * _outputNodes[j]._error;

		_layers.back()._nodes[i]._backError = sum * _layers.back()._nodes[i]._backSig * (1.0f - _layers.back()._nodes[i]._backSig) * _layers.back()._nodes[i]._sdrOutput;
	}

	// Back propagate - all other layers (exclude first, want to use sparseness as input)
	for (int l = _layerDescs.size() - 2; l >= 0; l--) {
		int nl = l + 1;

		for (int rx = 0; rx < _layerDescs[l]._width; rx++)
		for (int ry = 0; ry < _layerDescs[l]._height; ry++) {
			int i = rx + ry * _layerDescs[l]._width;

			float sum = 0.0f;

			for (int j = 0; j < _layers[l]._nodes[i]._backConnections.size(); j++) {
				int ni = _layers[l]._nodes[i]._backConnections[j]._nodeIndex;
				int wi = _layers[l]._nodes[i]._backConnections[j]._weightIndex;

				sum += _layers[nl]._nodes[ni]._backWeights[wi]._weight * _layers[nl]._nodes[ni]._backError;
			}

			_layers[l]._nodes[i]._backError = sum *  _layers[l]._nodes[i]._backSig * (1.0f - _layers[l]._nodes[i]._backSig) * _layers[l]._nodes[i]._sdrOutput;
		}
	}

	// Update weights
	for (int i = 0; i < _outputNodes.size(); i++) {
		float alphaError = backWeightOutputLayerAlpha * _outputNodes[i]._error;

		for (int rx = 0; rx < _layerDescs.back()._width; rx++)
		for (int ry = 0; ry < _layerDescs.back()._height; ry++) {
			int j = rx + ry * _layerDescs.back()._width;

			_outputNodes[i]._connections[j]._weight += alphaError * _layers.back()._nodes[j]._backOutput;
		}

		_outputNodes[i]._bias._weight += alphaError;
	}

	int prevLayerWidth = _inputWidth;
	int prevLayerHeight = _inputHeight;

	std::vector<float> prevLayerOutput = input;

	for (int l = 0; l < _layers.size(); l++) {
		float rbfWidthInv = 1.0f / _layerDescs[l]._width;
		float rbfHeightInv = 1.0f / _layerDescs[l]._height;

		for (int rx = 0; rx < _layerDescs[l]._width; rx++)
		for (int ry = 0; ry < _layerDescs[l]._height; ry++) {
			int i = rx + ry * _layerDescs[l]._width;

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

					float prevWeight = _layers[l]._nodes[i]._backWeights[weightIndex]._weight;

					_layers[l]._nodes[i]._backWeights[weightIndex]._weight += _layers[l]._nodes[i]._backWeights[weightIndex]._prevDWeight * momentum + backWeightAlpha * _layers[l]._nodes[i]._backError * prevLayerOutput[j];
				
					_layers[l]._nodes[i]._backWeights[weightIndex]._prevDWeight = _layers[l]._nodes[i]._backWeights[weightIndex]._weight - prevWeight;
				}

				weightIndex++;
			}

			float prevWeight = _layers[l]._nodes[i]._backBias._weight;

			_layers[l]._nodes[i]._backBias._weight += _layers[l]._nodes[i]._backBias._prevDWeight * momentum + backWeightAlpha * _layers[l]._nodes[i]._backError;

			_layers[l]._nodes[i]._backBias._prevDWeight = _layers[l]._nodes[i]._backBias._weight - prevWeight;
		}

		prevLayerOutput.resize(_layers[l]._nodes.size());

		for (int i = 0; i < _layers[l]._nodes.size(); i++)
			prevLayerOutput[i] = _layers[l]._nodes[i]._backOutput;

		prevLayerWidth = _layerDescs[l]._width;
		prevLayerHeight = _layerDescs[l]._height;
	}
}

void SDRNetwork::getImages(std::vector<sf::Image> &images) {
	images.clear();
	images.reserve(_layers.size());

	float maxActivation = 0.0001f;

	for (int l = 0; l < _layers.size(); l++) {
		for (int x = 0; x < _layerDescs[l]._width; x++)
			for (int y = 0; y < _layerDescs[l]._height; y++) {
				maxActivation = std::max(maxActivation, _layers[l]._nodes[x + y * _layerDescs[l]._width]._sdrOutput);
			}
	}

	float activationMult = 1.0f / maxActivation;

	for (int l = 0; l < _layers.size(); l++) {
		sf::Image img;

		img.create(_layerDescs[l]._width, _layerDescs[l]._height);

		for (int x = 0; x < img.getSize().x; x++)
		for (int y = 0; y < img.getSize().y; y++) {
			sf::Color color = sf::Color::White;

			color.r = color.b = color.g = std::min(1.0f, std::max(0.0f, _layers[l]._nodes[x + y * _layerDescs[l]._width]._sdrOutput)) * 255.0f;

			img.setPixel(x, y, color);
		}

		images.push_back(img);
	}
}

void SDRNetwork::getReceptiveFields(int layer, sf::Image &image) {
	int windowSize = _layerDescs[layer]._receptiveRadius * 2 + 1;

	image.create(windowSize * _layerDescs[layer]._width, windowSize * _layerDescs[layer]._height);

	float minWeight = 9999.0f;
	float maxWeight = -9999.0f;

	for (int wx = 0; wx < _layerDescs[layer]._width; wx++)
		for (int wy = 0; wy < _layerDescs[layer]._height; wy++) {
			for (int x = 0; x < windowSize; x++)
				for (int y = 0; y < windowSize; y++) {
					float w = _layers[layer]._nodes[wx + wy * _layerDescs[layer]._width]._sdrWeights[y + x * windowSize];

					minWeight = std::min(minWeight, w);

					maxWeight = std::max(maxWeight, w);
				}
		}

	float mult = 1.0f / (maxWeight - minWeight);

	for (int wx = 0; wx < _layerDescs[layer]._width; wx++)
		for (int wy = 0; wy < _layerDescs[layer]._height; wy++) {
			for (int x = 0; x < windowSize; x++)
				for (int y = 0; y < windowSize; y++) {
					sf::Color color = sf::Color::White;

					color.r = color.b = color.g = mult * (_layers[layer]._nodes[wx + wy * _layerDescs[layer]._width]._sdrWeights[y + x * windowSize] - minWeight) * 255.0f;

					image.setPixel(wx * windowSize + x, wy * windowSize + y, color);
				}
		}
}