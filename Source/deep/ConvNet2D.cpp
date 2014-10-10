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

#include <deep/ConvNet2D.h>

#include <algorithm>

using namespace deep;

void ConvNet2D::createRandom(int inputMapWidth, int inputMapHeight, int inputNumMaps, const std::vector<LayerPairDesc> &layerDescs, float minWeight, float maxWeight, std::mt19937 &generator) {
	std::uniform_real_distribution<float> distWeight(minWeight, maxWeight);

	_convolutionLayers.resize(layerDescs.size());
	_downsamplingLayers.resize(layerDescs.size());

	_inputLayer._mapWidth = inputMapWidth;
	_inputLayer._mapHeight = inputMapHeight;

	int inputMapSize = _inputLayer._mapWidth * _inputLayer._mapHeight;

	_inputLayer._maps.resize(inputNumMaps);

	for (int m = 0; m < inputNumMaps; m++) {
		_inputLayer._maps[m]._outputs.clear();
		_inputLayer._maps[m]._outputs.assign(inputMapSize, 0.0f);
	}

	// First convolution layer
	{
		int l = 0;

		_convolutionLayers[l]._filterSizeWidth = layerDescs[l]._filterSizeWidth;
		_convolutionLayers[l]._filterSizeHeight = layerDescs[l]._filterSizeHeight;

		_convolutionLayers[l]._strideWidth = layerDescs[l]._strideWidth;
		_convolutionLayers[l]._strideHeight = layerDescs[l]._strideHeight;

		int numConnectionsPerNode = _convolutionLayers[l]._filterSizeWidth * _convolutionLayers[l]._filterSizeHeight * _inputLayer._maps.size();

		_convolutionLayers[l]._mapWidth = (_inputLayer._mapWidth - _convolutionLayers[l]._filterSizeWidth + 1) / (_convolutionLayers[l]._strideWidth);
		_convolutionLayers[l]._mapHeight = (_inputLayer._mapHeight - _convolutionLayers[l]._filterSizeHeight + 1) / (_convolutionLayers[l]._strideHeight);

		_convolutionLayers[l]._maps.resize(layerDescs[l]._numFeatureMaps);

		int numNodesPerMap = _convolutionLayers[l]._mapWidth * _convolutionLayers[l]._mapHeight;

		for (int m = 0; m < _convolutionLayers[l]._maps.size(); m++) {
			_convolutionLayers[l]._maps[m]._outputs.clear();
			_convolutionLayers[l]._maps[m]._outputs.assign(numNodesPerMap, 0.0f);

			Node &node = _convolutionLayers[l]._maps[m]._node;

			node._bias._weight = distWeight(generator);

			node._connections.resize(numConnectionsPerNode);

			for (int c = 0; c < numConnectionsPerNode; c++)
				node._connections[c]._weight = distWeight(generator);
		}
	}

	// First downsampling layer
	{
		int l = 0;

		_downsamplingLayers[l]._downsampleWidth = layerDescs[l]._downsampleWidth;
		_downsamplingLayers[l]._downsampleHeight = layerDescs[l]._downsampleHeight;

		_downsamplingLayers[l]._mapWidth = static_cast<int>(static_cast<float>(_convolutionLayers[l]._mapWidth) / _downsamplingLayers[l]._downsampleWidth);
		_downsamplingLayers[l]._mapHeight = static_cast<int>(static_cast<float>(_convolutionLayers[l]._mapHeight) / _downsamplingLayers[l]._downsampleHeight);

		int numNodesPerMap = _downsamplingLayers[l]._mapWidth * _downsamplingLayers[l]._mapHeight;

		_downsamplingLayers[l]._maps.resize(_convolutionLayers[l]._maps.size());

		for (int m = 0; m < _downsamplingLayers[l]._maps.size(); m++) {
			_downsamplingLayers[l]._maps[m]._outputs.clear();
			_downsamplingLayers[l]._maps[m]._outputs.assign(numNodesPerMap, 0.0f);
		}
	}

	// All other layers
	for (int l = 1; l < layerDescs.size(); l++) {
		int prevLayerIndex = l - 1;

		// ------------------------------ Convolutional Layer ------------------------------

		{
			_convolutionLayers[l]._filterSizeWidth = layerDescs[l]._filterSizeWidth;
			_convolutionLayers[l]._filterSizeHeight = layerDescs[l]._filterSizeHeight;

			_convolutionLayers[l]._strideWidth = layerDescs[l]._strideWidth;
			_convolutionLayers[l]._strideHeight = layerDescs[l]._strideHeight;

			int numConnectionsPerNode = _convolutionLayers[l]._filterSizeWidth * _convolutionLayers[l]._filterSizeHeight * _downsamplingLayers[prevLayerIndex]._maps.size();

			_convolutionLayers[l]._mapWidth = (_downsamplingLayers[prevLayerIndex]._mapWidth - _convolutionLayers[l]._filterSizeWidth + 1) / (_convolutionLayers[l]._strideWidth);
			_convolutionLayers[l]._mapHeight = (_downsamplingLayers[prevLayerIndex]._mapHeight - _convolutionLayers[l]._filterSizeHeight + 1) / (_convolutionLayers[l]._strideHeight);

			_convolutionLayers[l]._maps.resize(layerDescs[l]._numFeatureMaps);

			int numNodesPerMap = _convolutionLayers[l]._mapWidth * _convolutionLayers[l]._mapHeight;

			for (int m = 0; m < _convolutionLayers[l]._maps.size(); m++) {
				_convolutionLayers[l]._maps[m]._outputs.clear();
				_convolutionLayers[l]._maps[m]._outputs.assign(numNodesPerMap, 0.0f);

				Node &node = _convolutionLayers[l]._maps[m]._node;

				node._bias._weight = distWeight(generator);

				node._connections.resize(numConnectionsPerNode);

				for (int c = 0; c < numConnectionsPerNode; c++)
					node._connections[c]._weight = distWeight(generator);
			}
		}

		// ------------------------------ Downsampling Layer ------------------------------

		{
			_downsamplingLayers[l]._downsampleWidth = layerDescs[l]._downsampleWidth;
			_downsamplingLayers[l]._downsampleHeight = layerDescs[l]._downsampleHeight;

			_downsamplingLayers[l]._mapWidth = static_cast<int>(static_cast<float>(_convolutionLayers[l]._mapWidth) / _downsamplingLayers[l]._downsampleWidth);
			_downsamplingLayers[l]._mapHeight = static_cast<int>(static_cast<float>(_convolutionLayers[l]._mapHeight) / _downsamplingLayers[l]._downsampleHeight);

			int numNodesPerMap = _downsamplingLayers[l]._mapWidth * _downsamplingLayers[l]._mapHeight;

			_downsamplingLayers[l]._maps.resize(_convolutionLayers[l]._maps.size());

			for (int m = 0; m < _downsamplingLayers[l]._maps.size(); m++) {
				_downsamplingLayers[l]._maps[m]._outputs.clear();
				_downsamplingLayers[l]._maps[m]._outputs.assign(numNodesPerMap, 0.0f);
			}
		}
	}
}

void ConvNet2D::activate() {
	// First convolution layer
	{
		int l = 0;

		int filterSize = _convolutionLayers[l]._filterSizeWidth * _convolutionLayers[l]._filterSizeHeight;

		// Convolve
		for (int m = 0; m < _convolutionLayers[l]._maps.size(); m++)
		for (int nx = 0; nx < _convolutionLayers[l]._mapWidth; nx++)
		for (int ny = 0; ny < _convolutionLayers[l]._mapHeight; ny++) {
			Node &node = _convolutionLayers[l]._maps[m]._node;

			int ix = nx * _convolutionLayers[l]._strideWidth;
			int iy = ny * _convolutionLayers[l]._strideHeight;

			// Go through filter
			float sum = node._bias._weight;

			for (int fm = 0; fm < _inputLayer._maps.size(); fm++)
			for (int fx = 0; fx < _convolutionLayers[l]._filterSizeWidth; fx++)
			for (int fy = 0; fy < _convolutionLayers[l]._filterSizeHeight; fy++) {
				int tx = ix + fx;
				int ty = iy + fy;

				if (tx < _inputLayer._mapWidth && ty < _inputLayer._mapHeight)
					sum += node._connections[fx + fy * _convolutionLayers[l]._filterSizeWidth + fm * filterSize]._weight * _inputLayer._maps[fm]._outputs[tx + ty * _inputLayer._mapWidth];
			}

			_convolutionLayers[l]._maps[m]._outputs[nx + ny * _convolutionLayers[l]._mapWidth] = sigmoid(sum);
		}
	}

	// First downsampling layer
	{
		int l = 0;

		for (int m = 0; m < _downsamplingLayers[l]._maps.size(); m++)
		for (int nx = 0; nx < _downsamplingLayers[l]._mapWidth; nx++)
		for (int ny = 0; ny < _downsamplingLayers[l]._mapHeight; ny++) {
			int ix = nx * _downsamplingLayers[l]._downsampleWidth;
			int iy = ny * _downsamplingLayers[l]._downsampleHeight;

			// Go through filter
			float maximum = -999999.0f;

			for (int fx = 0; fx < _downsamplingLayers[l]._downsampleWidth; fx++)
			for (int fy = 0; fy < _downsamplingLayers[l]._downsampleHeight; fy++) {
				int tx = ix + fx;
				int ty = iy + fy;

				// Make sure it is in bounds
				if (tx < _convolutionLayers[l]._mapWidth && ty < _convolutionLayers[l]._mapHeight)
					maximum = std::max(maximum, _convolutionLayers[l]._maps[m]._outputs[tx + ty * _convolutionLayers[l]._mapWidth]);
			}

			_downsamplingLayers[l]._maps[m]._outputs[nx + ny * _downsamplingLayers[l]._mapWidth] = maximum;
		}
	}

	for (int l = 1; l < _convolutionLayers.size(); l++) {
		int prevLayerIndex = l - 1;

		// ------------------------------ Convolutional Layer ------------------------------

		{
			int filterSize = _convolutionLayers[l]._filterSizeWidth * _convolutionLayers[l]._filterSizeHeight;

			// Convolve
			for (int m = 0; m < _convolutionLayers[l]._maps.size(); m++)
			for (int nx = 0; nx < _convolutionLayers[l]._mapWidth; nx++)
			for (int ny = 0; ny < _convolutionLayers[l]._mapHeight; ny++) {
				Node &node = _convolutionLayers[l]._maps[m]._node;

				int ix = nx * _convolutionLayers[l]._strideWidth;
				int iy = ny * _convolutionLayers[l]._strideHeight;

				// Go through filter
				float sum = node._bias._weight;

				for (int fm = 0; fm < _downsamplingLayers[prevLayerIndex]._maps.size(); fm++)
				for (int fx = 0; fx < _convolutionLayers[l]._filterSizeWidth; fx++)
				for (int fy = 0; fy < _convolutionLayers[l]._filterSizeHeight; fy++) {
					int tx = ix + fx;
					int ty = iy + fy;

					if (tx < _downsamplingLayers[prevLayerIndex]._mapWidth && ty < _downsamplingLayers[prevLayerIndex]._mapHeight)
						sum += node._connections[fx + fy * _convolutionLayers[l]._filterSizeWidth + fm * filterSize]._weight * _downsamplingLayers[prevLayerIndex]._maps[fm]._outputs[tx + ty * _downsamplingLayers[prevLayerIndex]._mapWidth];
				}

				_convolutionLayers[l]._maps[m]._outputs[nx + ny * _convolutionLayers[l]._mapWidth] = sigmoid(sum);
			}
		}

		// ------------------------------ Downsampling Layer ------------------------------

		{
			for (int m = 0; m < _downsamplingLayers[l]._maps.size(); m++)
			for (int nx = 0; nx < _downsamplingLayers[l]._mapWidth; nx++)
			for (int ny = 0; ny < _downsamplingLayers[l]._mapHeight; ny++) {
				int ix = nx * _downsamplingLayers[l]._downsampleWidth;
				int iy = ny * _downsamplingLayers[l]._downsampleHeight;

				// Go through filter
				float maximum = -999999.0f;

				for (int fx = 0; fx < _downsamplingLayers[l]._downsampleWidth; fx++)
				for (int fy = 0; fy < _downsamplingLayers[l]._downsampleHeight; fy++) {
					int tx = ix + fx;
					int ty = iy + fy;

					// Make sure it is in bounds
					if (tx < _convolutionLayers[l]._mapWidth && ty < _convolutionLayers[l]._mapHeight)
						maximum = std::max(maximum, _convolutionLayers[l]._maps[m]._outputs[tx + ty * _convolutionLayers[l]._mapWidth]);
				}

				_downsamplingLayers[l]._maps[m]._outputs[nx + ny * _downsamplingLayers[l]._mapWidth] = maximum;
			}
		}
	}
}

void ConvNet2D::activateAndLearn(float alpha, std::mt19937 &generator) {
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	// First convolution layer
	{
		int l = 0;

		int prevMapSize = _inputLayer._mapWidth * _inputLayer._mapHeight;
		int filterSize = _convolutionLayers[l]._filterSizeWidth * _convolutionLayers[l]._filterSizeHeight;

		//float scaledAlpha = alpha / (_convolutionLayers[l]._mapWidth * _convolutionLayers[l]._mapHeight * _convolutionLayers[l]._maps.size());

		std::vector<std::vector<float>> tempPrevMaps(_inputLayer._maps.size());

		for (int m = 0; m < _inputLayer._maps.size(); m++)
			tempPrevMaps[m].assign(prevMapSize, 0.0f);

		// Convolve
		for (int m = 0; m < _convolutionLayers[l]._maps.size(); m++) {
			Node &node = _convolutionLayers[l]._maps[m]._node;

			for (int nx = 0; nx < _convolutionLayers[l]._mapWidth; nx++)
			for (int ny = 0; ny < _convolutionLayers[l]._mapHeight; ny++) {
				int ix = nx * _convolutionLayers[l]._strideWidth;
				int iy = ny * _convolutionLayers[l]._strideHeight;

				node._bias._positive = node._bias._negative = 0.0f;

				for (int c = 0; c < node._connections.size(); c++)
					node._connections[c]._positive = node._connections[c]._negative = 0.0f;

				// Activate forward
				float sum = node._bias._weight;

				for (int fm = 0; fm < _inputLayer._maps.size(); fm++)
				for (int fx = 0; fx < _convolutionLayers[l]._filterSizeWidth; fx++)
				for (int fy = 0; fy < _convolutionLayers[l]._filterSizeHeight; fy++) {
					int tx = ix + fx;
					int ty = iy + fy;

					if (tx < _inputLayer._mapWidth && ty < _inputLayer._mapHeight)
						sum += node._connections[fx + fy * _convolutionLayers[l]._filterSizeWidth]._weight * _inputLayer._maps[fm]._outputs[tx + ty * _inputLayer._mapWidth];
				}

				float output = sigmoid(sum);

				_convolutionLayers[l]._maps[m]._outputs[nx + ny * _convolutionLayers[l]._mapWidth] = output;

				float binaryOutput = dist01(generator) < output ? 1.0f : 0.0f;

				// Compute positives
				node._bias._positive = output;

				for (int fm = 0; fm < _inputLayer._maps.size(); fm++)
				for (int fx = 0; fx < _convolutionLayers[l]._filterSizeWidth; fx++)
				for (int fy = 0; fy < _convolutionLayers[l]._filterSizeHeight; fy++) {
					int tx = ix + fx;
					int ty = iy + fy;

					if (tx < _inputLayer._mapWidth && ty < _inputLayer._mapHeight)
						node._connections[fx + fy * _convolutionLayers[l]._filterSizeWidth + fm * filterSize]._positive += output * _inputLayer._maps[fm]._outputs[tx + ty * _inputLayer._mapWidth];
				}

				// Activate backward from binary activation
				//if (binaryOutput > 0.0f)
				for (int fm = 0; fm < _inputLayer._maps.size(); fm++)
				for (int fx = 0; fx < _convolutionLayers[l]._filterSizeWidth; fx++)
				for (int fy = 0; fy < _convolutionLayers[l]._filterSizeHeight; fy++) {
					int tx = ix + fx;
					int ty = iy + fy;

					if (tx < _inputLayer._mapWidth && ty < _inputLayer._mapHeight)
						tempPrevMaps[fm][tx + ty * _inputLayer._mapWidth] = binaryOutput * node._connections[fx + fy * _convolutionLayers[l]._filterSizeWidth + fm * filterSize]._weight;
				}

				// Activate forward
				sum = node._bias._weight;

				for (int fm = 0; fm < _inputLayer._maps.size(); fm++)
				for (int fx = 0; fx < _convolutionLayers[l]._filterSizeWidth; fx++)
				for (int fy = 0; fy < _convolutionLayers[l]._filterSizeHeight; fy++) {
					int tx = ix + fx;
					int ty = iy + fy;

					if (tx < _inputLayer._mapWidth && ty < _inputLayer._mapHeight)
						sum += node._connections[fx + fy * _convolutionLayers[l]._filterSizeWidth]._weight * tempPrevMaps[fm][tx + ty * _inputLayer._mapWidth];
				}

				output = sigmoid(sum);

				//float binaryOutput = dist01(generator) < output ? 1.0f : 0.0f;

				// Compute negatives
				node._bias._negative = output;

				for (int fm = 0; fm < _inputLayer._maps.size(); fm++)
				for (int fx = 0; fx < _convolutionLayers[l]._filterSizeWidth; fx++)
				for (int fy = 0; fy < _convolutionLayers[l]._filterSizeHeight; fy++) {
					int tx = ix + fx;
					int ty = iy + fy;

					if (tx < _inputLayer._mapWidth && ty < _inputLayer._mapHeight)
						node._connections[fx + fy * _convolutionLayers[l]._filterSizeWidth + fm * filterSize]._negative += output * tempPrevMaps[fm][tx + ty * _inputLayer._mapWidth];
				}

				node._bias._weight += alpha * (node._bias._positive - node._bias._negative);

				for (int c = 0; c < node._connections.size(); c++)
					node._connections[c]._weight += alpha * (node._connections[c]._positive - node._connections[c]._negative);
			}
		}
	}

	// First downsampling layer
	{
		int l = 0;

		for (int m = 0; m < _downsamplingLayers[l]._maps.size(); m++)
		for (int nx = 0; nx < _downsamplingLayers[l]._mapWidth; nx++)
		for (int ny = 0; ny < _downsamplingLayers[l]._mapHeight; ny++) {
			int ix = nx * _downsamplingLayers[l]._downsampleWidth;
			int iy = ny * _downsamplingLayers[l]._downsampleHeight;

			// Go through filter
			float maximum = -999999.0f;

			for (int fx = 0; fx < _downsamplingLayers[l]._downsampleWidth; fx++)
			for (int fy = 0; fy < _downsamplingLayers[l]._downsampleHeight; fy++) {
				int tx = ix + fx;
				int ty = iy + fy;

				// Make sure it is in bounds
				if (tx < _convolutionLayers[l]._mapWidth && ty < _convolutionLayers[l]._mapHeight)
					maximum = std::max(maximum, _convolutionLayers[l]._maps[m]._outputs[tx + ty * _convolutionLayers[l]._mapWidth]);
			}

			_downsamplingLayers[l]._maps[m]._outputs[nx + ny * _downsamplingLayers[l]._mapWidth] = maximum;
		}
	}

	for (int l = 1; l < _convolutionLayers.size(); l++) {
		int prevLayerIndex = l - 1;

		// ------------------------------ Convolutional Layer ------------------------------

		{
			int prevMapSize = _downsamplingLayers[prevLayerIndex]._mapWidth * _downsamplingLayers[prevLayerIndex]._mapHeight;
			int filterSize = _convolutionLayers[l]._filterSizeWidth * _convolutionLayers[l]._filterSizeHeight;

			std::vector<std::vector<float>> tempPrevMaps(_downsamplingLayers[prevLayerIndex]._maps.size());

			for (int m = 0; m < _downsamplingLayers[prevLayerIndex]._maps.size(); m++)
				tempPrevMaps[m].assign(prevMapSize, 0.0f);

			// Convolve
			for (int m = 0; m < _convolutionLayers[l]._maps.size(); m++) {
				Node &node = _convolutionLayers[l]._maps[m]._node;

				for (int nx = 0; nx < _convolutionLayers[l]._mapWidth; nx++)
				for (int ny = 0; ny < _convolutionLayers[l]._mapHeight; ny++) {
					int ix = nx * _convolutionLayers[l]._strideWidth;
					int iy = ny * _convolutionLayers[l]._strideHeight;

					node._bias._positive = node._bias._negative = 0.0f;

					for (int c = 0; c < node._connections.size(); c++)
						node._connections[c]._positive = node._connections[c]._negative = 0.0f;

					// Activate forward
					float sum = node._bias._weight;

					for (int fm = 0; fm < _downsamplingLayers[prevLayerIndex]._maps.size(); fm++)
					for (int fx = 0; fx < _convolutionLayers[l]._filterSizeWidth; fx++)
					for (int fy = 0; fy < _convolutionLayers[l]._filterSizeHeight; fy++) {
						int tx = ix + fx;
						int ty = iy + fy;

						if (tx < _downsamplingLayers[prevLayerIndex]._mapWidth && ty < _downsamplingLayers[prevLayerIndex]._mapHeight)
							sum += node._connections[fx + fy * _convolutionLayers[l]._filterSizeWidth]._weight * _downsamplingLayers[prevLayerIndex]._maps[fm]._outputs[tx + ty * _downsamplingLayers[prevLayerIndex]._mapWidth];
					}

					float output = sigmoid(sum);

					_convolutionLayers[l]._maps[m]._outputs[nx + ny * _convolutionLayers[l]._mapWidth] = output;

					float binaryOutput = dist01(generator) < output ? 1.0f : 0.0f;

					// Compute positives
					node._bias._positive = output;

					for (int fm = 0; fm < _downsamplingLayers[prevLayerIndex]._maps.size(); fm++)
					for (int fx = 0; fx < _convolutionLayers[l]._filterSizeWidth; fx++)
					for (int fy = 0; fy < _convolutionLayers[l]._filterSizeHeight; fy++) {
						int tx = ix + fx;
						int ty = iy + fy;

						if (tx < _downsamplingLayers[prevLayerIndex]._mapWidth && ty < _downsamplingLayers[prevLayerIndex]._mapHeight)
							node._connections[fx + fy * _convolutionLayers[l]._filterSizeWidth + fm * filterSize]._positive += output * _downsamplingLayers[prevLayerIndex]._maps[fm]._outputs[tx + ty * _downsamplingLayers[prevLayerIndex]._mapWidth];
					}

					// Activate backward from binary activation
					if (binaryOutput > 0.0f)
					for (int fm = 0; fm < _downsamplingLayers[prevLayerIndex]._maps.size(); fm++)
					for (int fx = 0; fx < _convolutionLayers[l]._filterSizeWidth; fx++)
					for (int fy = 0; fy < _convolutionLayers[l]._filterSizeHeight; fy++) {
						int tx = ix + fx;
						int ty = iy + fy;

						if (tx < _downsamplingLayers[prevLayerIndex]._mapWidth && ty < _downsamplingLayers[prevLayerIndex]._mapHeight)
							tempPrevMaps[fm][tx + ty * _downsamplingLayers[prevLayerIndex]._mapWidth] = node._connections[fx + fy * _convolutionLayers[l]._filterSizeWidth + fm * filterSize]._weight;
					}

					// Activate forward
					sum = node._bias._weight;

					for (int fm = 0; fm < _downsamplingLayers[prevLayerIndex]._maps.size(); fm++)
					for (int fx = 0; fx < _convolutionLayers[l]._filterSizeWidth; fx++)
					for (int fy = 0; fy < _convolutionLayers[l]._filterSizeHeight; fy++) {
						int tx = ix + fx;
						int ty = iy + fy;

						if (tx < _downsamplingLayers[prevLayerIndex]._mapWidth && ty < _downsamplingLayers[prevLayerIndex]._mapHeight)
							sum += node._connections[fx + fy * _convolutionLayers[l]._filterSizeWidth]._weight * tempPrevMaps[fm][tx + ty * _downsamplingLayers[prevLayerIndex]._mapWidth];
					}

					output = sigmoid(sum);

					// Compute negatives
					node._bias._negative = output;

					for (int fm = 0; fm < _downsamplingLayers[prevLayerIndex]._maps.size(); fm++)
					for (int fx = 0; fx < _convolutionLayers[l]._filterSizeWidth; fx++)
					for (int fy = 0; fy < _convolutionLayers[l]._filterSizeHeight; fy++) {
						int tx = ix + fx;
						int ty = iy + fy;

						if (tx < _downsamplingLayers[prevLayerIndex]._mapWidth && ty < _downsamplingLayers[prevLayerIndex]._mapHeight)
							node._connections[fx + fy * _convolutionLayers[l]._filterSizeWidth + fm * filterSize]._negative += output * tempPrevMaps[fm][tx + ty * _downsamplingLayers[prevLayerIndex]._mapWidth];
					}

					node._bias._weight += alpha * (node._bias._positive - node._bias._negative);

					for (int c = 0; c < node._connections.size(); c++)
						node._connections[c]._weight += alpha * (node._connections[c]._positive - node._connections[c]._negative);
				}
			}
		}

		// ------------------------------ Downsampling Layer ------------------------------

		{
			for (int m = 0; m < _downsamplingLayers[l]._maps.size(); m++)
			for (int nx = 0; nx < _downsamplingLayers[l]._mapWidth; nx++)
			for (int ny = 0; ny < _downsamplingLayers[l]._mapHeight; ny++) {
				int ix = nx * _downsamplingLayers[l]._downsampleWidth;
				int iy = ny * _downsamplingLayers[l]._downsampleHeight;

				// Go through filter
				float maximum = -999999.0f;

				for (int fx = 0; fx < _downsamplingLayers[l]._downsampleWidth; fx++)
				for (int fy = 0; fy < _downsamplingLayers[l]._downsampleHeight; fy++) {
					int tx = ix + fx;
					int ty = iy + fy;

					// Make sure it is in bounds
					if (tx < _convolutionLayers[l]._mapWidth && ty < _convolutionLayers[l]._mapHeight)
						maximum = std::max(maximum, _convolutionLayers[l]._maps[m]._outputs[tx + ty * _convolutionLayers[l]._mapWidth]);
				}

				_downsamplingLayers[l]._maps[m]._outputs[nx + ny * _downsamplingLayers[l]._mapWidth] = maximum;
			}
		}
	}
}