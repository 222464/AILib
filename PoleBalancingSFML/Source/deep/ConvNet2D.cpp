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

void ConvNet2D::createRandom(size_t inputMapWidth, size_t inputMapHeight, size_t inputNumMaps, const std::vector<LayerPairDesc> &layerDescs, float minWeight, float maxWeight, std::mt19937 &generator) {
	std::uniform_real_distribution<float> distWeight(minWeight, maxWeight);

	_convolutionLayers.resize(layerDescs.size());
	_downsamplingLayers.resize(layerDescs.size());

	_inputLayer._mapWidth = inputMapWidth;
	_inputLayer._mapHeight = inputMapHeight;

	size_t inputMapSize = _inputLayer._mapWidth * _inputLayer._mapHeight;

	_inputLayer._maps.resize(inputNumMaps);

	for (size_t m = 0; m < inputNumMaps; m++) {
		_inputLayer._maps[m]._outputs.clear();
		_inputLayer._maps[m]._outputs.assign(inputMapSize, 0.0f);
	}

	// First convolution layer
	{
		size_t l = 0;

		_convolutionLayers[l]._filterSizeWidth = layerDescs[l]._filterSizeWidth;
		_convolutionLayers[l]._filterSizeHeight = layerDescs[l]._filterSizeHeight;

		_convolutionLayers[l]._strideWidth = layerDescs[l]._strideWidth;
		_convolutionLayers[l]._strideHeight = layerDescs[l]._strideHeight;

		size_t numConnectionsPerNode = _convolutionLayers[l]._filterSizeWidth * _convolutionLayers[l]._filterSizeHeight * _inputLayer._maps.size();

		_convolutionLayers[l]._mapWidth = _inputLayer._mapWidth - _convolutionLayers[l]._filterSizeWidth + 1;
		_convolutionLayers[l]._mapHeight = _inputLayer._mapHeight - _convolutionLayers[l]._filterSizeHeight + 1;

		_convolutionLayers[l]._maps.resize(layerDescs[l]._numFeatureMaps);

		size_t numNodesPerMap = _convolutionLayers[l]._mapWidth * _convolutionLayers[l]._mapHeight;

		for (size_t m = 0; m < _convolutionLayers[l]._maps.size(); m++) {
			_convolutionLayers[l]._maps[m]._outputs.clear();
			_convolutionLayers[l]._maps[m]._outputs.assign(numNodesPerMap, 0.0f);

			Node &node = _convolutionLayers[l]._maps[m]._node;

			node._bias._weight = distWeight(generator);

			node._connections.resize(numConnectionsPerNode);

			for (size_t c = 0; c < numConnectionsPerNode; c++)
				node._connections[c]._weight = distWeight(generator);
		}
	}

	// First downsampling layer
	{
		size_t l = 0;

		_downsamplingLayers[l]._downsampleWidth = layerDescs[l]._downsampleWidth;
		_downsamplingLayers[l]._downsampleHeight = layerDescs[l]._downsampleHeight;

		_downsamplingLayers[l]._mapWidth = static_cast<size_t>(static_cast<float>(_convolutionLayers[l]._mapWidth) / _downsamplingLayers[l]._downsampleWidth);
		_downsamplingLayers[l]._mapHeight = static_cast<size_t>(static_cast<float>(_convolutionLayers[l]._mapHeight) / _downsamplingLayers[l]._downsampleHeight);

		size_t numNodesPerMap = _downsamplingLayers[l]._mapWidth * _downsamplingLayers[l]._mapHeight;

		_downsamplingLayers[l]._maps.resize(_convolutionLayers[l]._maps.size());

		for (size_t m = 0; m < _downsamplingLayers[l]._maps.size(); m++) {
			_downsamplingLayers[l]._maps[m]._outputs.clear();
			_downsamplingLayers[l]._maps[m]._outputs.assign(numNodesPerMap, 0.0f);
		}
	}

	// All other layers
	for (size_t l = 1; l < layerDescs.size(); l++) {
		size_t prevLayerIndex = l - 1;

		// ------------------------------ Convolutional Layer ------------------------------

		{
			_convolutionLayers[l]._filterSizeWidth = layerDescs[l]._filterSizeWidth;
			_convolutionLayers[l]._filterSizeHeight = layerDescs[l]._filterSizeHeight;

			_convolutionLayers[l]._strideWidth = layerDescs[l]._strideWidth;
			_convolutionLayers[l]._strideHeight = layerDescs[l]._strideHeight;

			size_t numConnectionsPerNode = _convolutionLayers[l]._filterSizeWidth * _convolutionLayers[l]._filterSizeHeight * _downsamplingLayers[prevLayerIndex]._maps.size();

			_convolutionLayers[l]._mapWidth = (_downsamplingLayers[prevLayerIndex]._mapWidth - _convolutionLayers[l]._filterSizeWidth + 1) / (_convolutionLayers[l]._strideWidth);
			_convolutionLayers[l]._mapHeight = (_downsamplingLayers[prevLayerIndex]._mapHeight - _convolutionLayers[l]._filterSizeHeight + 1) / (_convolutionLayers[l]._strideHeight);

			_convolutionLayers[l]._maps.resize(layerDescs[l]._numFeatureMaps);

			size_t numNodesPerMap = _convolutionLayers[l]._mapWidth * _convolutionLayers[l]._mapHeight;

			for (size_t m = 0; m < _convolutionLayers[l]._maps.size(); m++) {
				_convolutionLayers[l]._maps[m]._outputs.clear();
				_convolutionLayers[l]._maps[m]._outputs.assign(numNodesPerMap, 0.0f);

				Node &node = _convolutionLayers[l]._maps[m]._node;

				node._bias._weight = distWeight(generator);

				node._connections.resize(numConnectionsPerNode);

				for (size_t c = 0; c < numConnectionsPerNode; c++)
					node._connections[c]._weight = distWeight(generator);
			}
		}

		// ------------------------------ Downsampling Layer ------------------------------

		{
			_downsamplingLayers[l]._downsampleWidth = layerDescs[l]._downsampleWidth;
			_downsamplingLayers[l]._downsampleHeight = layerDescs[l]._downsampleHeight;

			_downsamplingLayers[l]._mapWidth = static_cast<size_t>(static_cast<float>(_convolutionLayers[l]._mapWidth) / _downsamplingLayers[l]._downsampleWidth);
			_downsamplingLayers[l]._mapHeight = static_cast<size_t>(static_cast<float>(_convolutionLayers[l]._mapHeight) / _downsamplingLayers[l]._downsampleHeight);

			size_t numNodesPerMap = _downsamplingLayers[l]._mapWidth * _downsamplingLayers[l]._mapHeight;

			_downsamplingLayers[l]._maps.resize(_convolutionLayers[l]._maps.size());

			for (size_t m = 0; m < _downsamplingLayers[l]._maps.size(); m++) {
				_downsamplingLayers[l]._maps[m]._outputs.clear();
				_downsamplingLayers[l]._maps[m]._outputs.assign(numNodesPerMap, 0.0f);
			}
		}
	}
}

void ConvNet2D::activate() {
	// First convolution layer
	{
		size_t l = 0;

		// Convolve
		for (size_t m = 0; m < _convolutionLayers[l]._maps.size(); m++)
		for (size_t nx = 0; nx < _convolutionLayers[l]._mapWidth; nx++)
		for (size_t ny = 0; ny < _convolutionLayers[l]._mapHeight; ny++) {
			Node &node = _convolutionLayers[l]._maps[m]._node;

			size_t ix = nx * _convolutionLayers[l]._strideWidth;
			size_t iy = ny * _convolutionLayers[l]._strideHeight;

			// Go through filter
			float sum = node._bias._weight;

			for (size_t fm = 0; fm < _inputLayer._maps.size(); fm++)
			for (size_t fx = 0; fx < _convolutionLayers[l]._filterSizeWidth; fx++)
			for (size_t fy = 0; fy < _convolutionLayers[l]._filterSizeHeight; fy++) {
				size_t tx = ix + fx;
				size_t ty = iy + fy;

				if (tx < _inputLayer._mapWidth && ty < _inputLayer._mapHeight)
					sum += node._connections[fx + fy * _convolutionLayers[l]._filterSizeWidth]._weight * _inputLayer._maps[fm]._outputs[tx + ty * _inputLayer._mapWidth];
			}

			_convolutionLayers[l]._maps[m]._outputs[nx + ny * _convolutionLayers[l]._mapWidth] = sigmoid(sum);
		}
	}

	// First downsampling layer
	{
		size_t l = 0;

		for (size_t m = 0; m < _downsamplingLayers[l]._maps.size(); m++)
		for (size_t nx = 0; nx < _downsamplingLayers[l]._mapWidth; nx++)
		for (size_t ny = 0; ny < _downsamplingLayers[l]._mapHeight; ny++) {
			size_t ix = nx * _downsamplingLayers[l]._downsampleWidth;
			size_t iy = ny * _downsamplingLayers[l]._downsampleHeight;

			// Go through filter
			float maximum = -999999.0f;

			for (size_t fx = 0; fx < _downsamplingLayers[l]._downsampleWidth; fx++)
			for (size_t fy = 0; fy < _downsamplingLayers[l]._downsampleHeight; fy++) {
				size_t tx = ix + fx;
				size_t ty = iy + fy;

				// Make sure it is in bounds
				if (tx < _convolutionLayers[l]._mapWidth && ty < _convolutionLayers[l]._mapHeight)
					maximum = std::max(maximum, _convolutionLayers[l]._maps[m]._outputs[tx + ty * _convolutionLayers[l]._mapWidth]);
			}

			_downsamplingLayers[l]._maps[m]._outputs[nx + ny * _downsamplingLayers[l]._mapWidth] = maximum;
		}
	}

	for (size_t l = 1; l < _convolutionLayers.size(); l++) {
		size_t prevLayerIndex = l - 1;

		// ------------------------------ Convolutional Layer ------------------------------

		{
			// Convolve
			for (size_t m = 0; m < _convolutionLayers[l]._maps.size(); m++)
			for (size_t nx = 0; nx < _convolutionLayers[l]._mapWidth; nx++)
			for (size_t ny = 0; ny < _convolutionLayers[l]._mapHeight; ny++) {
				Node &node = _convolutionLayers[l]._maps[m]._node;

				size_t ix = nx * _convolutionLayers[l]._strideWidth;
				size_t iy = ny * _convolutionLayers[l]._strideHeight;

				// Go through filter
				float sum = node._bias._weight;

				for (size_t fm = 0; fm < _downsamplingLayers[prevLayerIndex]._maps.size(); fm++)
				for (size_t fx = 0; fx < _convolutionLayers[l]._filterSizeWidth; fx++)
				for (size_t fy = 0; fy < _convolutionLayers[l]._filterSizeHeight; fy++) {
					size_t tx = ix + fx;
					size_t ty = iy + fy;

					if (tx < _downsamplingLayers[prevLayerIndex]._mapWidth && ty < _downsamplingLayers[prevLayerIndex]._mapHeight)
						sum += node._connections[fx + fy * _convolutionLayers[l]._filterSizeWidth]._weight * _downsamplingLayers[prevLayerIndex]._maps[fm]._outputs[tx + ty * _downsamplingLayers[prevLayerIndex]._mapWidth];
				}

				_convolutionLayers[l]._maps[m]._outputs[nx + ny * _convolutionLayers[l]._mapWidth] = sigmoid(sum);
			}
		}

		// ------------------------------ Downsampling Layer ------------------------------

		{
			for (size_t m = 0; m < _downsamplingLayers[l]._maps.size(); m++)
			for (size_t nx = 0; nx < _downsamplingLayers[l]._mapWidth; nx++)
			for (size_t ny = 0; ny < _downsamplingLayers[l]._mapHeight; ny++) {
				size_t ix = nx * _downsamplingLayers[l]._downsampleWidth;
				size_t iy = ny * _downsamplingLayers[l]._downsampleHeight;

				// Go through filter
				float maximum = -999999.0f;

				for (size_t fx = 0; fx < _downsamplingLayers[l]._downsampleWidth; fx++)
				for (size_t fy = 0; fy < _downsamplingLayers[l]._downsampleHeight; fy++) {
					size_t tx = ix + fx;
					size_t ty = iy + fy;

					// Make sure it is in bounds
					if (tx < _convolutionLayers[l]._mapWidth && ty < _convolutionLayers[l]._mapHeight)
						maximum = std::max(maximum, _convolutionLayers[l]._maps[m]._outputs[tx + ty * _convolutionLayers[l]._mapWidth]);
				}

				_downsamplingLayers[l]._maps[m]._outputs[nx + ny * _downsamplingLayers[l]._mapWidth] = maximum;
			}
		}
	}
}