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

#include <deep/RBM.h>
#include <vector>
#include <random>

namespace deep {
	class ConvNet2D {
	public:
		struct LayerPairDesc {
			size_t _filterSizeWidth, _filterSizeHeight;
			size_t _numFeatureMaps;
			size_t _strideWidth, _strideHeight;
			size_t _downsampleWidth, _downsampleHeight;

			LayerPairDesc()
				: _filterSizeWidth(4), _filterSizeHeight(4),
				_numFeatureMaps(6),
				_strideWidth(1), _strideHeight(1),
				_downsampleWidth(2), _downsampleHeight(2)
			{}
		};

		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

	private:
		struct Connection {
			float _weight;
			float _positive;
			float _negative;

			Connection()
				: _positive(0.0f),
				_negative(0.0f)
			{}
		};

		struct Node {
			std::vector<Connection> _connections;
			Connection _bias;
		};

		struct Map {
			std::vector<float> _outputs;

			Node _node;
		};

		struct MaxpoolMap {
			std::vector<float> _outputs;
		};

		struct InputMap {
			std::vector<float> _outputs;
		};

		struct ConvolutionLayer {		
			size_t _filterSizeWidth, _filterSizeHeight;
			size_t _strideWidth, _strideHeight;
			size_t _mapWidth, _mapHeight;

			std::vector<Map> _maps;
		};

		struct DownsamplingLayer {
			size_t _mapWidth, _mapHeight;
			size_t _downsampleWidth, _downsampleHeight;

			std::vector<MaxpoolMap> _maps;
		};

		struct InputLayer {
			size_t _mapWidth, _mapHeight;

			std::vector<InputMap> _maps;
		};

		std::vector<float> _input;

		InputLayer _inputLayer;

		std::vector<ConvolutionLayer> _convolutionLayers;
		std::vector<DownsamplingLayer> _downsamplingLayers;

	public:
		void createRandom(size_t inputMapWidth, size_t inputMapHeight, size_t inputNumMaps, const std::vector<LayerPairDesc> &layerDescs, float minWeight, float maxWeight, std::mt19937 &generator);

		void activate();
		void activateAndLearn();

		size_t getInputWidth() const {
			return _inputLayer._mapWidth;
		}

		size_t getInputHeight() const {
			return _inputLayer._mapHeight;
		}

		size_t getInputNumMaps() const {
			return _inputLayer._maps.size();
		}

		size_t getOutputWidth() const {
			return _downsamplingLayers.back()._mapWidth;
		}

		size_t getOutputHeight() const {
			return _downsamplingLayers.back()._mapHeight;
		}

		size_t getOutputNumMaps() const {
			return _downsamplingLayers.back()._maps.size();
		}

		void setInput(size_t x, size_t y, size_t m, float value) {
			_inputLayer._maps[m]._outputs[x + y * _inputLayer._mapWidth] = value;
		}

		float getOutput(size_t x, size_t y, size_t m) const {
			return _downsamplingLayers.back()._maps[m]._outputs[x + y * _downsamplingLayers.back()._mapWidth];
		}
	};
}