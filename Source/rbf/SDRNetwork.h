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
#include <algorithm>

#include <SFML/Graphics.hpp>

namespace sdr {
	class SDRNetwork {
	public:
		struct BackConnection {
			int _nodeIndex;
			int _weightIndex;
		};

		struct BackWeight {
			float _weight;
			float _prevDWeight;

			BackWeight()
				: _prevDWeight(0.0f)
			{}
		};

		struct Node {
			std::vector<float> _sdrWeights;
			std::vector<float> _sdrInhibition;
			std::vector<BackWeight> _backWeights;
			BackWeight _backBias;

			float _sdrBias;

			std::vector<BackConnection> _backConnections;

			float _sdrActivation;
			float _sdrOutput;

			float _backOutput;
			float _backSig;

			float _backError;

			Node()
				: _sdrActivation(0.0f), _sdrOutput(0.0f), _backOutput(0.0f), _backSig(0.0f), _backError(0.0f)
			{}
		};

		struct LayerDesc {
			int _width, _height;
			int _receptiveRadius;
			int _inhibitionRadius;

			float _sparsity;

			float _sdrActivationLeak;

			float _similarityDistanceFactor;

			LayerDesc()
				: _width(32), _height(32), _receptiveRadius(4), _inhibitionRadius(4), _sparsity(8.0f / 121.0f),
				_sdrActivationLeak(0.02f), _similarityDistanceFactor(0.1f)
			{}
		};

		struct Layer {
			std::vector<Node> _nodes;	
		};

		struct OutputConnection {
			float _weight;
		};

		struct OutputNode {
			std::vector<OutputConnection> _connections;

			OutputConnection _bias;

			float _error;
		};

		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

	private:
		std::vector<Layer> _layers;
		std::vector<LayerDesc> _layerDescs;
		std::vector<OutputNode> _outputNodes;

		int _inputWidth, _inputHeight;

	public:
		void createRandom(int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, int numOutputs, float minSDRWeight, float maxSDRWeight, float minInhibitionWeight, float maxInhibitionWeight, float minBackWeight, float maxBackWeight, std::mt19937 &generator);

		void getOutput(const std::vector<float> &input, std::vector<float> &output, std::mt19937 &generator);
	
		void updateUnsupervised(const std::vector<float> &input, float sdrWeightAlpha, float inhibitionAlpha, float biasAlpha);
		void updateSupervised(const std::vector<float> &input, const std::vector<float> &output, const std::vector<float> &target, float backWeightAlpha, float backWeightOutputLayerAlpha, float momentum);

		void getImages(std::vector<sf::Image> &images);
		void getReceptiveFields(int layer, sf::Image &image);

		int getNumOutputs() const {
			return _outputNodes.size();
		}
	};
}