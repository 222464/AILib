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

namespace rbf {
	/*class CHTMRL {
	public:
		struct BackConnection {
			int _nodeIndex;
			unsigned char _contextIndex;
			unsigned short _weightIndex;
		};

		struct FFNNConnection {
			float _weight;
			float _trace;

			FFNNConnection()
				: _trace(0.0f)
			{}
		};

		struct ContextNode {
			std::vector<float> _contextWeights;
			float _contextBias;

			std::vector<FFNNConnection> _ffnnConnections;
			FFNNConnection _ffnnBias;

			std::vector<BackConnection> _backConnections;

			float _contextOutput;
			float _contextPrediction;

			float _ffnnOutput;
			float _ffnnSquashed;

			float _ffnnError;
		};

		struct RBFNode {
			std::vector<float> _center;

			std::vector<ContextNode> _contextNodes;

			float _rbfActivation;
			float _rbfOutput;
			float _rbfPrediction;

			float _dutyCycle;

			RBFNode()
				: _rbfActivation(0.0f), _rbfOutput(0.0f), _rbfPrediction(0.0f), _dutyCycle(1.0f)
			{}
		};

		struct LayerDesc {
			int _rbfWidth, _rbfHeight;
			int _numContextNodes;

			int _rbfReceptiveRadius;
			int _ffnnReceptiveRadius;
			int _contextReceptiveRadius;
			int _inhibitionRadius;

			float _rbfLocalActivity;
			float _rbfOutputIntensity;

			float _rbfCenterAlpha;
			float _ffnnWeightAlpha;
			float _contextWeightAlpha;	

			LayerDesc()
				: _rbfWidth(32), _rbfHeight(32), _numContextNodes(4),
				_rbfReceptiveRadius(3), _ffnnReceptiveRadius(3), _contextReceptiveRadius(3), _inhibitionRadius(5),
				_rbfLocalActivity(8.0f), _rbfOutputIntensity(0.01f),
				_rbfCenterAlpha(0.01f), _ffnnWeightAlpha(0.01f), _contextWeightAlpha(0.01f)
			{}
		};

		struct Layer {
			std::vector<RBFNode> _rbfNodes;	
		};

		struct OutputNode {
			std::vector<FFNNConnection> _connections;

			FFNNConnection _bias;

			float _error;
		};

		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

		static float boostFunction(float active, float minimum) {
			return std::max(0.0f, minimum - active) / minimum;
		}

	private:
		std::vector<Layer> _layers;
		std::vector<LayerDesc> _layerDescs;
		
		OutputNode _qOutput;

		int _inputWidth, _inputHeight;

	public:
		void createRandom(int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, int numOutputs, float minCenter, float maxCenter, float minContextWeight, float maxContextWeight, float minFFNNWeight, float maxFFNNWeight, std::mt19937 &generator);

		void step(const std::vector<float> &input, std::vector<float> &output, float alpha);
	};*/
}