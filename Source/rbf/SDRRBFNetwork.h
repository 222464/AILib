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

namespace rbf {
	class SDRRBFNetwork {
	public:
		struct RBFNode {
			std::vector<float> _center;

			float _width;

			float _activation;
			float _output;

			float _dutyCycle;
			float _minDutyCycle;

			RBFNode()
				: _activation(0.0f), _output(0.0f), _dutyCycle(0.5f), _minDutyCycle(0.0f)
			{}
		};

		struct LayerDesc {
			int _rbfWidth, _rbfHeight;
			int _receptiveRadius;
			int _inhibitionRadius;

			float _outputMultiplier;

			LayerDesc()
				: _rbfWidth(32), _rbfHeight(32), _receptiveRadius(3), _inhibitionRadius(5), _outputMultiplier(1.0f)
			{}
		};

		struct Layer {
			std::vector<RBFNode> _rbfNodes;	
		};

		struct Connection {
			float _weight;
			float _eligibility;

			Connection()
				: _eligibility(0.0f)
			{}
		};

		struct OutputNode {
			std::vector<Connection> _connections;

			Connection _bias;
		};

		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

		static float boostFunction(float active, float minimum) {
			return (1.0f - minimum) + std::max(0.0f, -(minimum - active));
		}

	private:
		std::vector<Layer> _layers;
		std::vector<LayerDesc> _layerDescs;
		std::vector<OutputNode> _outputNodes;

		int _inputWidth, _inputHeight;

	public:
		void createRandom(int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, int numOutputs, float minCenter, float maxCenter, float minWidth, float maxWidth, float minWeight, float maxWeight, std::mt19937 &generator);

		void getOutput(const std::vector<float> &input, std::vector<float> &output, float localActivity, float activationIntensity, float outputIntensity, float minDutyCycleRatio, float dutyCycleDecay, float randomFireChance, float randomFireStrength, std::mt19937 &generator);
	
		void update(const std::vector<float> &input, std::vector<float> &output, const std::vector<float> &target, float weightAlpha, float centerAlpha, float widthAlpha, float widthScalar, float minDistance, float minLearningThreshold);

		int getNumOutputs() const {
			return _outputNodes.size();
		}
	};
}