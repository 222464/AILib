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
	class SDRRBFNetwork {
	public:
		struct BackConnection {
			int _nodeIndex;
			int _weightIndex;
		};

		struct Weight {
			float _weight;
			float _prevDWeight;

			Weight()
				: _prevDWeight(0.0f)
			{}
		};

		struct RBFNode {
			std::vector<float> _center;
			std::vector<Weight> _weights;
			Weight _bias;

			std::vector<BackConnection> _backConnections;

			float _width;

			float _rbfActivation;
			float _rbfOutput;

			float _output;
			float _sig;

			float _dutyCycle;
			float _dutyCyclePrev;

			float _error;

			float _learn;

			RBFNode()
				: _rbfActivation(0.0f), _rbfOutput(0.0f), _dutyCycle(1.0f), _dutyCyclePrev(1.0f), _output(0.0f), _sig(0.0f), _error(0.0f), _learn(0.0f)
			{}
		};

		struct LayerDesc {
			int _rbfWidth, _rbfHeight;
			int _receptiveRadius;
			int _inhibitionRadius;

			float _outputMultiplier;

			float _localActivity;
			float _outputIntensity;

			float _uniquenessPower;

			float _learnIntensity;
			float _minLearn;

			float _activationRangeDecayIntensity;

			float _weightAlpha;

			float _tolerance;

			LayerDesc()
				: _rbfWidth(32), _rbfHeight(32), _receptiveRadius(2), _inhibitionRadius(2), _outputMultiplier(1.0f), _localActivity(8.0f), _outputIntensity(4.0f), _uniquenessPower(4.0f),
				_learnIntensity(4.0f), _minLearn(0.0f), _activationRangeDecayIntensity(0.1f), _weightAlpha(0.005f), _tolerance(0.05f)
			{}
		};

		struct Layer {
			std::vector<RBFNode> _rbfNodes;	
		};

		struct Connection {
			float _weight;
		};

		struct OutputNode {
			std::vector<Connection> _connections;

			Connection _bias;

			float _error;
		};

		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

		static float boostFunction(float active, float minimum) {
			return std::max(0.0f, minimum - active) / minimum;
		}

		static float relu(float x) {
			return std::max(0.0f, x);
		}

		static float reluPrime(float x) {
			return x > 0.0f ? 1.0f : 0.01f;
		}

	private:
		std::vector<Layer> _layers;
		std::vector<LayerDesc> _layerDescs;
		std::vector<OutputNode> _outputNodes;

		int _inputWidth, _inputHeight;

	public:
		void createRandom(int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, int numOutputs, float minCenter, float maxCenter, float minWidth, float maxWidth, float minWeight, float maxWeight, std::mt19937 &generator);

		void getOutput(const std::vector<float> &input, std::vector<float> &output, float activationIntensity, float dutyCycleDecay, float randomFireChance, float randomFireStrength, float minDistance, float minDutyCycle, std::mt19937 &generator);
	
		void updateUnsupervised(const std::vector<float> &input, float weightAlpha, float centerAlpha, float widthAlpha, float widthScalar, float minDistance, float minDutyCycle, float momentum);
		void updateSupervised(const std::vector<float> &input, const std::vector<float> &output, const std::vector<float> &target, float weightAlpha, float centerAlpha, float widthAlpha, float widthScalar, float minDistance, float minDutyCycle, float momentum);

		void getImages(std::vector<sf::Image> &images);

		int getNumOutputs() const {
			return _outputNodes.size();
		}
	};
}