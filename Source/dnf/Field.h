#pragma once

#include <vector>
#include <random>
#include <iostream>

#include <Consts.h>

namespace dnf {
	class Field {
	public:
		struct Connection {
			float _weight;
			float _eligibilityTrace;
		};

		struct Node {
			std::vector<Connection> _wee;
			std::vector<Connection> _wei;
			std::vector<Connection> _wie;

			std::vector<Connection> _wext;

			float _activationE;
			float _outputE;
			float _averageRateE;
			float _bdnfE;

			float _activationI;
			float _outputI;
			//float _averageRateI;
			//float _bdnfI;

			float _hE;
		};

		struct InputData {
			float _averageRate;
			float _bdnf;
		};

		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

		static float bdnf(float average, float target, float learningRate) {
			return 1.0f + learningRate * (average - target) / target;
		}

	private:
		int _numInputs;
		int _size;
		int _weightRadius;
		float _weightDeviation;

		std::vector<Node> _nodes;

		std::vector<float> _gLookup;

		std::vector<InputData> _inputData;

	public:
		float _hI;

		Field();

		void createRandom(int numInputs, int size, int weightRadius, float weightDeviation, float minWeight, float maxWeight, std::mt19937 &generator);

		void step(const std::vector<float> &input, float dt, float threshold, float gain, float reward, float eligibilityDecay,
			float averageDecay, float homeoAlphaH, float homeoAlphaT, float targetActivation);

		int getNumInputs() const {
			return _numInputs;
		}

		int getSize() const {
			return _size;
		}

		int getWeightRadius() const {
			return _weightRadius;
		}

		float getOutputE(int x, int y) const {
			return _nodes[x + y * _size]._outputE;
		}
	};
}