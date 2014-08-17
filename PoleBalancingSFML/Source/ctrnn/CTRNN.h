#pragma once

#include <vector>
#include <random>

namespace ctrnn {
	class CTRNN {
	public:
		struct Node {
			float _bias;
			float _state;

			float _tauInv;
			float _noiseStdDev;

			float _prevOutput;
			float _output;
		};

		struct Weight {
			float _weight;
			float _trace;
		};

		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

	private:
		std::vector<Weight> _weightMatrix;
		std::vector<Node> _nodes;
		std::vector<float> _inputs;

		size_t _numOutputs, _numHidden;
		size_t _numNodesTotal, _numNodesHiddenOutput;

		Weight &getWeight(size_t i, size_t j) {
			return _weightMatrix[j + i * _numNodesTotal];
		}

	public:
		void createRandom(size_t numInputs, size_t numOutputs, size_t numHidden, float minWeight, float maxWeight, float minTau, float maxTau, float minNoiseStdDev, float maxNoiseStdDev, std::mt19937 &generator);
		void createFromParents(const CTRNN &parent1, const CTRNN &parent2, float averageWeightsChance, float averageTausChance, float averageNoiseStdDevChance, std::mt19937 &generator);

		void mutate(float weightPerturbationChance, float maxWeightPerturbation, float tauPerturbationChance, float maxTauPerturbation, float noiseStdDevPerturbationChance, float maxNoiseStdDevPerturbation, std::mt19937 &generator);

		void setInput(size_t index, float value) {
			_inputs[index] = value;
		}

		float getOutput(size_t index) const {
			return _nodes[_numHidden + index]._output;
		}

		void clear();

		void step(float dt, float reward, float traceDecay, std::mt19937 &generator);
	};
}