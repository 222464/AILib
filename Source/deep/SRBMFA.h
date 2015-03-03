#pragma once

#include <vector>
#include <random>
#include <functional>
#include <iostream>

namespace deep {
	class SRBMFA {
	public:
		struct LayerDesc {
			int _numHiddenUnits;
		};

	private:
		float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

		struct RBMConnection {
			float _rbmWeight;
			float _ffnnWeight;
			float _rbmPrevDWeight;
			float _ffnnPrevDWeight;

			RBMConnection()
				: _rbmPrevDWeight(0.0f), _ffnnPrevDWeight(0.0f)
			{}
		};

		struct OutputConnection {
			float _weight;
			float _prevDWeight;

			OutputConnection()
				: _prevDWeight(0.0f)
			{}
		};

		struct VisibleUnit {
			RBMConnection _bias;

			float _input;
			float _probability;

			VisibleUnit()
				: _input(0.0f), _probability(0.0f)
			{}
		};

		struct HiddenUnit {
			RBMConnection _bias;

			std::vector<RBMConnection> _connections;

			float _activation;
			float _firstProbability;
			float _probability;
			float _output;

			float _sig;
			float _ffnnOutput;
			float _error;

			HiddenUnit()
				: _activation(0.0f), _firstProbability(0.0f), _probability(0.0f), _output(0.0f), _sig(0.0f), _ffnnOutput(0.0f),_error(0.0f)
			{}
		};

		struct OutputUnit {
			OutputConnection _bias;
			
			std::vector<OutputConnection> _connections;

			float _output;
			float _target;

			float _error;

			OutputUnit()
				: _output(0.0f), _target(0.0f), _error(0.0f)
			{}
		};

		struct RBMLayer {
			std::vector<VisibleUnit> _visibleUnits;
			std::vector<HiddenUnit> _hiddenUnits;
		};

		std::vector<RBMLayer> _rbmLayers;
		std::vector<OutputUnit> _outputUnits;

	public:
		void createRandom(int numInputs, const std::vector<LayerDesc> &layerDescs, int numOutputUnits, float weightStdDev, std::mt19937 &generator);

		int getNumOutputUnits() const {
			return _outputUnits.size();
		}

		void setInput(int index, float value) {
			_rbmLayers.front()._visibleUnits[index]._input = value;
		}

		float getOutputUnitOutput(int index) const {
			return _outputUnits[index]._output;
		}

		void setOutputUnitTarget(int index, float target) {
			_outputUnits[index]._target = target;
		}

		void activate(float sparsity, int numLayersActivate = -1);

		std::vector<float> operator()(const std::vector<float> &input, float averageOutputDecay);

		void learnRBM(int layer, float rbmAlpha, float rbmBeta, float rbmMomentum, float columnRandomness, float sparsity, int gibbsSampleIterations, std::mt19937 &generator);
		void learnFFNN(float ffnnAlpha, float ffnnMomentum);

		friend std::ostream &operator<<(std::ostream &os, const SRBMFA &fa);
	};

	std::ostream &operator<<(std::ostream &os, const SRBMFA &fa);
}