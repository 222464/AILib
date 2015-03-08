#include <deep/RecurrentSparseAutoencoder.h>

#include <list>

namespace deep {
	class RSARL {
	private:
		struct QNode {
			float _q;
			float _trace;

			QNode()
				: _q(0.0f), _trace(0.0f)
			{}
		};

		struct Experience {
			std::vector<float> _hiddenStates;
			std::vector<float> _maxVisibleStates;
			std::vector<float> _visibleStates;

			float _originalQ;
			float _q;
		};

		deep::RecurrentSparseAutoencoder _rsa;

		std::vector<QNode> _qNodes;

		float _prevValue;

		int _numInputs;

		float _sparsity;

		std::vector<float> _outputs;
		std::vector<float> _maxOutputs;

		std::list<Experience> _experiences;

		void dqOverDo(std::vector<float> &deltaO);

	public:
		int _experienceBufferLength;

		RSARL()
			: _experienceBufferLength(300)
		{}

		void createRandom(int numInputs, int numOutputs, int numHidden, float sparsity, float minWeight, float maxWeight, float recurrentScalar, std::mt19937 &generator);

		void step(float reward, int actionSamples, int experienceSamples, float rsaStateLeak, float rsaAlpha, float rsaBeta, float rsaGamma, float rsaEpsilon, float rsaDutyCycleDecay, float rsaMomentum, float rsaTraceDecay, float rsaTemperature, float qTraceDecay, float qAlpha, float qUpdateAlpha, float qGamma, float breakChance, std::mt19937 &generator);

		void setInput(int index, float value) {
			_rsa.setVisibleNodeState(index, value);
		}

		float getOutput(int index) const {
			return _outputs[index];
		}

		int getNumInputs() const {
			return _numInputs;
		}

		int getNumOutputs() const {
			return _outputs.size();
		}

		int getNumHidden() const {
			return _rsa.getNumHiddenNodes();
		}

		const deep::RecurrentSparseAutoencoder &getRSA() const {
			return _rsa;
		}
	};
}