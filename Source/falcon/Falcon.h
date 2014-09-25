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
#include <list>
#include <array>
#include <random>

namespace falcon {
	class Falcon {
	public:
		enum Field {
			_inputField = 0, _outputField, _rewardField
		};

		struct FieldParams {
			float _alpha;
			float _beta;
			float _gamma;
			float _baseVigilance;

			FieldParams()
				: _alpha(0.01f),
				_beta(0.4f),
				_gamma(0.333f),
				_baseVigilance(0.25f)
			{}
		};

		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

		static float logit(float x) {
			return std::log(x / (1.0f - x));
		}

	private:
		struct Node {
			std::vector<float> _weights;

			bool _committed;

			float _eligibility;
		};

		std::list<Node> _nodes;

		std::vector<float> _inputs;
		std::vector<float> _outputs;

		size_t _artInputSize;

		float _prevQ;
		std::vector<float> _prevInputs;
		std::vector<float> _prevOutputs;

		std::list<Node>::iterator findNode(const std::vector<float> &artInputs, const std::array<FieldParams, 3> &fieldParams, std::list<Node> &nodes, bool ignoreOutput, bool rewardChoice, bool runTournament, float tournamentRatio);
		float calculateChoice(const Node &node, float minReward, float maxReward, const std::vector<float> &artInputs, const std::array<FieldParams, 3> &fieldParams, bool ignoreOutput, bool rewardChoice) const;
		bool calculateMatch(const Node &node, const std::vector<float> &artInputs, std::array<float, 3> &vigilances) const;
		void learn(const std::vector<float> &artInputs, const std::array<FieldParams, 3> &fieldParams);

	public:
		Falcon();

		void create(size_t numInputs, size_t numOutputs);

		void update(float reward, float epsilon, float gamma, float alpha, std::array<FieldParams, 3> &fieldParams, float rewardFactor, float eligibilityDecay, float tournamentRatio, std::mt19937 &generator);

		void setInput(size_t index, float value) {
			_inputs[index] = value;
		}

		float getInput(size_t index) const {
			return _inputs[index];
		}

		float getOutput(size_t index) const {
			return _outputs[index];
		}

		size_t getNumInputs() const {
			return _inputs.size();
		}

		size_t getNumOutputs() const {
			return _outputs.size();
		}

		size_t getNumNodes() const {
			return _nodes.size();
		}
	};
}