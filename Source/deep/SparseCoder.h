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

namespace deep {
	class SparseCoder {
	private:
		struct Connection {
			float _weight;

			Connection()
			{}
		};

		struct HiddenNode {
			std::vector<Connection> _visibleHiddenConnections;
			std::vector<Connection> _hiddenHiddenConnections;

			Connection _bias;

			float _state;
			float _statePrev;
			float _activation;
			float _activationPrev;

			HiddenNode()
				: _state(0.0f), _statePrev(0.0f), _activation(0.0f), _activationPrev(0.0f)
			{}
		};

		struct VisibleNode {
			float _input;
			float _reconstruction;

			VisibleNode()
				: _input(0.0f), _reconstruction(0.0f)
			{}
		};

		std::vector<VisibleNode> _visible;
		std::vector<HiddenNode> _hidden;

		float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

	public:
		void createRandom(int numVisible, int numHidden, std::mt19937 &generator);

		void activate(float sparsity, float lambda, float dt);
		void reconstruct();
		void learn(float alpha, float beta, float gamma, float sparsity);
		void stepEnd();

		void setVisibleInput(int index, float value) {
			_visible[index]._input = value;
		}

		float getHiddenState(int index) const {
			return _hidden[index]._state;
		}

		int getNumVisible() const {
			return _visible.size();
		}

		int getNumHidden() const {
			return _hidden.size();
		}

		float getVHWeight(int hi, int vi) const {
			return _hidden[hi]._visibleHiddenConnections[vi]._weight;
		}
	};
}