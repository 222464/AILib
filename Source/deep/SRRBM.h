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
	class SRRBM {
	public:
		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

	private:
		struct Connection {
			float _weight;

			Connection()
			{}
		};

		struct HiddenNode {
			std::vector<Connection> _visibleHiddenConnections;
			std::vector<Connection> _hiddenHiddenConnections;

			float _state;
			float _probability;

			HiddenNode()
				: _state(0.0f), _probability(0.0f)
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

	public:
		void createRandom(int numVisible, int numHidden, float minWeight, float maxWeight, std::mt19937 &generator);

		void activate(std::mt19937 &generator);
		void learn(float alpha);

		void setVisibleInput(int index, float value) {
			_visible[index]._input = value;
		}

		float getHiddenProbability(int index) const {
			return _hidden[index]._probability;
		}

		int getNumVisible() const {
			return _visible.size();
		}

		int getNumHidden() const {
			return _hidden.size();
		}

		friend class DBN;
	};
}