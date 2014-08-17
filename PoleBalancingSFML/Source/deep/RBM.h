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
	class RBM {
	public:
		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

	private:
		struct Connection {
			float _weight;
			float _positive;
			float _negative;

			Connection()
				: _positive(0.0f),
				_negative(0.0f)
			{}
		};

		struct HiddenNode {
			std::vector<Connection> _connections;

			float _output;

			Connection _bias;

			HiddenNode()
				: _output(0.0f)
			{}
		};

		struct VisibleNode {
			float _output;

			Connection _bias;

			VisibleNode()
				: _output(0.0f)
			{}
		};

		std::vector<VisibleNode> _visible;
		std::vector<HiddenNode> _hidden;

	public:
		void createRandom(size_t numVisible, size_t numHidden, float minWeight, float maxWeight, std::mt19937 &generator);

		void activate(std::mt19937 &generator);

		void learn(float alpha, std::mt19937 &generator);

		void setVisible(size_t index, float value) {
			_visible[index]._output = value;
		}

		float getHidden(size_t index) const {
			return _hidden[index]._output;
		}

		size_t getNumVisible() const {
			return _visible.size();
		}

		size_t getNumHidden() const {
			return _hidden.size();
		}
	};
}