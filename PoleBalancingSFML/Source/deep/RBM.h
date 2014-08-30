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
		static double sigmoid(double x) {
			return 1.0 / (1.0 + std::exp(-x));
		}

	private:
		struct Connection {
			double _weight;
			double _positive;
			double _negative;

			Connection()
				: _positive(0.0),
				_negative(0.0)
			{}
		};

		struct HiddenNode {
			std::vector<Connection> _connections;

			double _output;
			double _probability;

			HiddenNode()
				: _output(0.0), _probability(0.0)
			{}
		};

		struct VisibleNode {
			double _probability;

			VisibleNode()
				: _probability(0.0)
			{}
		};

		std::vector<VisibleNode> _visible;
		std::vector<HiddenNode> _hidden;

	public:
		void createRandom(size_t numVisible, size_t numHidden, double minWeight, double maxWeight, std::mt19937 &generator);

		void activate(std::mt19937 &generator);
		void activateLight();

		void learn(double alpha, std::mt19937 &generator);

		void setVisible(size_t index, double value) {
			_visible[index]._probability = value;
		}

		double getHidden(size_t index) const {
			return _hidden[index]._probability;
		}

		size_t getNumVisible() const {
			return _visible.size() - 1; // -1 to account for bias
		}

		size_t getNumHidden() const {
			return _hidden.size() - 1; // -1 to account for bias
		}

		friend class DBN;
	};
}