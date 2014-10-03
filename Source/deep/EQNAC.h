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

#include <deep/RBM.h>

namespace deep {
	class EQNAC {
	private:
		struct Connection {
			float _weight;
			float _eligibility;

			Connection()
				: _eligibility(0.0f)
			{}
		};

		struct Node {
			std::vector<Connection> _connections;
			Connection _bias;
		};

		int _numState;
		int _numAction;

		std::vector<Node> _outputNodes;
		Node _qNode;

		deep::RBM _rbm;

		float _baseline;

	public:
		EQNAC();

		void createRandom(int numState, int numAction, int numHidden, float minWeight, float maxWeight, std::mt19937 &generator);

		void step(float reward, float rbmAlpha, std::mt19937 &generator);
	};
}