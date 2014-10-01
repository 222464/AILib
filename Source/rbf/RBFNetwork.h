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

namespace rbf {
	class RBFNetwork {
	public:
		struct RBFNode {
			std::vector<float> _center;

			float _width;
		};

		struct OutputNode {
			std::vector<float> _weights;

			float _bias;
		};

	private:
		std::vector<RBFNode> _rbfNodes;
		std::vector<OutputNode> _outputNodes;

	public:
		void createRandom(int numInputs, int numRBF, int numOutputs, float minCenter, float maxCenter, float minWidth, float maxWidth, float minWeight, float maxWeight, std::mt19937 &generator);

		void getOutput(const std::vector<float> &input, std::vector<float> &output) {
			std::vector<float> rbfOutputs(_rbfNodes.size());

			getOutput(input, rbfOutputs, output);
		}

		void getOutput(const std::vector<float> &input, std::vector<float> &rbfOutputs, std::vector<float> &output);

		void update(const std::vector<float> &input, std::vector<float> &output, const std::vector<float> &target, float centerAlpha, float widthAlpha, float weightAlpha);
	
		int getNumInputs() const {
			return _rbfNodes[0]._center.size();
		}

		int getNumOutputs() const {
			return _outputNodes.size();
		}

		int getNumRBFNodes() const {
			return _rbfNodes.size();
		}
	};
}