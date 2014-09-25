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

#include <vector>

namespace deep {
	class DBN {
	public:
		struct LinearNode {
			float _output;
			float _error;

			float _bias;
			float _prevDBias;
			float _accumulatedDBias;

			std::vector<float> _weights;
			std::vector<float> _prevDWeights;
			std::vector<float> _accumulatedDWeight;

			LinearNode()
				: _output(0.0f), _error(0.0f), _prevDBias(0.0f), _accumulatedDBias(0.0f)
			{}
		};
	private:
		std::vector<RBM> _rbmLayers;
		std::vector<std::vector<float>> _rbmErrors;

		std::vector<LinearNode> _outputNodes;

		std::vector<float> _input;

	public:
		void createRandom(int numInputs, int numOutputs, const std::vector<int> &rbmNumHiddens, float minWeight, float maxWeight, std::mt19937 &generator);
	
		void trainLayerUnsupervised(int layerIndex, const std::vector<float> &input, float alpha, std::mt19937 &generator);

		void getLayerOutputMean(int layerIndex, const std::vector<float> &input, std::vector<float> &mean);
		void getOutputMeanThroughLayers(int numLayers, const std::vector<float> &input, std::vector<float> &mean);

		void prepareForGradientDescent();

		void execute(const std::vector<float> &input, std::vector<float> &output);

		void getError(const std::vector<float> &target);

		void moveAlongGradient(float alpha, float momentum, float alphaLayerMuliplier);
		void signError();

		void accumulateGradient();
		void moveAlongAccumulatedGradient(float alpha);

		void decayWeights(float decayMultiplier);
		void decayWeightsLastLayerOnly(float decayMultiplier);

		int getNumLayers() const {
			return _rbmLayers.size();
		}
	};
}