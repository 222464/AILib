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
			double _output;
			double _error;

			double _bias;
			double _prevDBias;
			double _accumulatedDBias;

			std::vector<double> _weights;
			std::vector<double> _prevDWeights;
			std::vector<double> _accumulatedDWeight;

			LinearNode()
				: _output(0.0), _error(0.0), _prevDBias(0.0), _accumulatedDBias(0.0)
			{}
		};
	private:
		std::vector<RBM> _rbmLayers;
		std::vector<std::vector<double>> _rbmErrors;

		std::vector<LinearNode> _outputNodes;

		std::vector<double> _input;

	public:
		void createRandom(size_t numInputs, size_t numOutputs, const std::vector<size_t> &rbmNumHiddens, double minWeight, double maxWeight, std::mt19937 &generator);
	
		void trainLayerUnsupervised(size_t layerIndex, const std::vector<double> &input, double alpha, std::mt19937 &generator);

		void getLayerOutputMean(size_t layerIndex, const std::vector<double> &input, std::vector<double> &mean);
		void getOutputMeanThroughLayers(size_t numLayers, const std::vector<double> &input, std::vector<double> &mean);

		void prepareForGradientDescent();

		void execute(const std::vector<double> &input, std::vector<double> &output);

		void getError(const std::vector<double> &target);

		void moveAlongGradient(double alpha, double momentum, double alphaLayerMuliplier);
		void signError();

		void accumulateGradient();
		void moveAlongAccumulatedGradient(double alpha);

		void decayWeights(float decayMultiplier);
		void decayWeightsLastLayerOnly(float decayMultiplier);

		size_t getNumLayers() const {
			return _rbmLayers.size();
		}
	};
}