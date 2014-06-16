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

#include <hypernet/Boid.h>
#include <hypernet/Encoder.h>
#include <hypernet/Decoder.h>
#include <tuple>

namespace hn {
	class HyperNet {
	private:
		hn::FunctionApproximator _linkProcessor;
		hn::FunctionApproximator _boidFiringProcessor;
		hn::FunctionApproximator _boidConnectProcessor;
		hn::FunctionApproximator _boidDisconnectProcessor;

		hn::FunctionApproximator _encoder;
		hn::FunctionApproximator _decoder;

		// Initialization ranges for memory
		std::vector<std::tuple<float, float>> _linkMemoryInitRange;
		std::vector<std::tuple<float, float>> _boidMemoryInitRange;
		std::vector<std::tuple<float, float>> _encoderMemoryInitRange;
		std::vector<std::tuple<float, float>> _decoderMemoryInitRange;

		std::vector<float> _linkProcessorInputBuffer;
		std::vector<float> _linkProcessorOutputBuffer;
		std::vector<float> _boidFiringProcessorInputBuffer;
		std::vector<float> _boidFiringProcessorOutputBuffer;
		std::vector<float> _boidConnectProcessorInputBuffer;
		std::vector<float> _boidConnectProcessorOutputBuffer;
		std::vector<float> _boidDisconnectProcessorInputBuffer;
		std::vector<float> _boidDisconnectProcessorOutputBuffer;

		std::vector<float> _encoderInputBuffer;
		std::vector<float> _encoderOutputBuffer;
		std::vector<float> _decoderInputBuffer;
		std::vector<float> _decoderOutputBuffer;

		std::vector<int> _dimensions;

		std::vector<std::shared_ptr<Boid>> _boids;

		std::vector<float> _inputs;
		std::vector<float> _outputs;

		std::vector<std::vector<size_t>> _inputIndices;
		std::vector<std::vector<size_t>> _outputIndices;

		std::vector<std::shared_ptr<Encoder>> _encoders;
		std::vector<std::shared_ptr<Decoder>> _decoders;

	public:
		bool _connectDisconnectEnabled;

		HyperNet();

		void createRandom(const Config &config, int preTrainIterations, float preTrainAlpha, float preTrainMin, float preTrainMax, std::mt19937 &generator, float activationMultiplier);

		void createFromParents(const HyperNet &parent1, const HyperNet &parent2,
			float weightAverageChance, float memoryAverageChance,
			std::mt19937 &generator);

		void mutate(float weightPerturbationChance, float weightPerturbationStdDev,
			float memoryPerturbationChance, float memoryPerturbationStdDev, std::mt19937 &generator);

		void createFromWeightsVector(const Config &config, const std::vector<float> &weights);
		void getWeightsVector(std::vector<float> &weights);

		void generateNetwork(const Config &config, const std::vector<int> &dimensions, std::mt19937 &generator);
		void generateFeedForward(const Config &config, size_t numHiddenLayers, size_t numBoidsPerHiddenLayer, std::mt19937 &generator);

		int getLinearCoordinate(const std::vector<int> &coordinates);
		void getMultiDimCoordinatesFromLinear(int linearCoordinate, std::vector<int> &coordinates);

		const Boid &getBoid(size_t index) const {
			return *_boids[index];
		}

		void setInput(size_t index, float value) {
			_inputs[index] = value;
		}

		float getInput(size_t index) const {
			return _inputs[index];
		}

		float getOutput(size_t index) const {
			return _outputs[index];
		}

		void step(const Config &config, float reward, std::mt19937 &generator, int substeps, float activationMultiplier);

		void writeToStream(std::ostream &os) const;
		void readFromStream(std::istream &is);

		friend class Boid;
		friend class Link;
		friend class Encoder;
		friend class Decoder;
	};
}