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

#include <nn/Neuron.h>
#include <nn/BrownianPerturbation.h>

namespace nn {
	class FeedForwardNeuralNetwork {
	public:
		// Structure for gradient of network with respect to error in outputs
		struct Gradient {
			std::vector<float> _outputGradient;
			std::vector<std::vector<float>> _hiddenLayersGradient;

			// Running average with decay
			void decayTowards(float decay, const Gradient &other);
			void setValue(float value);
		};

		// Structure for random perturbations for each output in the network
		struct BrownianPerturbationSet {
			std::mt19937 _generator;

			std::vector<BrownianPerturbation> _outputPerturbations;
			std::vector<std::vector<BrownianPerturbation>> _hiddenLayerPerturbations;

			void update(float dt);
			void setAll(const BrownianPerturbation &perturbation);
			void reset();
		};

	private:
		std::vector<Sensor> _inputs;
		std::vector<Neuron> _outputs;
		std::vector<std::vector<Neuron>> _hidden;

	public:
		float _activationMultiplier; // Sensitivity of neurons
		float _outputTraceDecay; // Decay rate of output traces
		float _weightTraceDecay; // Decay rate of weight traces

		FeedForwardNeuralNetwork()
			: _activationMultiplier(1.0f), _outputTraceDecay(0.01f), _weightTraceDecay(0.01f)
		{}

		FeedForwardNeuralNetwork(const FeedForwardNeuralNetwork &other) {
			*this = other;
		}

		const FeedForwardNeuralNetwork &operator=(const FeedForwardNeuralNetwork &other);

		// Create neural network with random weights
		void createRandom(size_t numInputs, size_t numOutputs,
			size_t numHiddenLayers, size_t numNeuronsPerHiddenLayer,
			float minWeight, float maxWeight, std::mt19937 &generator);

		// For genetic algorithm
		void createFromParents(const FeedForwardNeuralNetwork &parent1, const FeedForwardNeuralNetwork &parent2,
			float averageWeightsChance, unsigned long seed);

		void mutate(float weightMutationChance, float maxWeightPerturbation, unsigned long seed);

		// Activations
		void activate();
		void activateTraceless();
		void activateAndReinforce(float error);
		void activateAndReinforceTraceless(float error);
		void activateLinearOutputLayer();

		void activateArp(std::mt19937 &generator);

		// Stand-alone reinforce
		void reinforce(float error);
		void reinforceTraceless(float error);

		void reinforceArp(float reward, float alpha, float lambda);
		void reinforceArpWithTraces(float reward, float alpha, float lambda);
		void reinforceArpMomentum(float reward, float alpha, float lambda, float momentum);

		// Set both weight memory and output traces to 0
		void zeroTraces();

		// Vanilla backpropagation algorithm
		void getGradient(const std::vector<float> &targets, Gradient &grad);
		void getGradientLinearOutputLayer(const std::vector<float> &targets, Gradient &grad);
		void getGradientFromError(const std::vector<float> &error, Gradient &grad);

		void getEmptyGradient(Gradient &grad);

		void moveAlongGradient(const Gradient &grad, float alpha);
		void moveAlongGradientSign(const Gradient &grad, float alpha);
		void moveAlongGradientMomentum(const Gradient &grad, float alpha, float momentum);

		void storeCurrentOutputsInTraces();
		void updateValueFunction(float tdError, float alpha, float traceDecay);

		void decayWeights(float decay);
		void decayWeightsExcludingOutputLayer(float decay);

		void getInputGradient(const Gradient &existingGrad, std::vector<float> &inputGrad);

		void getBrownianPerturbationSet(BrownianPerturbationSet &set);

		void writeToStream(std::ostream &stream);
		void readFromStream(std::istream &stream);

		void getWeightVector(std::vector<float> &weights);
		void setWeightVector(const std::vector<float> &weights);
		size_t getWeightVectorSize() const;

		size_t getNumInputs() const {
			return _inputs.size();
		}

		size_t getNumOutputs() const {
			return _outputs.size();
		}

		size_t getNumHiddenLayers() const {
			return _hidden.size();
		}

		size_t getNumNeuronsPerHiddenLayer() const {
			return getNumHiddenLayers() == 0 ? 0 : _hidden[0].size();
		}

		float getInput(size_t i) const {
			return _inputs[i]._output;
		}

		void setInput(size_t i, float value) {
			_inputs[i]._output = value;
		}

		float getOutput(size_t i) const {
			return _outputs[i]._output;
		}

		float getHiddenOutput(size_t l, size_t i) const {
			return _hidden[l][i]._output;
		}
	};
}