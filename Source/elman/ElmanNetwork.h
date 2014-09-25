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

namespace elman {
	class ElmanNetwork {
	public:
		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

	private:
		struct Connection {
			float _weight;
			float _trace;

			Connection()
				: _weight(0.0f),
				_trace(0.0f)
			{}
		};

		struct Node {
			std::vector<Connection> _connections;

			Connection _bias;

			float _output;

			float _error;

			Node()
				: _output(0.0f),
				_error(0.0f)
			{}
		};

		struct Context {
			float _output;
			float _error;

			float _hiddenBias;

			Context()
				: _output(0.5f),
				_error(0.0f),
				_hiddenBias(0.0f)
			{}
		};

		std::vector<float> _input;
		std::vector<Node> _hidden;
		std::vector<Context> _context;
		std::vector<Node> _output;

	public:
		void createRandom(size_t numInputs, size_t numOutputs, size_t numHidden, float minWeight, float maxWeight, std::mt19937 &generator);
	
		void activate();
		void updateContext();

		void calculateGradient(std::vector<float> &targets);
		void moveAlongGradient(float alpha);

		void setInput(size_t index, float value) {
			_input[index] = value;
		}

		float getInput(size_t index) const {
			return _input[index];
		}

		float getOutput(size_t index) const {
			return _output[index]._output;
		}

		size_t getNumInputs() const {
			return _input.size();
		}

		size_t getNumOutputs() const {
			return _output.size();
		}
	};
}