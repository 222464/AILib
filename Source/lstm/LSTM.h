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

namespace lstm {
	class LSTM {
	public:
		struct Connection {
			float _weight;
		};

		struct Node {
			std::vector<Connection> _connections;

			Connection _bias;

			float _output;

			Node()
				: _output(0.0f)
			{}
		};

		struct MemoryNode {
			Node _inputGater;
			Node _forgetGater;
			Node _outputGater;
			Node _input;

			float _memory;
			float _output;

			MemoryNode()
				: _memory(0.0f), _output(0.0f)
			{}
		};

		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

	private:
		std::vector<float> _inputs;
		std::vector<Node> _hiddenNodes;
		std::vector<MemoryNode> _memoryNodes;
		std::vector<Node> _outputNodes;

	public:
		void createRandom(int numInputs, int numOutputs, int numHidden, int numMemory, float initWeightStdDev, std::mt19937 &generator);

		void setInput(int index, float value) {
			_inputs[index] = value;
		}

		float getInput(int index) const {
			return _inputs[index];
		}

		float getOutput(int index) const {
			return _outputNodes[index]._output;
		}

		int getNumInputs() const {
			return _inputs.size();
		}

		int getNumOutputs() const {
			return _outputNodes.size();
		}

		int getNumHidden() const {
			return _hiddenNodes.size();
		}

		int getNumMemory() const {
			return _memoryNodes.size();
		}
	};
}