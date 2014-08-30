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

#include <deep/DSOM.h>
#include <deep/FA.h>

namespace deep {
	class DeepSOMNet {
	private:
		std::vector<float> _inputs;
		std::vector<float> _outputs;

		std::vector<std::vector<float>> _layerOutputs;

	public:
		std::vector<DSOM> _SOMChain;

		FA _fa;

		void createRandom(int numInputs, int numOutputs, int SOMChainSize, int startNumDimensions, int numDimensionsLessPerSOM, int SOMSize, int FANumHidden, int FANumPerHidden, float minSOMWeight, float maxSOMWeight, float minFAWeight, float maxFAWeight, std::mt19937 &generator);
	
		void activate();
		void activateAndLearn(const std::vector<float> &targets, float FAAlpha);

		void setInput(size_t index, float value) {
			_inputs[index] = value;
		}

		float getInput(size_t index) const {
			return _inputs[index];
		}

		float getOutput(size_t index) const {
			return _outputs[index];
		}

		size_t getNumInputs() const {
			return _inputs.size();
		}

		size_t getNumOutputs() const {
			return _outputs.size();
		}
	};
}