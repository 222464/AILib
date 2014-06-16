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

#include <istream>
#include <ostream>

namespace hn {
	class Config {
	private:
	public:
		int _numInputGroups;
		int _numInputsPerGroup;
		int _numOutputGroups;
		int _numOutputsPerGroup;

		float _boidConnectionRadius;

		float _initLinkChance;

		int _boidNumOutputs;
		int _boidFiringNumHiddenLayers;
		int _boidFiringHiddenSize;
		int _boidMemorySize;

		int _boidConnectNumHiddenLayers;
		int _boidConnectHiddenSize;

		int _boidDisconnectNumHiddenLayers;
		int _boidDisconnectHiddenSize;

		int _linkResponseSize;
		int _linkNumHiddenLayers;
		int _linkHiddenSize;
		int _linkMemorySize;

		int _encoderMemorySize;
		int _decoderMemorySize;
		int _encoderNumHiddenLayers;
		int _decoderNumHiddenLayers;
		int _encoderNumPerHidden;
		int _decoderNumPerHidden;

		float _initMemoryMin;
		float _initMemoryMax;

		float _initWeightMin;
		float _initWeightMax;

		float _boidOutputScalar;
		float _linkResponseScalar;

		int _maxBoidsPerInput;
		int _maxBoidsPerOutput;

		void writeToStream(std::ostream &os) const;
		void readFromStream(std::istream &is);

		Config();
	};
}