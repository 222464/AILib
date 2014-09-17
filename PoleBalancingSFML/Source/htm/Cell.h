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

#include <htm/Segment.h>

namespace htm {
	class Cell {
	private:
		bool _activeState;
		bool _predictiveState;
		bool _learnState;

		bool _prevActiveState;
		bool _prevPredictiveState;
		bool _prevLearnState;

		int _numPredictionSteps;
		int _prevNumPredictionSteps;

		std::vector<Segment> _segments;

		std::vector<SegmentUpdate> _segmentUpdates;

	public:
		Cell()
			: _activeState(false), _predictiveState(false), _learnState(false),
			_prevActiveState(false), _prevPredictiveState(false), _prevLearnState(false),
			_numPredictionSteps(0), _prevNumPredictionSteps(0)
		{}

		bool isActive() const {
			return _activeState;
		}

		bool getOutput() const {
			return _activeState || _predictiveState;
		}

		friend class Column;
		friend class Region;
	};
}