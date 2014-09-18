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

#include <htm/Connection.h>

#include <vector>
#include <unordered_map>

namespace htm {
	struct SegmentUpdate {
		enum Source {
			_1, _2, _3, _null
		};

		Source _src;

		int _columnIndex;
		int _cellIndex;
		int _segmentIndex;
		std::vector<ColumnAndCellIndices> _activeConnectionIndices;
		std::vector<ColumnAndCellIndices> _inactiveConnectionIndices;

		bool _isNew;
		bool _dueToPredictive;
		int _numPredictionSteps;

		SegmentUpdate()
			: _columnIndex(-1), _cellIndex(-1), _segmentIndex(-1),
			_isNew(true), _dueToPredictive(true), _numPredictionSteps(1),
			_src(_null)
		{}

		size_t operator()(const SegmentUpdate &value) const {
			return static_cast<size_t>(_columnIndex ^ _cellIndex ^ _segmentIndex);
		}

		bool operator==(const SegmentUpdate &other) const {
			return _columnIndex == other._columnIndex && _cellIndex == other._cellIndex && _segmentIndex == other._segmentIndex;
		}
	};

	class Segment {
	private:
		std::unordered_map<ColumnAndCellIndices, Connection, ColumnAndCellIndices> _connections;

		int _activeActivity;
		int _learnActivity;

		int _prevActiveActivity;
		int _prevLearnActivity;

		int _numPredictionSteps;

		bool _sequenceSegment;

	public:
		Segment()
			: _activeActivity(0), _learnActivity(0),
			_prevActiveActivity(0), _prevLearnActivity(0),
			_numPredictionSteps(-1),
			_sequenceSegment(false)
		{}

		friend class Cell;
		friend class Column;
		friend class Region;
	};
}