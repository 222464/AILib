#pragma once

#include <vector>
#include <random>
#include <array>
#include <complex>

namespace mfcc {
	extern const float _pi;

	class MelFilterBank {
	public:
		static float hzToMels(float hz) {
			return 1125.0f * std::log(1.0f + hz / 700.0f);
		}

		static float melsToHz(float mels) {
			return 700.0f * (std::exp(mels / 1125.0f) - 1.0f);
		}
	private:
		int _numFilters;
		int _filterSize;

		std::vector<float> _bank;

	public:
		void create(int numFilters, int filterSize, float minFrequency, float maxFrequency, float sampleRate);

		float getValue(int fi, int si) const {
			return _bank[si + fi * _filterSize];
		}

		int getNumFilters() const {
			return _numFilters;
		}

		int getFilterSize() const {
			return _filterSize;
		}
	};

	class AudioFeatureMFCC {
	public:
		static float hamming(float x) {
			return 0.54f - 0.46f * std::cos(2 * _pi * x);
		}

		static float rhamming(float x) {
			return std::acos((0.54f - x) / 0.46f) / (2 * _pi);
		}

		static void dft(const std::vector<short> &samples, int start, int length, std::vector<std::complex<float>> &result);
		static void dct(const std::vector<float> &data, std::vector<float> &result);

		// Reverse
		static void rdft(const std::vector<std::complex<float>> &result, int start, int length, std::vector<short> &samples);
		static void rdct(const std::vector<float> &result, std::vector<float> &data);

	private:
		std::vector<float> _coeffs;

	public:
		void extract(const std::vector<short> &samples, int start, int length, const MelFilterBank &filterBank);

		void reverse(std::vector<short> &samples, int start, int length, const MelFilterBank &filterBank);

		float getCoeff(int i) const {
			return _coeffs[i];
		}

		int getNumCoeffs() const {
			return _coeffs.size();
		}
	};
}