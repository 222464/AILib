#include <featureExtraction/AudioFeatureMFCC.h>

#include <assert.h>

using namespace mfcc;

const float mfcc::_pi = 3.14159f;

void MelFilterBank::create(int numFilters, int filterSize, float minFrequency, float maxFrequency, float sampleRate) {
	_numFilters = numFilters;
	_filterSize = filterSize;

	float minMels = hzToMels(minFrequency);
	float maxMels = hzToMels(maxFrequency);

	std::vector<float> m(numFilters + 2);

	float numFiltersMinusOneInv = 1.0f / (numFilters + 1);

	for (int n = 0; n < m.size(); n++)
		m[n] = minMels + (maxMels - minMels) * n * numFiltersMinusOneInv;
	
	std::vector<float> h(numFilters + 2);

	for (int n = 0; n < h.size(); n++)
		h[n] = melsToHz(m[n]);

	std::vector<int> f(numFilters + 2);

	float sampleRateInv = 1.0f / sampleRate;

	for (int n = 0; n < f.size(); n++)
		f[n] = std::floor(filterSize * h[n] * sampleRateInv);

	_bank.resize(numFilters * filterSize);

	for (int fi = 0; fi < numFilters; fi++) {
		float fAtFiMinusOne = f[fi];
		float fAtFiPlusZero = f[fi + 1];
		float fAtFiPlusOne = f[fi + 2];

		for (int si = 0; si < filterSize; si++) {
			if (si < fAtFiMinusOne)
				_bank[si + fi * filterSize] = 0.0f;
			else if (fAtFiMinusOne <= si && si <= fAtFiPlusZero)
				_bank[si + fi * filterSize] = (si - fAtFiMinusOne) / (fAtFiPlusZero - fAtFiMinusOne);
			else if (fAtFiPlusZero <= si && si <= fAtFiPlusOne)
				_bank[si + fi * filterSize] = (fAtFiPlusOne - si) / (fAtFiPlusOne - fAtFiPlusZero);
			else
				_bank[si + fi * filterSize] = 0.0f;
		}
	}
}

void AudioFeatureMFCC::dft(const std::vector<short> &samples, int start, int length, std::vector<std::complex<float>> &result) {
	if (result.size() != length)
		result.resize(length);

	const float sizeInv = 1.0f / 64.0f;
	
	float lengthInv = 1.0f / length;

	// Sweep "length" number of frequencies
	for (int k = 0; k < length; k++) {
		float realSum = 0.0f;
		float imaginarySum = 0.0f;

		for (int n = 0; n < length; n++) {
			float sample = samples[start + n] * sizeInv;

			float h = hamming(n * lengthInv);

			float rads = 2.0f * _pi * k * n * lengthInv;
			
			realSum += sample * h * std::cos(rads);
			imaginarySum += sample * h * std::sin(rads);
		}

		result[k] = std::complex<float>(realSum, imaginarySum);
	}
}

void AudioFeatureMFCC::dct(const std::vector<float> &data, std::vector<float> &result) {
	if (result.size() != data.size())
		result.resize(data.size());

	float lengthInv = 1.0f / data.size();

	// Sweep "length" number of frequencies
	for (int k = 0; k < data.size(); k++) {
		float sum = 0.0f;

		for (int n = 0; n < data.size(); n++) {
			float sample = data[n];

			sum += sample * std::cos(_pi * lengthInv * (n + 0.5f) * k);
		}

		result[k] = sum;
	}
}

void AudioFeatureMFCC::rdft(const std::vector<std::complex<float>> &result, int start, int length, std::vector<short> &samples) {
	const float size = 64.0f;
	const float sizeInv = 1.0f / 64.0f;

	float lengthInv = 1.0f / length;

	for (int n = 0; n < length; n++) {
		int i = start + n;

		float h = hamming(n * lengthInv);

		float hInv = 1.0f / h;

		float sum = 0.0f;

		// Add frequencies
		for (int k = 0; k < length; k++) {
			float rads = 2.0f * _pi * k * n * lengthInv;

			sum += (std::cos(rads) / result[k].real() + std::sin(rads) / result[k].imag()) * hInv;
		}

		samples[i] = sum * size;
	}
}

void AudioFeatureMFCC::rdct(const std::vector<float> &result, std::vector<float> &data) {
	float lengthInv = 1.0f / data.size();

	for (int n = 0; n < data.size(); n++) {
		float sum = 0.0f;

		// Add frequencies
		for (int k = 0; k < data.size(); k++) {
			float rads = 2.0f * _pi * k * n * lengthInv;

			sum += std::cos(_pi * lengthInv * (n + 0.5f) * k) / result[k];
		}

		data[n] = sum;
	}
}

void AudioFeatureMFCC::extract(const std::vector<short> &samples, int start, int length, const MelFilterBank &filterBank) {
	assert(filterBank.getFilterSize() == length);
	
	// Perform discrete Fourier transform
	std::vector<std::complex<float>> dftResult;

	dft(samples, start, length, dftResult);

	// Compute periodogram
	std::vector<float> periodogram(length);

	float lengthInv = 1.0f / length;

	for (int n = 0; n < length; n++) {
		float absVal = std::abs(dftResult[n]);

		periodogram[n] = lengthInv * absVal * absVal;
	}

	// Multiply by the filter banks
	std::vector<float> energies(filterBank.getNumFilters());

	for (int n = 0; n < filterBank.getNumFilters(); n++) {
		float sum = 0.0f;

		for (int k = 0; k < length; k++)
			sum += filterBank.getValue(n, k) * periodogram[k];

		energies[n] = sum;
	}

	std::vector<float> logEnergies(filterBank.getNumFilters());

	for (int n = 0; n < filterBank.getNumFilters(); n++)
		logEnergies[n] = std::log(energies[n]);

	// Perform discrete cosine transform
	dct(logEnergies, _coeffs);
}

void AudioFeatureMFCC::reverse(std::vector<short> &samples, int start, int length, const MelFilterBank &filterBank) {
	// Reverse dct
	std::vector<float> logEnergies(filterBank.getNumFilters());

	rdct(_coeffs, logEnergies);

	// Reverse log
	std::vector<float> energies(filterBank.getNumFilters());

	for (int n = 0; n < filterBank.getNumFilters(); n++)
		energies[n] = std::exp(logEnergies[n]);

	// Divide by filter bank results
	std::vector<float> periodogram(length);

	for (int k = 0; k < length; k++) {
		float sum = 0.0f;

		for (int n = 0; n < filterBank.getNumFilters(); n++) {
			float filter = filterBank.getValue(n, k);

			sum += energies[n] * filter;
		}

		periodogram[k] = sum;
	}

	// Get DFT result
	std::vector<std::complex<float>> dftResult(length);

	float lengthInv = 1.0f / length;

	const float root2Over2 = std::sqrt(2.0f) * 0.5f;

	for (int n = 0; n < length; n++) {
		float absVal = std::sqrt(periodogram[n] * length);

		dftResult[n] = std::complex<float>(root2Over2 * absVal, root2Over2 * absVal);
	}

	// Reverse DFT
	rdft(dftResult, start, length, samples);
}