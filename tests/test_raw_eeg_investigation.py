"""Unit and band-integration checks on synthetic signals; no EDF/model access."""

import unittest
import numpy as np
from src.raw_eeg_investigation import window_spectrum
from src.preprocess import causal_bandpass


class SpectrumTests(unittest.TestCase):
    def test_known_absolute_power_and_relative_gamma(self):
        fs=256
        t=np.arange(5*fs)/fs
        signal=20*np.sin(2*np.pi*2*t)+10*np.sin(2*np.pi*35*t)
        f,p,total,gamma=window_spectrum(signal[None,:],fs)
        self.assertEqual(f[1]-f[0],.5)
        self.assertAlmostEqual(total[0],250,places=8)
        self.assertAlmostEqual(gamma[0],50,places=8)
        self.assertAlmostEqual(gamma[0]/total[0],.2,places=8)

    def test_amplitude_scaling_changes_absolute_not_relative_power(self):
        t=np.arange(1280)/256
        signal=(np.sin(2*np.pi*2*t)+np.sin(2*np.pi*35*t))[None,:]
        _,_,a,b=window_spectrum(signal,256)
        _,_,c,d=window_spectrum(signal*1e6,256)
        np.testing.assert_allclose(c,a*1e12)
        np.testing.assert_allclose(d,b*1e12)
        np.testing.assert_allclose(b/a,d/c)

    def test_causal_prefix_equals_complete_filter_crop(self):
        signal=np.random.default_rng(42).normal(size=(2,2560))
        np.testing.assert_array_equal(causal_bandpass(signal,256)[:,:2000],causal_bandpass(signal[:,:2000],256))

    def test_incomplete_window_rejected(self):
        with self.assertRaises(ValueError): window_spectrum(np.zeros((1,100)),256)


if __name__=='__main__': unittest.main()
