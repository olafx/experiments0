'''
Get RGB color from light wavelength.

This requires a good bit of explanation.

Humans perceive color via three different cone types: S (short), M (middle), and L (long). A particular combination of
relative intensities will lead to seeing some subjective color. Each cone has a sensitivity function dependent on
wavelength. This means that a single wavelength, in general, is measured by all three cones. And so there exists a
function that maps wavelength onto intensities of the S, L, and M cones. Here the CIE 1931 color space is used to make
this transformation, with the standard analytical approximation to the cone sensitivity functions. This approximation
fits the sensitivity curves to a set of asymmetrical Gaussians (asymmetry caused by different variance on either side of
the mean). This gives the values X, Y, Z. (Nevermind their meaning.) There exists no 1 to 1 transformation from XYZ to
RGB. This is because RGB can't represent every color visible to the human eye, while XYZ can, since XYZ is directly
defined via the cone sensitivities. When an image is constructed from multiple wavelengths per pixel, it is possible
that the contribution of a particular wavelength yields an RGB values outside of the normal range. This is because, once
again, not every monochromatic wavelength that is visible by the human eye is representable by the RGB spectrum.
Sometimes the contributions at a single wavelength are negative even though the total contribution when integrated over
all wavelengths is positive. But sometimes, e.g. for monochromatic light, even the sum is outside of the RGB range, and
then the XYZ color does not map to a valid RGB color.

(The wavelength to S, M, L transformation is obviously not 1 to 1. There are many more combinations of S, M, L
intensities than there are wavelengths. This means essentially all colors humans can perceive can't be created by mono-
chromatic light, which is interesting. Obviously humans can't see all wavelengths, but humans can also see colors that
have no singular wavelength description. Weirder still, since cones can't know what particular wavelength of light
activated them, many different monochromatic combinations of light lead to the same perceived color.
Conclusion:
-   ~0% of the human perceivable colors can be made via monochromatic light.
-   ~0% of multichromatic light is perceivable as a unique color, so the eye can be tricked.
-   monochromatic light space << color space << multichromatic light space.)

Practically a RGB image can be created from spectrum data with the following algorithm:
1.  consider the wavelength range under which the wavelength -> XYZ transformation is defined, i.e. the visible spectrum
2.  iterate over every pixel
2.1     starts a RGB counter at 0
2.2     iterate over every wavelength in the consider range
2.2.1       problem dependent: calculate the intensity of this pixel at this wavelength
2.2.2       make the wavelength -> XYZ -> RGB transformation
2.2.3       add the RGB value to the RGB counter
2.3     normalize the RGB counter by dividing by the number of wavelengths considered
3.  (optional) correct invalid RGB values somehow, e.g. make anything that's negative a 0, or don't display those pixels

The CIE 1931 color space defines the necessary wavelength -> XYZ transformations, and defines the RGB -> XYZ
transformation. It does not provide a XYZ -> RGB transformation because this is inherently impossible, as XYZ is larger
than RGB, but it's acceptable in this application to just calculate the inverse transformation matrix, since negative
RGB values at a particular wavelength are not an issue. This does not mean the results are qualitatively bad, this is
perfectly normal; the CIE 1931 color space also defines a wavelength -> RGB function after all with negative
coefficients, but data and approximations for these functions seem hard to find in literature. The only potential
qualitative mistake that can be made is if a particular pixel unfortunately happened to have a XYZ value that is not
representable in RGB, so even when integrated over all wavelengths. Can't do anything about that.
'''

import numpy as np
import matplotlib.pyplot as plt

n = 10**3

# asymmetric Gaussian
def g(λ, μ, σ1, σ2):
    return np.exp(-.5*((λ-μ)/(σ1 if λ < μ else σ2))**2)

# CIE 1931 provided wavelength to XYZ approximation via asymmetric Gaussians
def λ_to_XYZ(λ):
    return np.array([1.056*g(λ, 599.8, 37.9, 31.0)+0.362*g(λ, 442.0, 16.0, 26.7)-0.065*g(λ, 501.1, 20.4, 26.2),
                     0.821*g(λ, 568.8, 46.9, 40.5)+0.286*g(λ, 530.9, 16.3, 31.1),
                     1.217*g(λ, 437.0, 11.8, 36.0)+0.681*g(λ, 459.0, 26.0, 13.8)])

# CIE 1931 provided RGB -> XYZ transformer
RGB_to_XYZ = 1/0.17697*np.array([[.49, .31, .2], [.17697, .81240, .01063], [0, .01, .99]])

XYZ_to_RGB = np.linalg.inv(RGB_to_XYZ)

λs = np.linspace(360, 830, n)
colors = np.empty([n, 3])
for i in range(n):
    colors[i] = np.matmul(XYZ_to_RGB, λ_to_XYZ(λs[i]))

plt.plot(λs, colors[:, 0], 'r', λs, colors[:, 1], 'g', λs, colors[:, 2], 'b')
plt.savefig('Airy_5.png', dpi=400)
