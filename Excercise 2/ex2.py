import numpy as np

fxya = np.random.normal(2 * 1, 1, 100)
fxyb = np.random.normal(2 * 2, 1, 100)
fxyc = np.random.normal(2 * 3, 1, 100)

pxya = fxya / (fxya + fxyb + fxyc)
pxyb = fxyb / (fxya + fxyb + fxyc)
pxyc = fxyc / (fxya + fxyb + fxyc)

print pxya
print pxyb
print pxyc