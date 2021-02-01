from skimage import io, color, transform, data
from EdgesDetector import EdgesDetector


''' Vamos assumir que buscamos detectar
    bordas em imagens. Usaremos como exemplo a 
    imagem de uma parede de tijolos: '''


img = data.brick()
edgeDetector = EdgesDetector(img)
edgeDetector.plotHorizontalKernel()
edgeDetector.plotVerticalKernel()
edgeDetector.convolveImg(kernel='horizontal')
edgeDetector.plotFeatureMap()
edgeDetector.convolveImg(kernel='vertical')
edgeDetector.plotFeatureMap()
