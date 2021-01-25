from VOC import VOC

def main():
    voc = VOC()
    voc.printDatasetInfo()
    voc.plotDataSetSample()
    voc.printTargetSample()
    voc.plotSampleBoundingBox()

if __name__ == "__main__":
    main()