import hmmModule
import crfModule
import plotting


if __name__ == '__main__':
    hmm = hmmModule.hmmClinet()
    crf = crfModule.crfClinet()
    precisionH, recallH, f1H, accH = hmm.start()
    precisionC, recallC, f1C, accC = crf.start()

    chp = plotting.plotter(precisionC, recallC, f1C, accC, precisionH, recallH, f1H, accH)
    chp.plotM()




