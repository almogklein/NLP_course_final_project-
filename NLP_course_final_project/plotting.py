import matplotlib.pyplot as plt
import numpy as np

class plotter():
    def __init__(self, precisionC, recallC, f1C, accC, precisionH, recallH, f1H, accH):
        self.precisionC = precisionC
        self.recallC = recallC
        self.f1C = f1C
        self.accC = accC
        self.precisionH = precisionH
        self.recallH = recallH
        self.f1H = f1H
        self.accH = accH

    def plotM(self):
        names = ['Accuracy', 'F1_core', 'Precision', 'Recall']
        valuesC = [self.accC, self.f1C, self.precisionC, self.recallC]
        valuesH = [self.accH, self.f1H, self.precisionH, self.recallH]

        x = np.arange(len(names))  # the label locations
        width = 0.35

        fig, ax = plt.subplots()
        CRFbar = ax.bar(x - width / 2, valuesC, width, label='CRF')
        HMMbar = ax.bar(x + width / 2, valuesH, width, label='HMM')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Percentage')
        ax.set_title('Comparison of model performance for language analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.legend()

        ax.bar_label(CRFbar, padding=3)
        ax.bar_label(HMMbar, padding=3)

        fig.tight_layout()

        plt.show()
