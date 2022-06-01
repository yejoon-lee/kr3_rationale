# train end-task with extracted rationales
# Implementation of FRESH(Faithful Rationale Extraction from Saliency tHresholding) from Jain et al., 2020

class Run:
    def __init__(self, 
                rationale_path : str):
        self.rationale_path = rationale_path

    def make(self):
        