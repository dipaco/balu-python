# -*- coding: utf-8 -*-
import numpy as np


def Bft_sigmoid(sc, d, show=False):
    """ function [param]=fitSigmoid2Scores(posFeats, negFeats, classifier, display)
     original de A.Soto
     nPosExamples=size(posFeats,1);
     nNegExamples=size(negFeats,1);

     [predict_label, accuracy, posScores]=svmpredict31EchoOff(ones(nPosExamples,1), posFeats, classifier);
     [predict_label, accuracy, negScores]=svmpredict31EchoOff(ones(nNegExamples,1), negFeats, classifier);

     Modificado por D.Mery

     With collaboration from:
     Diego Patiño (dapatinoco@unal.edu.co) -> Translated implementation into python (2016)
    """

    '''if d.size > 2:
        dneg = np.amin(d);
        dpos = np.amax(d);

        posScores = sc(d == dpos);
        negScores = sc(d == dneg);

        nPosExamples = length(posScores);
        nNegExamples = length(negScores);

        x = [negScores;
        posScores];

        y = [zeros(nNegExamples, 1);
        ones(nPosExamples, 1)];

        param = nlinfit(x, y, @ fsig, [-1 0.05]);'''


    return 0