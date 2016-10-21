import numpy as np
from scipy.misc import imresize


def Bfx_fourier(I, *args):
    '''
    % [X,Xn,Xu] = Xfourier(I,R,options)
    % [X,Xn,Xu] = Xfourier(I,options)
    %
    % Toolbox Xvis: Fourier features
    %
    %    X is the features vector, Xn is the list of feature names (see Example
    %    to see how it works).
    %
    %   Example:
    %      options.Nfourier  = 64;                % imresize vertical
    %      options.Mfourier  = 64;                % imresize horizontal
    %      options.mfourier  = 2;                 % imresize frequency vertical
    %      options.nfourier  = 2;                 % imresize frequency horizontal
    %      options.show    = 1;                   % display results
    %      I = imread('testimg1.jpg');            % input image
    %      R = Bim_segbalu(I);                    % segmentation
    %      J = double(I(:,:,2))/256;              % normalized green channel
    %      [X,Xn] = Xfourier(J,R,options);     % Fourier features
    %      Bio_printfeatures(X,Xn)
    %
    %   See also Xharalick, Xclp, Xgabor, Xdct, Xlbp.
    %
    % (c) D.Mery, PUC-DCC, 2010
    % http://dmery.ing.puc.cl
    '''

    assert len(args) > 0, "Please provide at least the 'options' argument"

    if len(args) == 1:
        R = np.ones(I.shape)
        options = args[0]
    else:
        R = args[0]
        options = args[1]

    I[np.where(R == 0)] = 0

    N = options['Nfourier']
    M = options['Mfourier']
    n = options['nfourier']
    m = options['mfourier']

    N2 = np.round(N / 2.0)
    M2 = np.round(M / 2.0)

    if 'show' in options and options['show'] is True:
        print('--- extracting Fourier features...')

    Im = imresize(I.astype(float), (N, M))
    FIm = np.fft.fft2(Im)
    x = np.abs(FIm)
    F = imresize(x[0:N2, 0:M2], (n, m))
    x = np.angle(FIm)
    A = imresize(x[0:N2, 0:M2], (n, m))

    LS = 2 * n * m
    X = np.zeros((1, LS))
    Xn = LS * [None]
    k = -1
    for i in range(n):  # i=1:n
        for j in range(m):  # j=1:m
            k += 1
            # s = sprintf('Fourier Abs (%d,%d)              ', i, j)
            # Xn(k,:)  = s(1:24)
            Xn[k] = 'Fourier Abs ({}, {})'.format(i, j)
            X[0, k] = F[i, j]

    for i in range(n):
        for j in range(m):
            k += 1
            # s = sprintf('Fourier Ang (%d,%d)[rad]         ', i, j)
            # Xn(k,:)  = s(1:24)
            Xn[k] = 'Fourier Ang ({}, {})[rad]'.format(i, j)
            X[0, k] = A[i, j]

    return X, Xn

    '''if nargin==2;
        options = R;
        R = ones(size(I));
    end

    I(R==0) = 0;

    N  = options.Nfourier;
    M  = options.Mfourier;
    n  = options.nfourier;
    m  = options.mfourier;

    N2 = round(N/2);
    M2 = round(M/2);

    if options.show
        disp('--- extracting Fourier features...');
    end


    Im  = imresize(double(I),[N M]);
    FIm = fft2(Im);
    x   = abs(FIm);
    F   = imresize(x(1:N2,1:M2),[n m]);
    x   = angle(FIm);
    A   = imresize(x(1:N2,1:M2),[n m]);

    LS = 2*n*m;
    X  = zeros(1,LS);
    Xn = char(zeros(LS,24));
    k = 0;
    for i=1:n
        for j=1:m
            k = k + 1;
            s = sprintf('Fourier Abs (%d,%d)              ',i,j);
            Xn(k,:)  = s(1:24);
            X(k) = F(i,j);
        end
    end
    for i=1:n
        for j=1:m
            k = k + 1;
            s = sprintf('Fourier Ang (%d,%d)[rad]         ',i,j);
            Xn(k,:)  = s(1:24);
            X(k) = A(i,j);
        end
    end'''
