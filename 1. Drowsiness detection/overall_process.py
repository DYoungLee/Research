"""
Code Description:
    - Classifier model (SVM) training using acquired PPG data
    - Feature extraction (heart rate, RR interval, entropy)
    - Binary classification (awakening and drowsiness)
"""

import numpy as np
import peakutils
from sklearn.svm import SVC
from sklearn.model_selection import KFold
import pandas as pd
import preprocessing
from pyentrp.entropy import permutation_entropy as PE

class process_class:
    def __init__(self, sub_num):
        """Load PPG data

        Args:
           sub_num: number of subject
        """
        self.sub_num = sub_num
        ppg_D = pd.read_csv('./data/drowsiness/subject' + str(sub_num) + '.csv')
        ppg_N = pd.read_csv('./data/normal/subject' + str(sub_num) + '.csv')
        
        self.ppg_D_set = np.array(ppg_D).astype(np.float)
        self.ppg_N_set = np.array(ppg_N).astype(np.float)
        
        self.model = None
        
    ####################################################################################################################
    def feature_ext(self, signal, flag):
        """Feature extraction of PPG data

        Args:
           signal: PPG data
           flag: Identification for classifier training or real time prediction

        Returns:
            HR_set: Heart rate
            RR_set: RR interval
            mpe_set: Multiscale permutation entropy
        """
        ################################################################################################################
        # set parameters
        Fs = 64             # sampling frequency
        m = 4               # embedding dimension for PE
        t = 1               # time delay for PE
        num_entropy = 30    # the number of samples for PE
        w = Fs * 15         # window length
        HR_set = []         # list for heart rate feature
        RR_set = []         # list for RR interval feature
        PE_set = []         # list for heart rate feature

        if flag == 0:       # for classifier training
            sig = np.array(signal).astype(np.float)
            [R, L] = sig.shape
            num_window = round(L/w)

        else:               # for real time prediction
            sig = np.array(signal, dtype=np.float)[np.newaxis]
            [R, L] = sig.shape
            num_window = round(L/w)

        ################################################################################################################
        # feature extraction
        for i in range(0, R):
            for j in range(0, num_window):
                tmp_sig = sig[i, w * j:w * (j + 1)]
                # preprocessing (low pass filter)
                filtered = preprocessing.butter_lowpass_filter(tmp_sig, 2.5, 64.0, 5)
                # find peak point
                locs = peakutils.indexes(filtered, thres=0.4, min_dist=Fs * 0.5)

                for k in range(0, len(locs) - 1):
                    # calculate RR interval and heart rate
                    RR_set.append((locs[k + 1] - locs[k])/Fs)
                    HR_set.append(Fs / (locs[k + 1] - locs[k]) * 60)

        # Removing abnormal data: HR < 50, RR < 2 seconds
        HR_set = [i for i in HR_set if i>=50 and i<=120]
        RR_set = [i for i in RR_set if i<=2 and i>=0.5]

        # Calculate permutation entropy for RR interval (RR_set)
        # 50% overlapping
        q = 0
        while 1:
            if num_entropy+q*int(num_entropy/2) >= len(RR_set):
                tmp = RR_set[q*int(num_entropy/2):]
                PE_set.append(PE(tmp, m, t))
                break
            else:
                tmp = RR_set[q*int(num_entropy/2):num_entropy+q*int(num_entropy/2)]
                PE_set.append(PE(tmp, m, t))
                q += 1

        HR_set = np.array(HR_set, dtype=np.float)[np.newaxis]
        RR_set = np.array(RR_set, dtype=np.float)[np.newaxis]
        PE_set = np.array(PE_set, dtype=np.float)[np.newaxis]
        return HR_set, RR_set, PE_set
    
    ####################################################################################################################
    def plot_features(self, n=2):
        """
        Plot features (randomly selects the same number of features)
        Args:
            n: 2(HR, RR) / 3(HR, RR, PE)
        """
        
        markers = ('s', 'x')
        colors = ('red', 'blue')
        
        X1 = self.HR_d
        Y1 = self.RR_d
        Z1 = self.pe_d
        X1 = X1[:, np.random.choice(X1.shape[-1], Z1.shape[-1])]
        Y1 = Y1[:, np.random.choice(Y1.shape[-1], Z1.shape[-1])]
        
        X2 = self.HR_n
        Y2 = self.RR_n
        Z2 = self.pe_n
        X2 = X2[:, np.random.choice(X2.shape[-1], Z2.shape[-1])]
        Y2 = Y2[:, np.random.choice(Y2.shape[-1], Z2.shape[-1])]            

        fig = plt.figure()
        if n == 2:
            plt.scatter(X1, Y1,
                        alpha=0.8, 
                        c=colors[0],
                        marker=markers[0], 
                        label='drowsiness', 
                        edgecolor='black')
            plt.scatter(X2, Y2,
                        alpha=0.8, 
                        c=colors[1],
                        marker=markers[1], 
                        label='normal', 
                        edgecolor='black')
            
            plt.xlabel('Heart rate')
            plt.ylabel('RR interval')
            plt.legend(loc='upper right')
            plt.show()
            
        elif n == 3:
            ax = fig.add_subplot(111, projection='3d')            
            ax.scatter(X1, Y1, Z1,
                        alpha=0.8, 
                        c=colors[0],
                        marker=markers[0], 
                        label='drowsiness', 
                        edgecolor='black')

            ax.scatter(X2, Y2, Z2,
                        alpha=0.8, 
                        c=colors[1],
                        marker=markers[1], 
                        label='normal', 
                        edgecolor='black')

            ax.set_xlabel('Heart rate')
            ax.set_ylabel('RR interval')
            ax.set_zlabel('Permutation entropy')
            plt.legend(loc='upper right')
            plt.show()
            
        else:
            return
        
    ####################################################################################################################
    def normalization(self, data, M_o, m_o, M, m):
        """Normalization (Min-Max Feature scaling)

        Args:
            data: feature
            M_o: data max
            m_o: data min
            M: normalization max
            m: normalization min

        Returns:
            Normalized feature
        """
        return (data - m_o)/(M_o - m_o) * (M-m) + m

    ####################################################################################################################
    def training(self):
        """SVM model training and test

        Returns:
            SVM model
        """
        ################################################################################################################
        # feature extraction
        [self.HR_n, self.RR_n, self.pe_n] = self.feature_ext(self.ppg_N_set, 0)
        [self.HR_d, self.RR_d, self.pe_d] = self.feature_ext(self.ppg_D_set, 0)

        ################################################################################################################
        # Segmentation and average
        L_n = pe_n.shape[1]  # Length for average of normal
        L_d = pe_d.shape[1]  # Length for average of drowsiness

        HR_n_t = np.transpose(preprocessing.average(HR_n, L_n))
        HR_n_t = self.normalization(HR_n_t, 120, 50, 1, 0)
        RR_n_t = np.transpose(preprocessing.average(RR_n, L_n))
        RR_n_t = self.normalization(RR_n_t, 2, 0.5, 1, 0)
        pe_n_t = np.transpose(pe_n)

        HR_d_t = np.transpose(preprocessing.average(HR_d, L_d))
        HR_d_t = self.normalization(HR_d_t, 120, 50, 1, 0)
        RR_d_t = np.transpose(preprocessing.average(RR_d, L_d))
        RR_d_t = self.normalization(RR_d_t, 2, 0.5, 1, 0)
        pe_d_t = np.transpose(pe_d)

        L_n = np.min([L_n, len(HR_n_t), len(RR_n_t)])
        L_d = np.min([L_d, len(HR_d_t), len(RR_d_t)])

        X = np.zeros((L_n+L_d, 3))
        X[:L_n, 0:1] = HR_n_t[:L_n, :]
        X[L_n:, 0:1] = HR_d_t[:L_d, :]
        X[:L_n, 1:2] = RR_n_t[:L_n, :]
        X[L_n:, 1:2] = RR_d_t[:L_d, :]
        X[:L_n, 2:3] = pe_n_t[:L_n, :]
        X[L_n:, 2:3] = pe_d_t[:L_d, :]
        
        Y = np.ones(L_n+L_d, )
        Y[L_n:] = 0

        ################################################################################################################
        # SVM model for online process
        # svm_model = SVC(kernel='linear', C=1.0, random_state=0).fit(X, Y)
        svm_model = SVC(kernel='rbf', C=1.0, gamma=2).fit(X, Y)
        self.model = svm_model
        
        ################################################################################################################
        # 5-fold cross validation for offline test
        acc = []
        cv = KFold(5, shuffle=True, random_state=0)
        for i, (idx_train, idx_test) in enumerate(cv.split(X)):
            input_x = X[idx_train]
            input_y = Y[idx_train]
            test_x = X[idx_test]
            test_y = Y[idx_test]

            # svm_model_test = SVC(kernel='linear', C=1.0, random_state=0).fit(input_x, input_y)
            svm_model_test = SVC(kernel='rbf', C=1.0, gamma=5).fit(input_x, input_y)

            pre = svm_model_test.predict(test_x)
            acc.append(len([i for i in range(len(pre)) if pre[i] == test_y[i]])/len(pre))
            print('%d fold: %.3f' % (i+1, acc[i]))

        print('Average accuracy: %.3f' % (np.average(acc)))

        return svm_model

if __name__ == '__main__':
    sub_num = 2
    DrowsinessDect = process_class(sub_num)
    DrowsinessDect.training()
    DrowsinessDect.plot_features(n=3)
