
import function_Evaluate_CNN

# print(function_Evaluate_CNN.runModel(False, 'use7', 'all', False, True))
# print(function_Evaluate_CNN.runModel(True, 'use7', 'all', False, True))

# data_params = { 'mean_value_subtraction': False, #A
#                 'data_resampling': 'average4', #B options include 'last, 'average4', 'use4' and 'use7'
#                 'features_selected': 'all', #C options include 'all', '1070' and '4050'
#                 'standard_scaler':False, #D options include True and False
#                 'binary classifier: False, #E options include True and False
#                }


A = [True, False]
B = ['use7']#,'use4', 'average4', 'last', ]
C = ['all', '1070', '4050']
D = [False]
E = [True, False]
i = 0

for a in A:
    for b in B:
        for c in C:
            for d in D:
                for e in E:
                        print(f"{i}: {a} {b} {c} {d} {e} ")
                        print(function_Evaluate_CNN.runModel(a,b,c,d,e))
                        i=i+1


# for a in A:
#     for b in B:
#         for c in C:
#             for d in D:
#                 for e in E:
#                         print(f"{i}: {a} {b} {c} {d} {e} {f} {g}")
#                         print(function_Evaluate_LSTM.runModel(a,b,c,d,e,f,g))
#                         i=i+1

# print(len(A)*len(B)*len(C)*len(D)*len(E))