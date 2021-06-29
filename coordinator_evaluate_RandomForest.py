
import function_Evaluate_RandomForest
import function_Evaluate_NaiveBayes


# print(function_Evaluate_RandomForest.runModel(False, 'average4', 'all', False, False, 10))
# print(function_Evaluate_NaiveBayes.runModel(False, 'average4', 'all', False, False, 10))

data_params = { 'mean_value_subtraction': False,
                    'data_resampling': 'average4', #options include 'last, 'average4', 'use4' and 'use7'
                    'features_selected': 'all', #options include 'all', '1070' and '4050'
                    'standard_scaler':False, #options include True and False
                    'PCA_reduction': False,
                    'PCA_number_of_features': 10
                    }


A = [True, False]
B = ['last', 'use4', 'use7', 'average4']
C = ['all', '1070', '4050']
D = [True, False]
E = [True, False]
f = 10
i = 0

for a in A:
    for b in B:
        for c in C:
            for d in D:
                for e in E:
                    print(f"{i}: {a} {b} {c} {d} {e} {f}")
                    print(function_Evaluate_NaiveBayes.runModel(a,b,c,d,e,f))
                    i=i+1


for a in A:
    for b in B:
        for c in C:
            for d in D:
                for e in E:
                    print(f"{i}: {a} {b} {c} {d} {e} {f}")
                    print(function_Evaluate_RandomForest.runModel(a,b,c,d,e,f))
                    i=i+1

print(len(A)*len(B)*len(C)*len(D)*len(E))