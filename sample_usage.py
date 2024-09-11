from megaprofiler import MegaProfiler as M
import yfinance as yf


symbol = "AAPL"

ticker = yf.Ticker(symbol)
data = ticker.history(period="1y")

print("=============================================================================")

print("\nBASIC PROFILE ANALYSIS : \n")
basic_analysis = M.basic_profile_analysis(data)
print(basic_analysis)

print("=============================================================================")

print("\nPEARSON CORRELATION ANALYSIS : \n")
correlation = M.pearson_correlation_analysis(data)
print(correlation)

print("=============================================================================")

print("\nCOVARIANCE ANALYSIS : \n")
covariance = M.covariance_analysis(data)
print(covariance)

print("=============================================================================")

outliers = M.zscore_outlier_analysis(data)
print("\nZ-SCORE OUTLIER TEST : \n")
print(outliers)

print("=============================================================================")

iqr_outliers = M.iqr_outlier_analysis(data)
print("\nIQR OUTLIER TEST : \n")
print(iqr_outliers)

print("=============================================================================")

cat_data_analysis = M.categorical_data_analysis(data)
print("\nCATEGORICAL DATA ANALYSIS : \n")
print(cat_data_analysis)

print("=============================================================================")

data_skewness = M.data_skewness(data)
print("\nDATA SKEWNESS ANALYSIS : \n")
print(data_skewness)

print("=============================================================================")

data_kurtosis = M.data_kurtosis(data)
print("\nDATA KURTOSIS ANALYSIS : \n")
print(data_kurtosis)

print("=============================================================================")

memory_usage_analysis = M.memory_usage_analysis(data)
print("\nMEMORY USAGE ANALYSIS : \n")
print(memory_usage_analysis)

print("=============================================================================")

pca = M.pca_analysis(data)
print("\nPCA ANALYSIS : \n")
print(pca)

print("=============================================================================")

if_analysis = M.isolation_forest_analysis(data)
print("\nISOLATION FOREST ANALYSIS : \n")
print(if_analysis)

print("=============================================================================")

multicollinearity_analysis = M.multicollinearity_analysis(data)
print("\nMULTICOLLINEARITY ANALYSIS : \n")
print(multicollinearity_analysis)

print("=============================================================================")

normality_test = M.normality_test(data)
print("\nSHAPIRO-WILK TEST : \n")
print(normality_test)

print("=============================================================================")

clusters = M.kmeans_clustering(data)
print("\nK-MEANS CLUSTERING : \n")
print(clusters)

print("=============================================================================")
